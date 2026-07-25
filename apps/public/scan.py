"""
The public gate kiosk: the unmanned "computer screen" at the entry/exit lane.

FLOW (mirrors a China LPR gate, but the image is uploaded instead of camera-fed):
  - Entry: driver uploads the plate photo → screen shows the plate + welcome.
  - Exit:  driver uploads the plate photo → screen shows the plate + the charge
           (billed to the linked account for a registered plate, or amount due for
           a guest).

TRUST BOUNDARY:
  Anyone can reach the kiosk shell, but mutation requires an activated browser
  capability bound to one lot and direction plus a one-time request nonce.
  Responses remain privacy-reduced and never include owner identity or balance.

  Every hardened upload guard (size/format/dimension checks, private storage,
  decode safety) is inherited unchanged from scan_core.run_plate_scan().
"""

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps

from django.conf import settings
from django.db import DatabaseError, transaction
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView

from apps.dashboard.scan_core import ScanOutcome, run_plate_scan
from apps.dashboard.utils import confidence_band
from apps.parking.models import (
    KioskDeviceCapability,
    KioskImageReplayDigest,
    ParkingLot,
)

from .ratelimit import rate_limit

logger = logging.getLogger("apps.public")


@dataclass(frozen=True)
class KioskRequestDecision:
    """
    Result of validating one kiosk request under the capability row lock.

    WHY a structured result: a stale one-time nonce is recoverable for an
    otherwise valid capability, while a missing, expired, or wrongly scoped
    capability must remain a plain rejection with no replacement credential.
    """

    accepted: bool
    next_nonce: str
    image_digest: str = ""


class KioskView(TemplateView):
    """Render the public gate kiosk shell (entry/exit selector + image upload)."""

    template_name = "public/kiosk.html"

    def get_context_data(self, **kwargs):
        """Expose only the activation choices or the server-bound kiosk scope."""
        context = super().get_context_data(**kwargs)
        context["lots"] = list(ParkingLot.objects.order_by("name"))
        capability = _valid_capability(self.request)
        context["kiosk_activated"] = capability is not None
        if capability is not None:
            context["kiosk_lot"] = capability.lot
            context["kiosk_event_type"] = capability.event_type
            context["kiosk_scan_nonce"] = _issue_kiosk_nonce(self.request)
        return context


def _kiosk_token_fingerprint(token: str) -> str:
    """Bind activation to the current token without storing that token in session."""
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _kiosk_session_is_valid(request: HttpRequest) -> bool:
    """Reject absent capabilities and sessions made stale by token rotation."""
    return _valid_capability(request) is not None


def _valid_capability(request: HttpRequest) -> KioskDeviceCapability | None:
    """Return the active DB capability bound to this Django session, if any."""
    session_key = request.session.session_key
    configured_token = settings.KIOSK_ACTIVATION_TOKEN
    if not session_key or not configured_token:
        return None
    capability = (
        KioskDeviceCapability.objects.select_related("lot")
        .filter(session_key=session_key, expires_at__gt=timezone.now())
        .first()
    )
    if capability is None or not secrets.compare_digest(
        capability.token_fingerprint,
        _kiosk_token_fingerprint(configured_token),
    ):
        return None
    return capability


def _nonce_hash(nonce: str) -> str:
    """Store one-use nonces as digests so database disclosure cannot replay them."""
    return hashlib.sha256(nonce.encode("utf-8")).hexdigest()


def _uploaded_image_digest(request: HttpRequest) -> str:
    """Hash uploaded bytes and rewind them for validation/storage downstream."""
    upload = request.FILES.get("image")
    if upload is None:
        return ""
    digest = hashlib.sha256()
    try:
        for chunk in upload.chunks():
            digest.update(chunk)
        upload.seek(0)
    except (OSError, ValueError):
        logger.exception("Could not hash kiosk upload for replay detection")
        return ""
    return digest.hexdigest()


def _issue_kiosk_nonce(request: HttpRequest) -> str:
    """Replace the current nonce while holding the device capability row lock."""
    nonce = secrets.token_urlsafe(32)
    with transaction.atomic():
        capability = KioskDeviceCapability.objects.select_for_update().get(
            session_key=request.session.session_key
        )
        capability.nonce_hash = _nonce_hash(nonce)
        capability.save(update_fields=["nonce_hash"])
    request.session["kiosk_scan_nonce"] = nonce
    return nonce


def _consume_kiosk_request(request: HttpRequest) -> KioskRequestDecision | None:
    """
    Atomically validate scope/nonce and return the locked request decision.

    The image digest is recorded here so concurrent identical uploads lose the
    race under the same row lock. Callers must forget the digest when the scan
    fails before a durable outcome, so the lane can retry the same photo.
    """
    submitted_event_type = (request.POST.get("event_type") or "").strip().lower()
    submitted_lot = (request.POST.get("lot") or "").strip()
    submitted_nonce = request.POST.get("kiosk_nonce", "")
    image_digest = _uploaded_image_digest(request)
    configured_fingerprint = _kiosk_token_fingerprint(
        settings.KIOSK_ACTIVATION_TOKEN
    )
    next_nonce = secrets.token_urlsafe(32)
    now = timezone.now()

    with transaction.atomic():
        capability = (
            KioskDeviceCapability.objects.select_for_update()
            .select_related("lot")
            .filter(
                session_key=request.session.session_key,
                expires_at__gt=now,
            )
            .first()
        )
        if capability is None:
            return None
        valid_scope = (
            secrets.compare_digest(
                capability.token_fingerprint,
                configured_fingerprint,
            )
            and (
                not submitted_event_type
                or submitted_event_type == capability.event_type
            )
            and (not submitted_lot or submitted_lot == capability.lot.name)
        )
        if not valid_scope:
            return None
        valid_nonce = bool(submitted_nonce) and secrets.compare_digest(
            _nonce_hash(submitted_nonce),
            capability.nonce_hash,
        )
        if not valid_nonce:
            # Only an active, correctly scoped capability reaches this branch.
            # Rotate under the same row lock so a lost prior response receives
            # one replacement without reviving expired/deleted capabilities.
            capability.nonce_hash = _nonce_hash(next_nonce)
            capability.save(update_fields=["nonce_hash"])
            request.session["kiosk_scan_nonce"] = next_nonce
            return KioskRequestDecision(
                accepted=False,
                next_nonce=next_nonce,
            )
        replay_cutoff = now - timedelta(
            seconds=settings.KIOSK_IMAGE_REPLAY_SECONDS
        )
        capability.replay_digests.filter(seen_at__lt=replay_cutoff).delete()
        if image_digest and capability.replay_digests.filter(
            digest=image_digest
        ).exists():
            logger.warning("Rejected repeated kiosk image within replay window")
            return None
        capability.nonce_hash = _nonce_hash(next_nonce)
        capability.save(update_fields=["nonce_hash"])
        if image_digest:
            KioskImageReplayDigest.objects.create(
                capability=capability,
                digest=image_digest,
                seen_at=now,
            )
    request.session["kiosk_scan_nonce"] = next_nonce
    return KioskRequestDecision(
        accepted=True,
        next_nonce=next_nonce,
        image_digest=image_digest,
    )


def _forget_image_digest(request: HttpRequest, image_digest: str) -> None:
    """Drop a digest recorded for a scan that never produced a durable outcome."""
    if not image_digest:
        return
    session_key = request.session.session_key
    if not session_key:
        return
    try:
        with transaction.atomic():
            capability = (
                KioskDeviceCapability.objects.select_for_update()
                .filter(session_key=session_key)
                .first()
            )
            if capability is None:
                return
            capability.replay_digests.filter(digest=image_digest).delete()
    except DatabaseError:
        # A missed release self-heals via the replay-window purge; do not mask
        # the original scan error response with a secondary DB failure.
        logger.exception(
            "Could not release kiosk image replay digest after a failed scan"
        )


def _attach_nonce(response: HttpResponse, nonce: str) -> HttpResponse:
    """Return the already-rotated next nonce to the kiosk client."""
    response["X-Kiosk-Nonce"] = nonce
    return response


def kiosk_session_required(view):
    """Stop unauthorized scans before request data reaches storage or inference."""

    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not _kiosk_session_is_valid(request):
            return JsonResponse({"error": "Kiosk activation required."}, status=403)
        return view(request, *args, **kwargs)

    return wrapped


@csrf_protect
@require_POST
@rate_limit(scope="kiosk_activation", limit=5, window_seconds=300)
def activate_kiosk(request: HttpRequest) -> HttpResponse:
    """Exchange a token and fixed lane scope for a revocable browser capability."""
    configured_token = settings.KIOSK_ACTIVATION_TOKEN
    submitted_token = request.POST.get("token", "")
    if not configured_token:
        logger.error("Kiosk activation attempted without a configured token")
        return JsonResponse({"error": "Kiosk activation is unavailable."}, status=503)
    if not secrets.compare_digest(submitted_token, configured_token):
        logger.warning("Rejected invalid kiosk activation attempt")
        return JsonResponse({"error": "Kiosk activation failed."}, status=403)

    event_type = (request.POST.get("event_type") or "").strip().lower()
    if event_type not in {"entry", "exit"}:
        return JsonResponse({"error": "Choose an entry or exit lane."}, status=400)
    lot_name = (request.POST.get("lot") or "").strip()
    if lot_name:
        lot = ParkingLot.objects.filter(name=lot_name).first()
    else:
        lots = list(ParkingLot.objects.order_by("pk")[:2])
        lot = lots[0] if len(lots) == 1 else None
    if lot is None:
        return JsonResponse({"error": "Choose a valid parking lot."}, status=400)

    initial_nonce = secrets.token_urlsafe(32)
    expires_at = timezone.now() + timedelta(seconds=settings.KIOSK_SESSION_SECONDS)
    try:
        # Session + capability must commit together. On failure, flush the
        # in-memory session so SessionMiddleware cannot re-persist a half-baked
        # activation after the atomic block rolls back.
        with transaction.atomic():
            request.session.set_expiry(settings.KIOSK_SESSION_SECONDS)
            request.session.cycle_key()
            request.session["kiosk_token_fingerprint"] = _kiosk_token_fingerprint(
                configured_token
            )
            request.session["kiosk_scan_nonce"] = initial_nonce
            request.session.save()
            KioskDeviceCapability.objects.filter(
                expires_at__lte=timezone.now()
            ).delete()
            capability, _ = KioskDeviceCapability.objects.update_or_create(
                session_key=request.session.session_key,
                defaults={
                    "token_fingerprint": _kiosk_token_fingerprint(configured_token),
                    "lot": lot,
                    "event_type": event_type,
                    "nonce_hash": _nonce_hash(initial_nonce),
                    "expires_at": expires_at,
                },
            )
            capability.replay_digests.all().delete()
    except DatabaseError:
        logger.exception("Kiosk activation failed while writing device capability")
        try:
            request.session.flush()
        except DatabaseError:
            logger.exception(
                "Could not flush kiosk session after activation write failure"
            )
        return JsonResponse(
            {"error": "Kiosk activation is temporarily unavailable."},
            status=503,
        )

    response = HttpResponse(status=204)
    response["HX-Refresh"] = "true"
    return response


def _is_htmx(request: HttpRequest) -> bool:
    """Trust only HTMX's explicit request header when selecting HTML responses."""
    return request.headers.get("HX-Request", "").lower() == "true"


def _presentation_state(outcome: ScanOutcome) -> str:
    """Give the public UI one explicit state without leaking backend details."""
    if outcome.is_low_confidence:
        return "low_confidence"
    if outcome.outcome == "entry":
        return "entry_success"
    if outcome.outcome == "exit_matched":
        return "exit_success"
    if outcome.outcome == "unreadable_entry":
        return "unreadable"
    if outcome.outcome == "exit_unmatched":
        return "unmatched_exit"
    if outcome.outcome == "error":
        return "model_error" if outcome.status >= 500 else "invalid_image"
    logger.error("Unsupported kiosk scan outcome: %s", outcome.outcome)
    return "model_error"


def _public_payload(outcome: ScanOutcome) -> dict:
    """
    Shape a privacy-reduced kiosk payload from a ScanOutcome.

    Only ever includes what is safe to show to an anonymous person standing at the
    gate: plate text, confidence band, event type, and (on exit) the charge.
    """
    result = outcome.result
    plate_text = result["plate_text"] if result else ""
    band = confidence_band(float(result["confidence"])) if result else "error"

    payload: dict = {
        "presentation_state": _presentation_state(outcome),
        "outcome": outcome.outcome,
        "event_type": outcome.event_type,
        "plate_text": plate_text,
        "confidence_band": band,
        "is_low_confidence": outcome.is_low_confidence,
    }
    if outcome.error:
        payload["error"] = outcome.error

    if outcome.outcome == "entry":
        session = outcome.session
        payload["registered"] = session.user_id is not None
        payload["entry_time"] = session.entry_time.isoformat() if session else None
    elif outcome.outcome == "exit_matched":
        session = outcome.session
        # Decimal -> str preserves exact cents; never float in a money field.
        payload["charge_amount"] = str(session.charge_amount)
        payload["billed_to_account"] = session.user_id is not None
    # unreadable_entry / exit_unmatched / error carry only the generic fields; the
    # template renders a friendly "try again / see attendant" message for them.
    return payload


@csrf_protect
@require_POST
@rate_limit(scope="kiosk_scan", limit=20, window_seconds=60)
@kiosk_session_required
def kiosk_scan(request: HttpRequest):
    """
    Activated-kiosk endpoint: accept a plate image, run CV, open/close a session.

    Reuses the shared, hardened scan core and returns a privacy-reduced result.
    HTMX callers get an HTML fragment; other clients get JSON. Expected client
    errors (bad image, wrong type) render as HTML 200 for HTMX (which does not swap
    4xx by default) while JSON clients keep the real status code.
    """
    decision = _consume_kiosk_request(request)
    if decision is None:
        logger.warning("Rejected kiosk scan with invalid scope or nonce")
        return JsonResponse({"error": "Invalid kiosk request."}, status=403)
    if not decision.accepted:
        logger.warning("Rejected kiosk scan with a stale one-time nonce")
        response = JsonResponse({"error": "Invalid kiosk request."}, status=403)
        return _attach_nonce(response, decision.next_nonce)

    outcome = run_plate_scan(request)
    if outcome.outcome == "error":
        # Release the provisional replay mark so the driver can retry this photo.
        _forget_image_digest(request, decision.image_digest)

    payload = _public_payload(outcome)
    if _is_htmx(request):
        # Expected client errors and unreadable entry must render (200) for HTMX,
        # which does not swap 4xx responses by default.
        response = render(
            request, "public/kiosk_result.html", {"result": payload}, status=200
        )
    else:
        response = JsonResponse(payload, status=outcome.status)
    return _attach_nonce(response, decision.next_nonce)
