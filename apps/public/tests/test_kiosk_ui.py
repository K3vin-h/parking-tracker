"""Template contracts for the public kiosk's hybrid scan states."""

from types import SimpleNamespace

from django.template.loader import render_to_string


def _render_result(**overrides):
    """Render one complete result while keeping each test focused on its variation."""
    result = {
        "presentation_state": "entry_success",
        "event_type": "entry",
        "plate_text": "ABC123",
        "registered": False,
        "is_low_confidence": False,
    }
    result.update(overrides)
    return render_to_string("public/kiosk_result.html", {"result": result})


def test_entry_success_is_a_focused_replacement_result():
    """Removing the success-mode marker must break focused replacement."""
    html = _render_result()

    assert 'data-kiosk-result-mode="success"' in html
    assert "Welcome" in html
    assert "Entry recorded. Please proceed through the gate." in html
    assert "Scan another vehicle" in html


def test_low_confidence_is_a_stacked_recovery_result():
    """Low-confidence output must never reuse the visually decisive success state."""
    html = _render_result(
        presentation_state="low_confidence",
        is_low_confidence=True,
        plate_text="ABC128",
    )

    assert 'data-kiosk-result-mode="recovery"' in html
    assert "Check this plate" in html
    assert "The reading was uncertain." in html
    assert "Retake photo" in html
    assert "Call attendant" in html


def test_invalid_image_does_not_echo_unsafe_error_details():
    """Raw backend errors must not cross the anonymous kiosk trust boundary."""
    html = _render_result(
        presentation_state="invalid_image",
        error="/srv/media/private.jpg failed in Pillow",
        plate_text="",
    )

    assert "This image cannot be used" in html
    assert "valid JPEG or PNG smaller than 10 MB" in html
    assert "/srv/media" not in html
    assert "Pillow" not in html


def test_exit_success_exposes_charge_but_not_balance():
    """The gate may explain the charge disposition without revealing account funds."""
    html = _render_result(
        presentation_state="exit_success",
        event_type="exit",
        charge_amount="12.50",
        billed_to_account=True,
    )

    assert "$12.50" in html
    assert "Billed to your account" in html
    assert "balance" not in html.lower()


def test_kiosk_shell_exposes_browser_state_hooks():
    """Removing a panel hook must break the browser's state-transition contract."""
    html = render_to_string(
        "public/kiosk.html",
        {
            "kiosk_activated": True,
            "kiosk_lot": SimpleNamespace(name="Test Lot"),
            "kiosk_event_type": "entry",
            "kiosk_scan_nonce": "nonce-value",
        },
    )

    assert 'id="kiosk-panel"' in html
    assert 'id="kiosk-form"' in html
    assert 'id="kiosk-processing"' in html
    assert 'id="kiosk-result"' in html
    assert 'aria-busy="false"' in html


def test_activation_shell_exposes_visible_error_hooks():
    """Activation failures need an operator-visible target outside scan controls."""
    html = render_to_string(
        "public/kiosk.html",
        {
            "kiosk_activated": False,
            "lots": [SimpleNamespace(name="Test Lot")],
        },
    )

    assert "data-kiosk-activation-form" in html
    assert "data-kiosk-activation-result" in html
    assert 'aria-live="assertive"' in html
