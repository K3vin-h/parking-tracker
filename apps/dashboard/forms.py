"""Validated operator forms for parking-lot configuration."""

from decimal import Decimal

from django import forms

from apps.parking.models import LotSettings


def _coerce_retention(value: str) -> int | None:
    """Convert the finite choice to days while preserving blank as forever."""
    return int(value) if value else None


class LotSettingsForm(forms.ModelForm):
    """
    Edit billing, retention, and CV thresholds without exposing storage formats.

    WHY custom percentage/retention fields: operators work naturally with a
    percentage and a finite set of retention policies, while the model stores
    confidence as 0–1 and uses NULL to mean "forever".
    """

    rate = forms.DecimalField(
        min_value=Decimal("0.01"),
        max_digits=8,
        decimal_places=2,
        widget=forms.NumberInput(attrs={"min": "0.01", "step": "0.01"}),
    )
    daily_cap_amount = forms.DecimalField(
        min_value=Decimal("0.01"),
        max_digits=8,
        decimal_places=2,
        required=False,
        widget=forms.NumberInput(attrs={"min": "0.01", "step": "0.01"}),
    )
    confidence_threshold = forms.DecimalField(
        min_value=Decimal("0"),
        max_value=Decimal("100"),
        decimal_places=1,
        max_digits=4,
        help_text="Confidence percentage below which detections require review.",
    )
    image_retention_days = forms.TypedChoiceField(
        choices=[
            ("7", "7 days"),
            ("30", "30 days"),
            ("90", "90 days"),
            ("", "Forever"),
        ],
        coerce=_coerce_retention,
        empty_value=None,
        required=False,
        help_text="How long private detection images are retained.",
    )

    class Meta:
        model = LotSettings
        fields = [
            "rate",
            "billing_unit",
            "grace_period_minutes",
            "daily_cap_enabled",
            "daily_cap_amount",
        ]
        widgets = {
            "grace_period_minutes": forms.NumberInput(
                attrs={"min": "0", "max": "1440", "step": "1"}
            ),
        }

    def __init__(self, *args, **kwargs):
        """Populate UI-only values from their normalized model representation."""
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.initial["confidence_threshold"] = (
                Decimal(str(self.instance.confidence_threshold)) * Decimal("100")
            )
            self.initial["image_retention_days"] = (
                self.instance.image_retention_days or ""
            )

    def clean(self):
        """
        Enforce cross-field billing rules before configuration reaches services.

        WHY clear a disabled cap: retaining a stale amount is ambiguous during a
        later re-enable and makes audits imply a cap that was not active.
        """
        cleaned = super().clean()
        cap_enabled = cleaned.get("daily_cap_enabled", False)
        cap_amount = cleaned.get("daily_cap_amount")
        if cap_enabled and cap_amount is None:
            self.add_error(
                "daily_cap_amount",
                "Enter a positive daily cap amount when the cap is enabled.",
            )
        elif not cap_enabled:
            cleaned["daily_cap_amount"] = None
        return cleaned

    def save(self, commit=True):
        """Convert operator-facing values to the model's canonical storage form."""
        instance = super().save(commit=False)
        instance.confidence_threshold = float(
            self.cleaned_data["confidence_threshold"] / Decimal("100")
        )
        instance.image_retention_days = self.cleaned_data["image_retention_days"]
        if not instance.daily_cap_enabled:
            instance.daily_cap_amount = None
        if commit:
            instance.save()
        return instance
