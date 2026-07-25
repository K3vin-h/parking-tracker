"""
Forms for the public resident portal.

All of these are user-facing boundaries: they validate untrusted input and never
grant privilege. Account creation in particular NEVER exposes is_staff/is_superuser
— those are forced off in the view, and this form only accepts username/email/
password so the field set can't be abused for privilege escalation.
"""

from decimal import Decimal

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

from apps.parking.services import normalize_plate

User = get_user_model()

# A sane ceiling so a single provider-confirmed top-up cannot inject an absurd
# balance. A real connector may impose a lower provider-specific limit.
MAX_TOPUP = Decimal("1000.00")


class SignupForm(UserCreationForm):
    """Public account creation: username + email + password (with confirmation)."""

    email = forms.EmailField(
        required=True,
        help_text="Used for account recovery.",
    )

    class Meta(UserCreationForm.Meta):
        model = User
        # Deliberately only these fields — no is_staff/is_active/is_superuser can
        # ever be submitted through this form.
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        """Apply the shared form-control styling to every rendered widget."""
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")

    def clean_email(self):
        """Enforce unique emails case-insensitively so accounts stay distinct."""
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class PlateForm(forms.Form):
    """Add a plate to the current user's account."""

    plate_text = forms.CharField(
        max_length=20,
        label="License plate",
        help_text="Letters and numbers; spaces are ignored.",
        widget=forms.TextInput(
            attrs={"class": "form-control mono", "autocapitalize": "characters"}
        ),
    )
    label = forms.CharField(
        max_length=100,
        required=False,
        label="Label (optional)",
        help_text="e.g. 'Daily driver'.",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    def clean_plate_text(self):
        """Normalize exactly as the CV/billing path does so matching is consistent."""
        normalized = normalize_plate(self.cleaned_data["plate_text"])
        if not normalized:
            raise forms.ValidationError("Enter a valid plate.")
        if len(normalized) > 20:
            raise forms.ValidationError("Plate is too long.")
        return normalized


class TopupForm(forms.Form):
    """Collect the amount a payment connector must confirm before wallet credit."""

    amount = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        min_value=Decimal("0.01"),
        max_value=MAX_TOPUP,
        label="Amount",
        help_text="How much to add to your balance.",
        widget=forms.NumberInput(
            attrs={"class": "form-control mono", "step": "0.01", "min": "0.01"}
        ),
    )
