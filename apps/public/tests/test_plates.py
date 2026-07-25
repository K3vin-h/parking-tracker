"""
Tests for resident plate management — ownership scoping is the key invariant.
"""

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from apps.parking.models import LicensePlate

User = get_user_model()

PLATES_URL = reverse("public:plates")


@pytest.fixture
def user(db):
    return User.objects.create_user(username="owner", email="o@e.com", password="x")


@pytest.mark.django_db
class TestPlatesAuth:
    def test_anonymous_redirected_to_login(self, client):
        resp = client.get(PLATES_URL)
        assert resp.status_code == 302
        assert "/login/" in resp.url


@pytest.mark.django_db
class TestPlatesManagement:
    def test_add_plate_scoped_to_user(self, client, user):
        client.force_login(user)
        resp = client.post(PLATES_URL, {"plate_text": "abc 123", "label": "Car"})
        assert resp.status_code == 302
        plate = LicensePlate.objects.get(user=user)
        assert plate.plate_text == "ABC123"  # normalized

    def test_duplicate_plate_rejected(self, client, user):
        LicensePlate.objects.create(user=user, plate_text="ABC123")
        client.force_login(user)
        resp = client.post(PLATES_URL, {"plate_text": "ABC123", "label": ""})
        assert resp.status_code == 200  # form re-rendered with error
        assert LicensePlate.objects.filter(user=user, plate_text="ABC123").count() == 1

    def test_plate_owned_by_another_user_is_rejected_without_owner_details(
        self, client, user
    ):
        """Ownership conflicts must be generic and must not create ambiguity."""
        other = User.objects.create_user(
            username="other-owner",
            email="other@example.com",
            password="x",
        )
        LicensePlate.objects.create(user=other, plate_text="ABC123")
        client.force_login(user)

        resp = client.post(PLATES_URL, {"plate_text": " abc 123 "})

        assert resp.status_code == 200
        assert LicensePlate.objects.filter(plate_text="ABC123").count() == 1
        assert b"other-owner" not in resp.content

    def test_cannot_delete_another_users_plate(self, client, user):
        other = User.objects.create_user(
            username="other", email="x@e.com", password="x"
        )
        other_plate = LicensePlate.objects.create(user=other, plate_text="OTHER1")
        client.force_login(user)
        resp = client.post(reverse("public:plate_delete", args=[other_plate.pk]))
        assert resp.status_code == 404
        assert LicensePlate.objects.filter(pk=other_plate.pk).exists()

    def test_delete_own_plate(self, client, user):
        plate = LicensePlate.objects.create(user=user, plate_text="MINE01")
        client.force_login(user)
        resp = client.post(reverse("public:plate_delete", args=[plate.pk]))
        assert resp.status_code == 302
        assert not LicensePlate.objects.filter(pk=plate.pk).exists()

    def test_delete_requires_post(self, client, user):
        plate = LicensePlate.objects.create(user=user, plate_text="MINE01")
        client.force_login(user)
        resp = client.get(reverse("public:plate_delete", args=[plate.pk]))
        assert resp.status_code == 405

    def test_delete_confirmation_uses_csp_allowed_external_script(
        self, client, user
    ):
        """The confirmation must not depend on an inline handler blocked by CSP."""
        LicensePlate.objects.create(user=user, plate_text="MINE01")
        client.force_login(user)

        resp = client.get(PLATES_URL)
        html = resp.content.decode()

        assert "onsubmit=" not in html
        assert "data-plate-delete" in html
        assert "/static/js/plates.js" in html
