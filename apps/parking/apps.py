"""
App configuration for the 'parking' Django app.

The parking app is the core business logic domain of the entire system.
It owns:
  - Data models: LicensePlate, ParkingLot, LotSettings, ParkingSession,
    PlateDetectionEvent (Day 1)
  - Session services: handle_entry, handle_exit, calculate_charge,
    correct_plate (Day 7)
  - Plate normalization utilities (Day 7)
  - Image retention cleanup management command (Day 11)

WHY put billing, sessions, and plate data all in one app?
  They're tightly coupled. A ParkingSession references both a LicensePlate
  and a ParkingLot. PlateDetectionEvent references ParkingSession. The billing
  calculation (calculate_charge) needs LotSettings which is tied to ParkingLot.
  Splitting them into separate apps would create circular import dependencies
  and make the relationship graph harder to understand. One cohesive app with
  clear internal structure is better than two tightly-coupled apps.
"""

from django.apps import AppConfig


class ParkingConfig(AppConfig):
    """Configuration class for the parking application."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.parking'
    verbose_name = 'Parking'
