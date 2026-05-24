"""
Admin configuration for the accounts app.

Registers the custom User model with the Django admin site,
extending Django's built-in UserAdmin to preserve all the
password management UI and permission fieldsets.

WHY extend UserAdmin instead of ModelAdmin?
  ModelAdmin is the generic admin class for any Django model.
  UserAdmin (from django.contrib.auth.admin) is a specialized subclass designed
  specifically for User models — it adds:
    - A dedicated "Change password" form (so hashing is done correctly)
    - Two-stage password confirmation on user creation
    - Proper fieldsets that group username/password/personal info/permissions
    - Read-only treatment of date_joined and last_login
  If we used plain ModelAdmin, the admin would let you set a plaintext password
  and save it directly to the database — a serious security bug.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from apps.accounts.models import User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    """
    Admin view for the custom User model.

    Extends Django's built-in UserAdmin to inherit all password management
    and permission handling. We add custom list_display and filters that
    make it easier for a parking lot operator to manage staff accounts.
    """

    # Columns shown in the user list view (/admin/accounts/user/).
    # Ordered by what's most useful when scanning a list of users:
    #   email     — primary contact identifier
    #   username  — login identifier
    #   is_staff  — quickly see who has admin access
    #   is_active — see which accounts are enabled
    #   date_joined — audit trail
    list_display = ['email', 'username', 'is_staff', 'is_active', 'date_joined']

    # Sidebar filter panel — lets operators quickly narrow the list.
    list_filter = ['is_staff', 'is_active', 'date_joined']

    # Fields searched when the operator types in the search box.
    search_fields = ['email', 'username']

    # Show newest accounts first — most useful when troubleshooting recent logins.
    ordering = ['-date_joined']
