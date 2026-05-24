#!/usr/bin/env python
"""
Django command-line utility.

This file is the entry point for all Django management commands, such as:
  python manage.py runserver       — start the development web server
  python manage.py migrate         — apply pending database migrations
  python manage.py makemigrations  — generate new migration files from model changes
  python manage.py createsuperuser — create an admin account interactively
  python manage.py setup_defaults  — our custom command to create initial data
  python manage.py shell_plus      — interactive Python shell with all models imported

HOW IT WORKS:
  1. This script sets the DJANGO_SETTINGS_MODULE environment variable so Django
     knows which settings file to load ('config/settings.py').
  2. It then calls execute_from_command_line(sys.argv), which reads the command
     name from the first argument (e.g., 'migrate') and dispatches to the
     appropriate management command class.
"""
import os
import sys


def main():
    """Run administrative tasks."""
    # Tell Django where to find the project settings module.
    # 'config.settings' → Python looks for config/settings.py relative to the
    # Python path (which includes the project root directory).
    # setdefault() means if DJANGO_SETTINGS_MODULE is already in the environment
    # (e.g., set by pytest via pytest.ini), we don't override it.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # sys.argv contains the command-line arguments.
    # sys.argv[0] is always the script name (manage.py).
    # sys.argv[1] is the management command (e.g., 'migrate').
    # Django reads these and dispatches to the right command.
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
