"""
WSGI (Web Server Gateway Interface) entry point for the parking tracker.

WHAT IS WSGI?
  WSGI is the standard protocol between Python web applications and web servers.
  Think of it like an electrical socket standard — any WSGI-compatible server
  (gunicorn, uWSGI, mod_wsgi) can plug into any WSGI-compatible framework (Django,
  Flask, FastAPI) using the same interface.

HOW IT WORKS:
  A WSGI server (like gunicorn) receives HTTP requests from users.
  For each request, it calls the 'application' object defined below.
  Django's application callable reads the request, runs your view function,
  and returns an HTTP response back to the server, which sends it to the user.

WHEN IS THIS USED?
  - Production: gunicorn calls 'application' for every request.
  - Testing: Django's test client simulates requests through this interface.
  - NOT used by manage.py runserver (the dev server has its own simpler request handler).

For local development, you typically don't interact with this file directly.
'docker-compose exec web python manage.py runserver' bypasses WSGI entirely.
"""
import os

from django.core.wsgi import get_wsgi_application

# Set the settings module for this WSGI process.
# Mirrors the same setting in manage.py.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# get_wsgi_application() loads Django, applies all settings and middleware,
# and returns a callable that accepts (environ, start_response) per the WSGI spec.
application = get_wsgi_application()
