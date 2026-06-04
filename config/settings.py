"""
Django settings for the parking tracker project.

ALL SENSITIVE VALUES ARE READ FROM ENVIRONMENT VARIABLES — NEVER HARDCODED HERE.

For local development:
  1. Copy .env.example → .env
  2. Fill in all values (especially SECRET_KEY and DB_PASSWORD)
  3. Run: docker-compose up --build

WHY environment variables for secrets?
  If a secret is hardcoded in source code and the repo is ever made public
  (or shared with the wrong person), that secret is permanently compromised.
  Environment variables keep secrets OUT of the codebase entirely.
  The .env file that holds the actual values is excluded from git via .gitignore.

Django settings reference: https://docs.djangoproject.com/en/5.1/topics/settings/
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


def _require_env(var_name):
    """
    Read a required environment variable.

    Raises ImproperlyConfigured at startup if the variable is missing,
    rather than silently falling back to an insecure default.
    This is the "fail loudly" principle — a misconfigured app should
    refuse to start rather than run with broken or insecure settings.

    Args:
        var_name: Name of the required environment variable.

    Returns:
        The environment variable value as a string.

    Raises:
        ImproperlyConfigured: If the variable is not set.
    """
    value = os.environ.get(var_name, '').strip()
    if not value:
        raise ImproperlyConfigured(
            f"Required environment variable '{var_name}' is not set. "
            f"Copy .env.example to .env and configure all required values."
        )
    return value


# ── Path Setup ────────────────────────────────────────────────────────────────

# BASE_DIR is the project root — the directory containing manage.py.
# Path(__file__) = this file (config/settings.py)
# .resolve()     = convert to absolute path (no symlinks, no relative segments)
# .parent        = config/ directory
# .parent        = project root (one more level up)
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Security Settings ─────────────────────────────────────────────────────────

# SECRET_KEY is used by Django to sign cookies, sessions, CSRF tokens, and
# password reset URLs. It must be:
#   - Unique per environment (dev, staging, production all need different keys)
#   - Long and random (Django generates 50-char keys by default)
#   - Never committed to version control
# We fail loudly if it's not set, rather than defaulting to something insecure.
SECRET_KEY = _require_env('SECRET_KEY')

# DEBUG=True enables the detailed error page that shows:
#   - The full Python stack trace
#   - All local variables at each stack frame
#   - All HTTP headers from the request
#   - All Django settings (minus SECRET_KEY)
# This is invaluable for development but a serious security risk in production.
# We parse the string 'True'/'False' from the env var as a Python boolean.
DEBUG = os.environ.get('DEBUG', 'False').strip().lower() == 'true'

# ALLOWED_HOSTS is Django's defense against HTTP Host header injection attacks.
# An attacker could craft a request with Host: evil.com to your server.
# If your server naively uses the Host header (e.g., in emails), it would generate
# links pointing to evil.com. ALLOWED_HOSTS rejects requests with non-matching hosts.
_raw_hosts = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1')
ALLOWED_HOSTS = [h.strip() for h in _raw_hosts.split(',') if h.strip()]


# ── Installed Applications ────────────────────────────────────────────────────

# ORDER MATTERS: Django loads apps in this order.
# Built-in Django apps must come before our custom apps because:
#   - django.contrib.auth defines the base User model that our User subclasses
#   - django.contrib.admin requires auth and contenttypes to be loaded first
#   - Our apps depend on built-ins being available when their models load
INSTALLED_APPS = [
    # ── Django built-ins ──────────────────────────────────────────────────────
    'django.contrib.admin',        # Admin site at /admin/ — browse/edit all models
    'django.contrib.auth',         # Authentication: User model, login/logout, permissions
    'django.contrib.contenttypes', # Generic relations (allows FKs to any model by type)
    'django.contrib.sessions',     # Server-side session storage (keeps users logged in)
    'django.contrib.messages',     # One-time flash messages (e.g., "Settings saved!")
    'django.contrib.staticfiles',  # Serves CSS/JS files from app directories in development

    # ── Our apps ─────────────────────────────────────────────────────────────
    # Listed after built-ins so Django resolves built-in models first.
    'apps.accounts',    # Custom User model (Day 1)
    'apps.parking',     # Parking sessions, billing, plate data (Day 1 models)
    'apps.cv',          # Computer vision pipeline (Day 2+)
    'apps.dashboard',   # Web views and API endpoints (Day 8+)
]

# ── Middleware ────────────────────────────────────────────────────────────────

# Middleware is a chain of hooks that process every request before it reaches
# a view, and every response before it's sent to the user.
# ORDER MATTERS: middleware runs top-to-bottom on the way in, bottom-to-top on the way out.
MIDDLEWARE = [
    # Adds security HTTP headers: HSTS, X-Content-Type-Options, Referrer-Policy.
    'django.middleware.security.SecurityMiddleware',
    # Loads the session from the database based on the session cookie.
    'django.contrib.sessions.middleware.SessionMiddleware',
    # Normalizes URL paths (adds trailing slashes when APPEND_SLASH=True).
    'django.middleware.common.CommonMiddleware',
    # Validates CSRF tokens on POST/PUT/PATCH/DELETE requests.
    # CSRF (Cross-Site Request Forgery): a malicious page tricks a logged-in user's
    # browser into making a request to our server. The CSRF token is a secret that
    # the malicious page can't read, so forged requests are rejected.
    'django.middleware.csrf.CsrfViewMiddleware',
    # Attaches request.user based on the session data (who is logged in).
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # Stores flash messages in the session and retrieves them in the next request.
    'django.contrib.messages.middleware.MessageMiddleware',
    # Sets X-Frame-Options: DENY header to prevent clickjacking.
    # Clickjacking: embedding our site in an iframe on a malicious page to trick users
    # into clicking buttons they don't intend to.
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# ── Templates ─────────────────────────────────────────────────────────────────

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # Project-level templates directory — for base.html, login.html, etc.
        # App templates (if any) are found via APP_DIRS=True below.
        'DIRS': [BASE_DIR / 'templates'],
        # Also search each installed app's 'templates/' subdirectory.
        # E.g., Django's admin templates live in django/contrib/admin/templates/.
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                # Makes settings.DEBUG available as 'debug' in templates.
                'django.template.context_processors.debug',
                # Makes the current request object available as 'request' in templates.
                # Required by HTMX patterns that check request.htmx.
                'django.template.context_processors.request',
                # Makes request.user and request.user.perms available in templates.
                'django.contrib.auth.context_processors.auth',
                # Makes messages (flash messages) available in templates.
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# ── Database ──────────────────────────────────────────────────────────────────

# WHY PostgreSQL instead of SQLite?
#   - DecimalField precision: PostgreSQL stores Decimal values exactly using the
#     NUMERIC type. SQLite doesn't have a NUMERIC type and uses floating-point,
#     which introduces rounding errors in monetary calculations (critical for billing).
#   - JSONField: PostgreSQL has a native JSON/JSONB column type. SQLite stores JSON
#     as plain text (no indexing, no query operators).
#   - Concurrency: PostgreSQL handles multiple simultaneous writers safely.
#     SQLite uses file-level locking which causes errors under concurrent load.
#
# The connection params come from environment variables set in .env.
# 'db' is the Docker Compose service name — Docker's internal DNS resolves it
# to the PostgreSQL container's IP address.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        # DB_NAME and DB_USER have safe defaults (non-secret identifiers).
        # DB_PASSWORD has no default — fail loudly if it is not explicitly set
        # rather than connecting with an empty password (which some Postgres
        # trust-auth configs would silently accept).
        'NAME': os.environ.get('DB_NAME', 'parking_tracker'),
        'USER': os.environ.get('DB_USER', 'parking_user'),
        'PASSWORD': _require_env('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
        # Reuse DB connections for up to 60 seconds instead of opening a new
        # connection on every request.  Reduces connection overhead under load.
        'CONN_MAX_AGE': 60,
    }
}


# ── Custom User Model ─────────────────────────────────────────────────────────

# WHY point to a custom User model?
#   Django's built-in User model (django.contrib.auth.models.User) is frozen
#   after the first migration. You CANNOT change it later without wiping the database.
#   By defining our own User subclass from the start — even if it inherits everything
#   unchanged — we preserve the ability to add fields in the future with a simple migration.
#
#   This is Django's own recommendation for all new projects:
#   https://docs.djangoproject.com/en/5.1/topics/auth/customizing/#using-a-custom-user-model-when-starting-a-project
AUTH_USER_MODEL = 'accounts.User'


# ── Authentication Redirects ──────────────────────────────────────────────────

# Where unauthenticated users are redirected when they try to access a protected page.
# Django's @login_required decorator and LoginRequiredMixin use this setting.
LOGIN_URL = '/login/'

# Where users land after a successful login (if no 'next' URL param is present).
LOGIN_REDIRECT_URL = '/'

# Where users land after clicking logout.
# Redirecting to login ensures they can't navigate back to a cached protected page.
LOGOUT_REDIRECT_URL = '/login/'


# ── Password Validation ───────────────────────────────────────────────────────

# These validators run when a user creates or changes their password.
# Django enforces all of them — a password must pass ALL validators.
AUTH_PASSWORD_VALIDATORS = [
    # Rejects passwords similar to the user's username or email.
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    # Rejects passwords shorter than 8 characters (Django's default minimum).
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    # Rejects commonly used passwords (Django ships a list of ~20,000 common ones).
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    # Rejects entirely numeric passwords (e.g., '12345678').
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# ── Internationalization ──────────────────────────────────────────────────────

LANGUAGE_CODE = 'en-us'
# Store all datetimes in the database as timezone-aware UTC.
# WHY UTC? It avoids Daylight Saving Time bugs and makes multi-timezone deployments
# straightforward — you always know exactly what timezone the stored value is in.
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# ── Static Files ──────────────────────────────────────────────────────────────

# Static files: CSS, JavaScript, fonts checked into the repo.
# NOT the same as media files (user uploads).

# URL prefix for static files when served in development.
STATIC_URL = '/static/'

# Source directories where Django looks for static files in development.
# 'manage.py runserver' serves these directly without collectstatic.
# These directories ARE committed to git (they contain our CSS/JS source files).
STATICFILES_DIRS = [BASE_DIR / 'static']

# Where 'manage.py collectstatic' gathers ALL static files for production serving.
# This directory is SEPARATE from the source 'static/' in STATICFILES_DIRS above —
# Django raises an error if STATIC_ROOT is inside STATICFILES_DIRS.
# WHY separate? collectstatic copies files from all apps into one directory so a
# web server (nginx) can serve them from a single location without knowing about apps.
# In development, STATIC_ROOT is never used — 'runserver' reads from STATICFILES_DIRS.
STATIC_ROOT = BASE_DIR / 'staticfiles'


# ── Media Files ───────────────────────────────────────────────────────────────

# Media files: user-uploaded content (plate images in our case).
# These are dynamic — they change as users upload images.

# URL prefix for accessing uploaded media files.
MEDIA_URL = '/media/'

# Filesystem path where uploaded files are stored.
# In Docker, this directory is mounted as a named volume (docker-compose.yml)
# so uploaded images persist across container restarts.
MEDIA_ROOT = BASE_DIR / 'media'


# ── Primary Key Type ──────────────────────────────────────────────────────────

# Default auto-generated primary key type for all models.
# BigAutoField = 64-bit integer (supports 9.2 quintillion rows).
# IntegerField (the old Django default) caps at ~2.1 billion rows.
# For a parking system that might process thousands of events per day,
# BigAutoField is the safe choice.
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ── Logging ───────────────────────────────────────────────────────────────────

# Log application messages to the console.
# In development, this shows up in 'docker-compose logs web'.
# In production, these would be forwarded to a log aggregator (Datadog, CloudWatch, etc.).
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            # Format: [timestamp] LEVEL logger_name: message
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        # Logger for our application code.
        # Usage: import logging; logger = logging.getLogger(__name__)
        'apps': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}


# ── Development-only apps ─────────────────────────────────────────────────────

# django-extensions provides shell_plus, show_urls, graph_models, etc.
# These are development tools only — they add no value in production and
# increase the attack surface slightly, so they're excluded from prod images.
if DEBUG:
    INSTALLED_APPS += ['django_extensions']


# ── Production security settings ─────────────────────────────────────────────

# These settings are only meaningful when running behind HTTPS (production).
# Enabling them in development (where there is no TLS) would break the dev server.
if not DEBUG:
    # Tell browsers: "only ever connect to this site over HTTPS for 1 year."
    # Once this header is seen, the browser will refuse HTTP connections for the
    # site — protecting users even if they type http:// manually.
    SECURE_HSTS_SECONDS = 31_536_000       # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True  # apply to all subdomains too
    SECURE_HSTS_PRELOAD = True             # eligible for browser HSTS preload list

    # Redirect all HTTP requests to HTTPS at the Django level.
    # In production, nginx/load-balancer should do this too — defence in depth.
    # The proxy must pass X-Forwarded-Proto=https so Django does not redirect
    # already-secure client traffic that arrived over HTTP from the TLS proxy.
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = True

    # Session and CSRF cookies must only be sent over HTTPS.
    # Without these, a network attacker could steal session/CSRF tokens over HTTP.
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
