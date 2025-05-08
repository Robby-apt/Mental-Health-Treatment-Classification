#!/bin/bash
set -e

echo "Running Flask app with Python for debugging..."

# You can uncomment the line below to use Gunicorn once everything works
# exec gunicorn app:app -b 0.0.0.0:5000

exec python app.py
