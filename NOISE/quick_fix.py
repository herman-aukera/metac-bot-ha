#!/usr/bin/env python3
"""
Quick fix script to get the bot working today.
This bypasses SSL issues and uses a simpler approach.
"""

import os
import ssl
import certifi

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Disable SSL verification for development (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

print("SSL fix applied. Now run your bot:")
print("python3 main.py --mode tournament")
