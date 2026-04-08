"""
config.py - Load secrets and configuration from environment variables.

HOW TO USE:
1. Copy `.env.example` to `.env` in the same directory
2. Fill in your actual secrets in `.env`
3. Never commit `.env` to git (it's in .gitignore)
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ---------- Neo4j ----------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------- Together AI ----------
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# ---------- App ----------
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdf")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"