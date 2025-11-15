"""
API module for the Intelligent Traffic System.

This module provides REST API endpoints for all traffic monitoring functionality.
"""

from .main import app, router

__all__ = ["app", "router"]

