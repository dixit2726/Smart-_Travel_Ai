"""
App/frontend/pdf_generator.py

Backward-compatibility shim re-exporting PDF generation functionality from App.backend.pdf_report.
"""
try:
    from App.backend.pdf_report import *
except ImportError:
    from backend.pdf_report import *
