"""
Local configuration for the DICOM viewer / chest X-ray analysis scripts.

Copy this file to config.py (which is gitignored) and fill in your own
paths. config.py is loaded automatically if present; without it, the
scripts simply skip auto-loading a default folder/file.
"""

# Directory auto-loaded by dicom_viewer.py on startup, if it exists.
DEFAULT_DICOM_DIR = r""

# Default DICOM file used by chest_xray_analysis.py when no path is given.
DEFAULT_DICOM_PATH = r""
