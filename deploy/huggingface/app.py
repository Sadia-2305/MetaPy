"""Hugging Face Space entry point for MetaPy (website mode: nothing is saved on the server)."""
import os

os.environ.setdefault("METAPY_WEB", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from metapy.app import launch  # noqa: E402

launch()
