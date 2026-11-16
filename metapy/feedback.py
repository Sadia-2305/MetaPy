"""Optional feedback from website users (MetaPy 5.2).

Only what the user types in the feedback box is sent: a 1-5 star rating, an optional comment, an optional role and the
MetaPy version. No data, results or file names are ever sent. Responses go to a Google Form owned by the developer,
configured in FORM below (or in Streamlit secrets under [feedback], or the METAPY_FEEDBACK environment variable as JSON).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from . import __version__

# Google Form "formResponse" address and the entry ids of its questions (not secret; filled in by the developer)
FORM = {"action": "https://docs.google.com/forms/d/e/1FAIpQLSc2KrjoYrY5Fha60YiRvJXzWVlutPeo83317Amkd7orCNc4VA/formResponse",
        "rating": "entry.390804516", "comment": "entry.829629288", "role": "entry.1196410390", "version": ""}

# must match the choices of the "Role" question in the Google Form exactly (the first item is never sent)
ROLES = ["Prefer not to say", "Student", "Researcher / academic", "Clinician / practitioner", "Statistician / methodologist", "Others"]


def config() -> dict:
    cfg = dict(FORM)
    try:
        import streamlit as st
        cfg.update({k: v for k, v in dict(st.secrets.get("feedback", {})).items() if v})
    except Exception:
        pass
    if os.environ.get("METAPY_FEEDBACK"):
        try:
            cfg.update(json.loads(os.environ["METAPY_FEEDBACK"]))
        except ValueError:
            pass
    return cfg


def configured() -> bool:
    c = config()
    return bool(c.get("action") and c.get("rating"))


def send(rating: int, comment: str = "", role: str = "", timeout: float = 8.0) -> tuple[bool, str]:
    """Send one response. Returns (ok, message for the user)."""
    c = config()
    if not (c.get("action") and c.get("rating")):
        return False, "Feedback is not set up on this server yet, so nothing was sent."
    fields = {c["rating"]: str(int(rating))}
    if c.get("comment") and comment.strip():
        fields[c["comment"]] = comment.strip()[:2000]
    if c.get("role") and role and role != ROLES[0]:
        fields[c["role"]] = role
    if c.get("version"):
        fields[c["version"]] = __version__
    data = urllib.parse.urlencode(fields).encode()
    try:
        req = urllib.request.Request(c["action"], data=data, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 400
    except Exception:
        ok = False
    return (True, "Thank you! Your feedback was sent.") if ok else (False, "Sorry, the feedback could not be sent. Please try again later.")
