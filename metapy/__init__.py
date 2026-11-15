"""MetaPy — an intelligent, transparent meta-analysis assistant.

Workflow: load (CSV/Excel) -> scan (what analyses are possible) -> prepare
(derive missing statistics with the derivation brain, ask the researcher) ->
analyse -> review/extend -> choose outputs -> export.
"""
__version__ = "5.0.0"

from .session import MetaPySession  # noqa: E402,F401
