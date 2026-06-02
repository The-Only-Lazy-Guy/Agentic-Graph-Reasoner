"""Offline graph growth helpers for V5.

The graph grower is intentionally separate from V4/V5 inference.  It audits
session-proposed graph edits, stages candidates, and leaves actual graph
mutation to later explicit promotion commands.
"""
