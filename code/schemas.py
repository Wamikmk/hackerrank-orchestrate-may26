"""
schemas.py — Pydantic models for pipeline input and output rows.

Python field names are lowercase snake_case internally.
CSV column names are Title Case — handled by the CSV reader/writer in main.py.
"""

from typing import Literal
from pydantic import BaseModel


class TicketInput(BaseModel):
    issue: str
    subject: str
    company: str


class TriageOutput(BaseModel):
    status: Literal["Replied", "Escalated"]
    product_area: str       # blank when status == "Escalated"
    response: str           # never blank
    justification: str      # internal — not written to output CSV
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
