"""
schemas.py — all Pydantic models for the CV parsing service.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    cvId: str = Field(..., examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"])
    cvText: str = Field(..., min_length=10, description="Plain text extracted from the CV PDF.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Echoed back in the local JSON file.")


class ParseAccepted(BaseModel):
    status: str = "accepted"
    cvId: str
    message: str = "CV queued for parsing. Result will be saved locally."


class Skills(BaseModel):
    technical: List[str] = Field(default_factory=list)
    nonTechnical: List[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    startDate: Optional[str] = Field(None, description="YYYY-MM or YYYY")
    endDate: Optional[str] = Field(None, description="YYYY-MM, YYYY, 'present', or null")
    description: Optional[str] = None


class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None


class ParsedData(BaseModel):
    fullName: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: Skills = Field(default_factory=Skills)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    cvId: str
    status: str = Field(..., description="'completed' or 'failed'")
    parsedData: Optional[ParsedData] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None