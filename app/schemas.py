"""
Pydantic schemas for the SHL Assessment Recommender API.
Matches the exact schema specified in the assignment — non-negotiable.
"""

from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat. Carries full conversation history (stateless)."""
    messages: list[Message]


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(..., description="Test type code: K=Knowledge, P=Personality, A=Ability, S=Skills, B=Behavioral")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str = Field(..., description="The agent's conversational reply")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when gathering context or refusing. 1-10 items when recommending."
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = "ok"
