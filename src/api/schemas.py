"""Pydantic schemas for WebSocket message serialization."""
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """WebSocket message types from client to server."""

    # Client → Server
    START_TRIAL = "start_trial"
    USER_INPUT = "user_input"
    PING = "ping"

    # Server → Client
    SESSION_CREATED = "session_created"
    NODE_EXECUTED = "node_executed"
    INTERRUPT_REQUEST = "interrupt_request"
    TRIAL_COMPLETED = "trial_completed"
    ERROR = "error"
    PONG = "pong"


class WSMessage(BaseModel):
    """Base WebSocket message wrapper."""

    type: MessageType
    thread_id: Optional[str] = None
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Client → Server Request Models


class StartTrialRequest(BaseModel):
    """Request to start a new trial session."""

    case_info: dict[str, Any]
    evidence_list: list[dict[str, Any]] = Field(default_factory=list)


class UserInputRequest(BaseModel):
    """Request to respond to an interrupt with user input."""

    interrupt_node: str
    input: Any  # Could be bool, str, or dict for evidence


# Server → Client Response Models


class SessionCreatedData(BaseModel):
    """Data sent when session is created."""

    message: str
    thread_id: str


class NodeExecutedData(BaseModel):
    """Data sent when a node completes execution."""

    node_name: str
    state_delta: dict[str, Any]
    current_phase: str
    progress: float
    message_count: int
    focus: list[str] = Field(default_factory=list)
    rounds: dict[str, int] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)


class InterruptRequestData(BaseModel):
    """Data sent when human input is required at an interrupt point."""

    node_name: str
    prompt: str
    input_type: str  # "boolean", "string", or "evidence"
    options: Optional[list[str]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialCompletedData(BaseModel):
    """Data sent when trial completes."""

    final_state: dict[str, Any]


class ErrorData(BaseModel):
    """Data sent when an error occurs."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# Helper models for evidence submission


class EvidenceSubmit(BaseModel):
    """Format for evidence submission at interrupt points."""

    current_evidence: list[dict[str, Any]] | dict[str, Any]
    messages: str
