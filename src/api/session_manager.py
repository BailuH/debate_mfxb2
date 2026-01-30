"""Session manager for handling trial sessions and thread IDs."""
from typing import Dict, Optional
from datetime import datetime
import uuid
import logging
from fastapi import WebSocket
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages trial sessions with thread IDs and WebSocket references."""

    def __init__(self, memory: MemorySaver) -> None:
        """
        Initialize the session manager.

        Args:
            memory: LangGraph MemorySaver for state persistence
        """
        self.memory = memory
        self.sessions: Dict[str, dict] = {}

    def create_session(self, websocket: WebSocket) -> str:
        """
        Create a new trial session and generate a thread ID.

        Args:
            websocket: The WebSocket connection for this session

        Returns:
            The generated thread ID (UUID4 string)
        """
        thread_id = str(uuid.uuid4())
        self.sessions[thread_id] = {
            "config": {"configurable": {"thread_id": thread_id}},
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "status": "active",  # active, interrupted, completed
            "current_node": None,
            "websocket": websocket,
        }
        logger.info(f"Created session with thread_id: {thread_id}")
        return thread_id

    def get_session(self, thread_id: str) -> Optional[dict]:
        """
        Get a session by thread ID.

        Args:
            thread_id: The thread ID to look up

        Returns:
            The session dict if found, None otherwise
        """
        return self.sessions.get(thread_id)

    def update_session(self, thread_id: str, **kwargs) -> None:
        """
        Update session properties and refresh last_active timestamp.

        Args:
            thread_id: The thread ID to update
            **kwargs: Properties to update (status, current_node, etc.)
        """
        if thread_id in self.sessions:
            self.sessions[thread_id].update(kwargs)
            self.sessions[thread_id]["last_active"] = datetime.now()
            logger.debug(f"Updated session {thread_id}: {kwargs}")

    async def cleanup_session(self, websocket: WebSocket) -> None:
        """
        Remove sessions associated with a WebSocket connection.

        Args:
            websocket: The WebSocket connection to clean up
        """
        to_remove = [
            thread_id
            for thread_id, session in self.sessions.items()
            if session["websocket"] == websocket
        ]
        for thread_id in to_remove:
            del self.sessions[thread_id]
            logger.info(f"Cleaned up session: {thread_id}")

    def get_all_active_sessions(self) -> Dict[str, dict]:
        """
        Get all currently active sessions.

        Returns:
            Dict of all sessions
        """
        return self.sessions.copy()
