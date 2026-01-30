"""WebSocket connection manager for handling active connections."""
from fastapi import WebSocket
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for trial sessions."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: List[WebSocket] = []
        self.thread_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, thread_id: str | None = None) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept
            thread_id: Optional thread ID to associate with the connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        if thread_id:
            self.thread_connections[thread_id] = websocket
        logger.info(f"WebSocket connected. Thread: {thread_id}, Active: {len(self.active_connections)}")

    def associate_thread(self, websocket: WebSocket, thread_id: str) -> None:
        """
        Associate a WebSocket connection with a thread_id.
        Called after a session is created.

        Args:
            websocket: The WebSocket connection to associate
            thread_id: The thread ID to associate with the connection
        """
        self.thread_connections[thread_id] = websocket
        logger.info(f"Associated thread_id {thread_id} with WebSocket")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection from tracking.

        Args:
            websocket: The WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from thread_connections
        to_remove = [
            thread_id for thread_id, ws in self.thread_connections.items()
            if ws == websocket
        ]
        for thread_id in to_remove:
            del self.thread_connections[thread_id]
            logger.info(f"WebSocket disconnected for thread: {thread_id}")

        logger.info(f"Active connections remaining: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket) -> None:
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: The message dict to send (will be JSON serialized)
            websocket: The target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            raise

    async def send_to_thread(self, message: dict, thread_id: str) -> None:
        """
        Send a message to the WebSocket associated with a specific thread.

        Args:
            message: The message dict to send
            thread_id: The thread ID to send the message to
        """
        if thread_id not in self.thread_connections:
            logger.warning(f"No WebSocket found for thread: {thread_id}")
            return

        websocket = self.thread_connections[thread_id]
        await self.send_personal_message(message, websocket)

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast a message to all active connections.

        Args:
            message: The message dict to broadcast
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")


# Global connection manager instance
manager = ConnectionManager()
