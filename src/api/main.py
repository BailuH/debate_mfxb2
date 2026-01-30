"""FastAPI application for Digital Court WebSocket backend."""
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.websocket import manager
from src.api.session_manager import SessionManager
from src.api.graph_manager import AsyncGraphExecutor, validate_user_input, INTERRUPT_INPUT_TYPES
from src.state import case_info, Evidence, PhaseEnum, Evidence_Show_Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global components
memory = None
session_manager = None
graph_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global memory, session_manager, graph_executor

    # Startup
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Starting Digital Court API...")
    memory = MemorySaver()
    session_manager = SessionManager(memory)
    graph_executor = AsyncGraphExecutor(memory, session_manager)
    logger.info("Graph executor initialized with MemorySaver")

    yield

    # Shutdown
    logger.info("Shutting down Digital Court API...")


# Initialize FastAPI
app = FastAPI(
    title="Digital Court API",
    version="0.1.0",
    description="WebSocket API for AI-powered courtroom simulation",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


@app.get("/config")
async def get_config():
    """Get API configuration and available interrupt nodes."""
    return {
        "interrupt_nodes": list(INTERRUPT_INPUT_TYPES.keys()),
        "phases": ["opening", "investigation", "debate", "verdict"],
        "version": "0.1.0",
    }


@app.websocket("/ws/trial")
async def websocket_trial_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for trial communication.

    Handles the following message types from client:
    - start_trial: Initialize a new trial session
    - user_input: Respond to an interrupt with user input
    - ping: Keep-alive ping

    Sends the following message types to client:
    - session_created: New session established with thread_id
    - node_executed: A node completed execution
    - interrupt_request: Human input required
    - trial_completed: Trial finished
    - error: An error occurred
    - pong: Response to ping
    """
    await manager.connect(websocket)
    thread_id = None

    try:
        logger.info("WebSocket connection established")

        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            message_data = data.get("data", {})

            logger.debug(f"Received message: {message_type}")

            if message_type == "start_trial":
                # Create new session
                thread_id = session_manager.create_session(websocket)
                session = session_manager.get_session(thread_id)
                config = session["config"]

                # Associate this WebSocket with the thread_id
                manager.associate_thread(websocket, thread_id)

                logger.info(f"Created new trial session: {thread_id}")

                # Send session_created response
                await manager.send_personal_message(
                    {
                        "type": "session_created",
                        "thread_id": thread_id,
                        "data": {
                            "message": "Trial session created successfully",
                            "thread_id": thread_id,
                        },
                    },
                    websocket,
                )

                # Prepare initial state from client data
                case_info_data = message_data.get("case_info", {})
                evidence_data = message_data.get("evidence_list", [])

                try:
                    initial_state = {
                        "messages": [],
                        "focus": [],
                        "phase": PhaseEnum.OPENING,
                        "evidence_list": [Evidence(**ev) for ev in evidence_data],
                        "current_evidence": None,
                        "evidence_show_type": Evidence_Show_Enum.SINGLE,
                        "meta": case_info(**case_info_data),
                        "pros_question_rounds": 3,
                        "pros_evidence_rounds": 3,
                        "pros_focus_rounds": 2,
                        "focus_index": 0,
                    }
                except Exception as e:
                    logger.error(f"Error creating initial state: {e}")
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "thread_id": thread_id,
                            "data": {
                                "code": "INVALID_INITIAL_STATE",
                                "message": f"Invalid case_info or evidence data: {e}",
                            },
                        },
                        websocket,
                    )
                    continue

                # Start graph execution
                await graph_executor.execute_trial(initial_state, config, manager)

            elif message_type == "user_input":
                # Resume execution with user input
                incoming_thread_id = data.get("thread_id")
                if not incoming_thread_id:
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "data": {
                                "code": "MISSING_THREAD_ID",
                                "message": "thread_id is required for user_input",
                            },
                        },
                        websocket,
                    )
                    continue

                if not session_manager.get_session(incoming_thread_id):
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "thread_id": incoming_thread_id,
                            "data": {
                                "code": "SESSION_NOT_FOUND",
                                "message": f"Thread {incoming_thread_id} not found",
                            },
                        },
                        websocket,
                    )
                    continue

                thread_id = incoming_thread_id
                interrupt_node = message_data.get("interrupt_node")
                user_input = message_data.get("input")

                # Validate input
                try:
                    validate_user_input(interrupt_node, user_input)
                except ValueError as e:
                    logger.warning(f"Invalid user input for {interrupt_node}: {e}")
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "thread_id": thread_id,
                            "data": {
                                "code": "INVALID_INPUT",
                                "message": str(e),
                                "node": interrupt_node,
                            },
                        },
                        websocket,
                    )
                    continue

                # Resume execution
                session = session_manager.get_session(thread_id)
                config = session["config"]

                await graph_executor.resume_execution(
                    thread_id,
                    user_input,
                    config,
                    manager,
                )

            elif message_type == "ping":
                # Pong back
                await manager.send_personal_message(
                    {
                        "type": "pong",
                        "data": {},
                    },
                    websocket,
                )

            else:
                logger.warning(f"Unknown message type: {message_type}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "data": {
                            "code": "UNKNOWN_MESSAGE_TYPE",
                            "message": f"Unknown message type: {message_type}",
                        },
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {thread_id}")
        manager.disconnect(websocket)
        if thread_id:
            await session_manager.cleanup_session(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "thread_id": thread_id,
                    "data": {
                        "code": "WEBSOCKET_ERROR",
                        "message": str(e),
                    },
                },
                websocket,
            )
        except Exception:
            pass
        manager.disconnect(websocket)
        if thread_id:
            await session_manager.cleanup_session(websocket)
