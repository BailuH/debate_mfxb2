"""LangGraph execution manager for handling courtroom simulation."""
import logging
from typing import Any, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.test import raw_graph
from src.state import CourtState, case_info, Evidence, PhaseEnum, Evidence_Show_Enum
from src.api.schemas import (
    InterruptRequestData,
    NodeExecutedData,
    TrialCompletedData,
    ErrorData,
)

logger = logging.getLogger(__name__)

# Progress tracking for all nodes in the trial flow
PROGRESS_NODES = {
    "START": 0,
    "clerk_rules": 1,
    "judge_open": 2,
    "judge_check": 3,
    "right_notify": 4,
    "pros_indictment": 5,
    "defense_objection": 10,
    "pros_question": 15,
    "defense_reply": 20,
    "defense_question": 25,
    "pros_summary": 30,
    "defense_summary": 35,
    "judge_start_evidence": 40,
    "pros_show_evidence": 45,
    "defense_cross": 50,
    "judge_confirm": 55,
    "defense_show_evidence": 60,
    "pros_cross": 65,
    "judge_start_debate": 70,
    "pros_statement": 75,
    "defense_self_statement": 77,
    "defense_statement": 80,
    "judge_summary": 85,
    "pros_focus": 88,
    "defense_focus": 90,
    "pros_sumup": 93,
    "defense_sumup": 95,
    "defense_final_statement": 98,
    "judge_verdict": 100,
    "END": 100,
}

# Map interrupt nodes to their expected input types and prompts
INTERRUPT_INPUT_TYPES = {
    "defense_defense_object_control": {
        "type": "boolean",
        "prompt": "公诉人现在已经念完了起诉书，你是否有异议？(true/false)",
    },
    "defense_objection": {
        "type": "string",
        "prompt": "请发表对起诉书的异议：",
    },
    "defense_question_control": {
        "type": "boolean",
        "prompt": "你是否还需要向被告人提问？",
    },
    "defense_question": {
        "type": "string",
        "prompt": "请输入你想提问的问题：",
    },
    "defense_summary": {
        "type": "string",
        "prompt": "请输入你的问被告小结：",
    },
    "defense_cross": {
        "type": "string",
        "prompt": "请输入你的质证意见：",
    },
    "defense_evidence_control": {
        "type": "boolean",
        "prompt": "你是否有补充证据要提出？",
    },
    "defense_show_evidence": {
        "type": "evidence",
        "prompt": "请输入补充证据以及证据意见（JSON格式）",
    },
    "defense_statement": {
        "type": "string",
        "prompt": "请输入你的第一轮辩护意见：",
    },
    "defense_focus": {
        "type": "string",
        "prompt": "请输入你针对当前争议焦点的回应：",
    },
    "defense_sumup": {
        "type": "string",
        "prompt": "请输入你的总结的辩护意见：",
    },
}


def validate_user_input(node_name: str, input_data: Any) -> bool:
    """
    Validate user input against expected type for an interrupt node.

    Args:
        node_name: The interrupt node name
        input_data: The user input to validate

    Returns:
        True if valid

    Raises:
        ValueError: If input doesn't match expected type
    """
    input_spec = INTERRUPT_INPUT_TYPES.get(node_name)
    if not input_spec:
        raise ValueError(f"Unknown interrupt node: {node_name}")

    input_type = input_spec["type"]

    if input_type == "boolean":
        if not isinstance(input_data, bool):
            raise ValueError(f"Expected boolean input for {node_name}, got {type(input_data)}")
    elif input_type == "string":
        if not isinstance(input_data, str):
            raise ValueError(f"Expected string input for {node_name}, got {type(input_data)}")
    elif input_type == "evidence":
        if not isinstance(input_data, dict):
            raise ValueError(f"Expected dict input for {node_name}, got {type(input_data)}")
        # Could add deeper validation here for evidence structure

    return True


def calculate_progress(node_name: str) -> float:
    """
    Calculate progress percentage based on current node.

    Args:
        node_name: The current node name

    Returns:
        Progress percentage (0-100)
    """
    return float(PROGRESS_NODES.get(node_name, 0))


def serialize_messages(messages: list[HumanMessage | AIMessage]) -> list[dict[str, Any]]:
    """
    Convert LangChain messages to JSON-serializable format.

    Args:
        messages: List of LangChain messages

    Returns:
        List of serialized message dicts
    """
    serialized = []
    for msg in messages:
        serialized.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "name": getattr(msg, "name", None),
        })
    return serialized


def serialize_state(state: CourtState | dict[str, Any]) -> dict[str, Any]:
    """
    Serialize CourtState to JSON-compatible dict.

    Args:
        state: The CourtState to serialize (can be object or dict)

    Returns:
        JSON-serializable dict representation
    """
    if isinstance(state, dict):
        # Already a dict, just ensure it's JSON-serializable
        phase = state.get("phase")
        phase_value = phase.value if hasattr(phase, "value") else phase

        messages = state.get("messages", [])
        evidence_list = state.get("evidence_list", [])
        focus = state.get("focus", [])

        return {
            "messages": serialize_messages(messages) if messages else [],
            "phase": phase_value,
            "evidence_list": [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence_list],
            "focus": list(focus) if focus else [],
            "rounds": {
                "pros_question_rounds": state.get("pros_question_rounds", 0),
                "pros_evidence_rounds": state.get("pros_evidence_rounds", 0),
                "pros_focus_rounds": state.get("pros_focus_rounds", 0),
            },
            "focus_index": state.get("focus_index", 0),
        }
    else:
        # CourtState object
        return {
            "messages": serialize_messages(state.messages),
            "phase": state.phase.value,
            "evidence_list": [e.model_dump() for e in state.evidence_list],
            "focus": state.focus,
            "rounds": {
                "pros_question_rounds": state.pros_question_rounds,
                "pros_evidence_rounds": state.pros_evidence_rounds,
                "pros_focus_rounds": state.pros_focus_rounds,
            },
            "focus_index": state.focus_index,
        }


def _state_to_dict(state: Any) -> dict[str, Any] | None:
    """
    Convert a state object (CourtState or dict-like) to a standard dict.

    Args:
        state: The state to convert

    Returns:
        A standard dict, or None if state is None
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return state
    # Try to convert to dict if it has dict-like interface
    if hasattr(state, "dict"):
        return state.dict()
    if hasattr(state, "model_dump"):
        return state.model_dump()
    # Fallback: try to cast to dict
    try:
        return dict(state)
    except (TypeError, ValueError):
        return None


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to make it JSON serializable.

    Args:
        obj: The object to sanitize

    Returns:
        A JSON-serializable version of the object
    """
    from langchain_core.messages import BaseMessage

    if isinstance(obj, BaseMessage):
        return {
            "type": "ai" if obj.__class__.__name__ == "AIMessage" else "human",
            "content": obj.content,
            "name": getattr(obj, "name", None),
        }
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif hasattr(obj, "value"):  # Handle Enum types
        return obj.value
    elif hasattr(obj, "model_dump"):  # Handle Pydantic models
        return _sanitize_for_json(obj.model_dump())
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback: try to convert to string
        return str(obj)


class AsyncGraphExecutor:
    """Handles async execution of the LangGraph courtroom simulation."""

    def __init__(self, memory: MemorySaver, session_manager):
        """
        Initialize the graph executor.

        Args:
            memory: LangGraph MemorySaver for checkpointing
            session_manager: SessionManager instance for tracking sessions
        """
        # Compile the graph with checkpointer
        self.app = raw_graph.compile(checkpointer=memory)
        self.session_manager = session_manager
        self.interrupt_map = INTERRUPT_INPUT_TYPES
        logger.info("AsyncGraphExecutor initialized with MemorySaver")

    async def execute_trial(
        self,
        initial_state: dict[str, Any],
        config: dict[str, Any],
        websocket_manager,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Execute the trial graph and yield events to WebSocket.

        Args:
            initial_state: The initial state to start the trial with
            config: LangGraph config containing thread_id
            websocket_manager: ConnectionManager for sending messages

        Yields:
            Event dicts to be sent to the WebSocket client
        """
        thread_id = config["configurable"]["thread_id"]

        try:
            logger.info(f"Starting trial execution for thread: {thread_id}")

            # Stream execution (use astream for async iteration)
            async for event in self.app.astream(initial_state, config):

                # Check for interrupt
                if "__interrupt__" in event:
                    interrupt_info = event["__interrupt__"]
                    logger.info(f"Interrupt detected for thread {thread_id}: {interrupt_info}")

                    # Get the node name from interrupt info
                    # The interrupt info structure depends on LangGraph version
                    # Try to extract node name
                    node_name = None
                    if isinstance(interrupt_info, dict):
                        node_name = interrupt_info.get("value") or interrupt_info.get("name")
                    elif hasattr(interrupt_info, "value"):
                        node_name = interrupt_info.value

                    # Fallback: check current state
                    if not node_name:
                        state_snapshot = self.app.get_state(config)
                        if state_snapshot and state_snapshot.next:
                            # next may be a tuple, extract first element
                            next_node = state_snapshot.next
                            node_name = next_node[0] if isinstance(next_node, (tuple, list)) else next_node

                    # Ensure node_name is a string
                    if isinstance(node_name, (tuple, list)):
                        node_name = node_name[0] if len(node_name) > 0 else None

                    # Get interrupt specification
                    interrupt_spec = self.interrupt_map.get(node_name, {})
                    if not interrupt_spec:
                        logger.warning(f"No interrupt spec found for node: {node_name}")
                        interrupt_spec = {"type": "string", "prompt": "请输入："}

                    # Get current phase
                    current_phase = self._get_current_phase(config)

                    # Ensure node_name is a string format before sending
                    clean_node_name = node_name[0] if isinstance(node_name, (tuple, list)) else node_name

                    # Send interrupt request to client
                    await websocket_manager.send_to_thread(
                        {
                            "type": "interrupt_request",
                            "thread_id": thread_id,
                            "data": {
                                "node_name": clean_node_name,
                                "prompt": interrupt_spec.get("prompt", ""),
                                "input_type": interrupt_spec.get("type", "string"),
                                "options": interrupt_spec.get("options"),
                                "metadata": {
                                    "phase": current_phase,
                                },
                            },
                        },
                        thread_id,
                    )

                    # Update session status
                    self.session_manager.update_session(
                        thread_id,
                        status="interrupted",
                        current_node=clean_node_name,
                    )

                    # Stop here - execution will continue on resume
                    logger.info(f"Waiting for user input at {node_name}")
                    return

                # Process node execution events
                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue

                    logger.debug(f"Node executed: {node_name}")

                    # Get full state snapshot
                    state_snapshot = self.app.get_state(config)
                    full_state = _state_to_dict(state_snapshot.values) if state_snapshot else None

                    # Extract and send state updates
                    state_updates = self._extract_state_updates(
                        node_name,
                        node_output,
                        full_state,
                    )

                    await websocket_manager.send_to_thread(
                        {
                            "type": "node_executed",
                            "thread_id": thread_id,
                            "data": state_updates,
                        },
                        thread_id,
                    )

                    # Update session
                    self.session_manager.update_session(
                        thread_id,
                        status="running",
                        current_node=node_name,
                    )

            # Trial completed
            logger.info(f"Trial completed for thread: {thread_id}")
            final_state_snapshot = self.app.get_state(config)
            final_state = _state_to_dict(final_state_snapshot.values) if final_state_snapshot else None

            await websocket_manager.send_to_thread(
                {
                    "type": "trial_completed",
                    "thread_id": thread_id,
                    "data": {
                        "final_state": serialize_state(final_state) if final_state else {},
                    },
                },
                thread_id,
            )

            self.session_manager.update_session(
                thread_id,
                status="completed",
                current_node=None,
            )

        except Exception as e:
            logger.error(f"Error executing trial for thread {thread_id}: {e}", exc_info=True)
            await websocket_manager.send_to_thread(
                {
                    "type": "error",
                    "thread_id": thread_id,
                    "data": {
                        "code": "GRAPH_EXECUTION_ERROR",
                        "message": str(e),
                    },
                },
                thread_id,
            )

    async def resume_execution(
        self,
        thread_id: str,
        user_input: Any,
        config: dict[str, Any],
        websocket_manager,
    ) -> None:
        """
        Resume graph execution after an interrupt with user input.

        Args:
            thread_id: The thread ID to resume
            user_input: The user input to resume with
            config: LangGraph config containing thread_id
            websocket_manager: ConnectionManager for sending messages
        """
        try:
            logger.info(f"Resuming execution for thread: {thread_id} with input: {user_input}")

            # Update session status
            self.session_manager.update_session(thread_id, status="running")

            # Resume execution with Command (use astream for async iteration)
            async for event in self.app.astream(Command(resume=user_input), config):

                # Check for another interrupt
                if "__interrupt__" in event:
                    interrupt_info = event["__interrupt__"]
                    logger.info(f"Interrupt detected during resume for thread {thread_id}")

                    # Extract node name
                    node_name = None
                    if isinstance(interrupt_info, dict):
                        node_name = interrupt_info.get("value") or interrupt_info.get("name")
                    elif hasattr(interrupt_info, "value"):
                        node_name = interrupt_info.value

                    if not node_name:
                        state_snapshot = self.app.get_state(config)
                        if state_snapshot and state_snapshot.next:
                            # next may be a tuple, extract first element
                            next_node = state_snapshot.next
                            node_name = next_node[0] if isinstance(next_node, (tuple, list)) else next_node

                    # Ensure node_name is a string
                    if isinstance(node_name, (tuple, list)):
                        node_name = node_name[0] if len(node_name) > 0 else None

                    interrupt_spec = self.interrupt_map.get(node_name, {})
                    if not interrupt_spec:
                        interrupt_spec = {"type": "string", "prompt": "请输入："}

                    current_phase = self._get_current_phase(config)

                    # Ensure node_name is a string format before sending
                    clean_node_name = node_name[0] if isinstance(node_name, (tuple, list)) else node_name

                    # Send interrupt request
                    await websocket_manager.send_to_thread(
                        {
                            "type": "interrupt_request",
                            "thread_id": thread_id,
                            "data": {
                                "node_name": clean_node_name,
                                "prompt": interrupt_spec.get("prompt", ""),
                                "input_type": interrupt_spec.get("type", "string"),
                                "options": interrupt_spec.get("options"),
                                "metadata": {
                                    "phase": current_phase,
                                },
                            },
                        },
                        thread_id,
                    )

                    self.session_manager.update_session(
                        thread_id,
                        status="interrupted",
                        current_node=clean_node_name,
                    )
                    return

                # Process node execution
                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue

                    logger.debug(f"Node executed (resume): {node_name}")

                    state_snapshot = self.app.get_state(config)
                    full_state = _state_to_dict(state_snapshot.values) if state_snapshot else None

                    state_updates = self._extract_state_updates(
                        node_name,
                        node_output,
                        full_state,
                    )

                    await websocket_manager.send_to_thread(
                        {
                            "type": "node_executed",
                            "thread_id": thread_id,
                            "data": state_updates,
                        },
                        thread_id,
                    )

                    self.session_manager.update_session(
                        thread_id,
                        status="running",
                        current_node=node_name,
                    )

            # Check if trial is complete
            state_snapshot = self.app.get_state(config)
            if state_snapshot and state_snapshot.next == ():
                logger.info(f"Trial completed after resume for thread: {thread_id}")
                final_state = _state_to_dict(state_snapshot.values)

                await websocket_manager.send_to_thread(
                    {
                        "type": "trial_completed",
                        "thread_id": thread_id,
                        "data": {
                            "final_state": serialize_state(final_state),
                        },
                    },
                    thread_id,
                )

                self.session_manager.update_session(
                    thread_id,
                    status="completed",
                    current_node=None,
                )

        except Exception as e:
            logger.error(f"Error resuming execution for thread {thread_id}: {e}", exc_info=True)
            await websocket_manager.send_to_thread(
                {
                    "type": "error",
                    "thread_id": thread_id,
                    "data": {
                        "code": "RESUME_ERROR",
                        "message": str(e),
                    },
                },
                thread_id,
            )

    def _extract_state_updates(
        self,
        node_name: str,
        node_output: dict[str, Any],
        full_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Extract relevant state updates to send to frontend.

        Args:
            node_name: The node that was executed
            node_output: The output from the node
            full_state: The full state snapshot as dict

        Returns:
            Dict of state updates (JSON-serializable)
        """
        # Handle phase - could be PhaseEnum or string
        phase = "unknown"
        if full_state:
            p = full_state.get("phase")
            if p:
                phase = p.value if hasattr(p, "value") else p

        # Get messages safely
        messages = full_state.get("messages", []) if full_state else []

        # Sanitize state_delta to ensure JSON serializability
        sanitized_delta = _sanitize_for_json(node_output)

        return {
            "node_name": node_name,
            "state_delta": sanitized_delta,
            "current_phase": phase,
            "progress": calculate_progress(node_name),
            "message_count": len(messages),
            "focus": list(full_state.get("focus", [])) if full_state else [],
            "rounds": {
                "pros_question_rounds": full_state.get("pros_question_rounds", 0) if full_state else 0,
                "pros_evidence_rounds": full_state.get("pros_evidence_rounds", 0) if full_state else 0,
                "pros_focus_rounds": full_state.get("pros_focus_rounds", 0) if full_state else 0,
            },
            "messages": serialize_messages(messages[-5:]) if messages else [],
        }

    def _get_current_phase(self, config: dict[str, Any]) -> str:
        """
        Get current trial phase from state.

        Args:
            config: LangGraph config

        Returns:
            Phase string value
        """
        state_snapshot = self.app.get_state(config)
        if state_snapshot and state_snapshot.values:
            state_dict = _state_to_dict(state_snapshot.values)
            if state_dict:
                phase = state_dict.get("phase")
                if phase:
                    return phase.value if hasattr(phase, "value") else phase
        return "unknown"
