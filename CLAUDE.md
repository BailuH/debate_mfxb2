# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Digital Court (数字法庭)** - an AI-powered courtroom simulation platform for legal training. It uses LangGraph's multi-agent orchestration to simulate realistic Chinese criminal trial proceedings with real-time WebSocket communication.

### Technology Stack
- **Python 3.13** with type hints
- **LangGraph** - Multi-agent orchestration and state management
- **LangChain** - LLM integration framework
- **FastAPI** - REST/WebSocket API backend
- **Vue.js 3** + Tailwind CSS - Frontend (demo.html)
- **DeepSeek-V3 / DeepSeek-R1 / KIMI-K2.5** - AI models via OpenAI-compatible API
- **uv** - Package manager

## Development Commands

```bash
# Install/Update dependencies
uv sync

# Run the courtroom simulation (direct execution)
python src/test.py

# Start FastAPI backend server
uvicorn src.api.main:app --reload

# Serve via LangGraph API (alternative)
langgraph dev
```

## Architecture Overview

### Project Structure

```
debate_mfxb2/
├── src/
│   ├── agents/           # AI agent implementations
│   │   ├── clerk.py      # Court clerk agent
│   │   ├── denfendant.py # Defendant agent (note: typo in filename)
│   │   ├── judge.py      # Judge agent
│   │   └── prosecutor.py # Prosecutor agent
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # FastAPI application entry point
│   │   ├── websocket.py  # WebSocket connection manager
│   │   ├── session_manager.py  # Session management
│   │   ├── graph_manager.py    # LangGraph execution manager
│   │   ├── schemas.py    # Pydantic models for WebSocket messages
│   │   └── config.py     # API configuration settings
│   ├── state.py          # State management and data models
│   ├── test.py           # LangGraph workflow definition (46 nodes)
│   ├── llmconfig.py      # LLM model configurations
│   ├── llm_wrapper.py    # LLM call wrapper with retry logic
│   └── prompt.py         # Agent prompt templates
├── demo.html             # Vue.js frontend demo
├── langgraph.json        # LangGraph configuration
└── main.py               # Main entry point
```

### Core Components

**State Management (`src/state.py`)**
- `CourtState`: Central state object managing the entire trial
- `case_info`: Case metadata including prosecutor/defendant details, court info
- `Evidence`: Evidence structure with id, name, content, and provider
- Enums:
  - `PhaseEnum`: OPENING, INVESTIGATION, DEBATE, VERDICT
  - `ProviderEnum`: PROSECUTOR, DEFENSE, JUDGE
  - `Evidence_Show_Enum`: SINGLE, MULTIPLE
- Constants: `Q_ROUNDS = 3`, `E_ROUNDS = 3`, `F_ROUNDS = 2`

**Multi-Agent System (`src/agents/`)**
- **Judge** (`judge.py`): Controls proceedings, identifies issues, delivers verdict
- **Prosecutor** (`prosecutor.py`): Presents charges, questions defendant, presents evidence
- **Defendant** (`denfendant.py`): Responds to charges, presents defense, cross-examines
- **Clerk** (`clerk.py`): Announces rules and procedures

**LangGraph Workflow (`src/test.py`)**
- LangGraph StateGraph with 46 nodes
- Conditional edges for dynamic routing based on trial state
- Uses `Command` pattern for agent-driven navigation
- Interrupt nodes for human interaction points

**API Backend (`src/api/`)**

*`main.py` - FastAPI Application*
- WebSocket endpoint: `/ws/trial`
- Health check: `/health`
- Configuration endpoint: `/config`
- Lifespan management for startup/shutdown

*`websocket.py` - Connection Manager*
- `ConnectionManager` class manages active WebSocket connections
- Methods: `connect()`, `disconnect()`, `send_personal_message()`, `send_to_thread()`, `broadcast()`

*`session_manager.py` - Session Management*
- `SessionManager` class handles trial sessions with UUID-based thread IDs
- Integrates with LangGraph MemorySaver for state persistence
- Tracks session status: active, interrupted, completed

*`graph_manager.py` - Graph Execution*
- `AsyncGraphExecutor` handles async LangGraph execution
- Progress tracking with node-based percentages (0-100)
- Interrupt handling with user input validation
- Message serialization for WebSocket transmission
- `INTERRUPT_INPUT_TYPES` maps interrupt nodes to expected input types:
  - `defense_defense_object_control`: boolean
  - `defense_objection`: string
  - `defense_question_control`: boolean
  - `defense_question`: string
  - `defense_cross`: string
  - `defense_evidence_control`: boolean
  - `defense_show_evidence`: dict (evidence)

*`schemas.py` - Pydantic Models*
- `MessageType`: Enum for WebSocket message types
- `WSMessage`: Base WebSocket message wrapper
- Request models: `StartTrialRequest`, `UserInputRequest`
- Response models: `SessionCreatedData`, `NodeExecutedData`, `InterruptRequestData`, `TrialCompletedData`, `ErrorData`

*`config.py` - Configuration*
- `Settings` class with Pydantic settings
- CORS origins configuration
- Checkpoint namespace settings

**LLM Configuration (`src/llmconfig.py`)**
- `models` dict containing:
  - `DeepSeek_V3`: ChatOpenAI instance
  - `DeepSeek_R1`: ChatOpenAI instance
  - `KIMI_K2.5`: ChatOpenAI instance
- Loads config from `.env` file

**LLM Wrapper (`src/llm_wrapper.py`)**
- `LLMWrapper` class provides retry logic for rate limit errors
- Exponential backoff: initial 2s delay, max 60s, base 2.0
- Max retries: 5
- Only retries `RateLimitError`, other exceptions propagate

### Trial Flow

1. **Opening Phase**: Clerk announces rules → Judge opens court → Verifies defendant identity
2. **Investigation Phase**: Prosecutor reads indictment → Q&A rounds (3 rounds) → Evidence presentation (3 rounds)
3. **Debate Phase**: Final arguments → Focus identification → Debate on focus points (2 rounds)
4. **Verdict Phase**: Judge delivers final judgment

### Key Architecture Patterns

- **Command-based routing**: Agents use `Command` to dynamically control graph navigation
- **Round counters**: `pros_question_rounds`, `pros_evidence_rounds`, `pros_focus_rounds` control iteration limits
- **State persistence**: LangGraph checkpoint system maintains conversation history
- **Agent specialization**: Each agent has specific prompts and responsibilities
- **Interrupt-driven interaction**: Strategic points for human input during trial
- **Async execution**: Full async/await support for WebSocket communication

### WebSocket Message Flow

**Client → Server:**
- `start_trial`: Initialize new session with case_info and evidence_list
- `user_input`: Respond to interrupt with user input
- `ping`: Keep-alive

**Server → Client:**
- `session_created`: New session established with thread_id
- `node_executed`: Node completed with state_delta, progress, phase
- `interrupt_request`: Human input required with prompt and input_type
- `trial_completed`: Trial finished with final_state
- `error`: Error occurred with code and message
- `pong`: Response to ping

### Graph Entry Point

The main graph is defined in `src/test.py` and exported as `app`. This is referenced in `langgraph.json` as the "agents" graph.

## Configuration

Environment variables (`.env`):
- `OPENAI_API_KEY` - DeepSeek API key
- `OPENAI_API_BASE` - DeepSeek API endpoint

## Constants (src/state.py)

- `Q_ROUNDS = 3` - Default prosecutor questioning rounds
- `E_ROUNDS = 3` - Default evidence presentation rounds
- `F_ROUNDS = 2` - Default focus debate rounds

## Progress Tracking (src/api/graph_manager.py)

Nodes are mapped to progress values:
- 0-4: Opening phase
- 5-39: Investigation phase
- 40-98: Debate phase
- 100: Verdict phase (END)
