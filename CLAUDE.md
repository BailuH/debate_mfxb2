# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Digital Court (数字法庭)** - an AI-powered courtroom simulation platform for legal training. It uses LangGraph's multi-agent orchestration to simulate realistic Chinese criminal trial proceedings.

### Technology Stack
- **Python 3.13** with type hints
- **LangGraph** - Multi-agent orchestration and state management
- **LangChain** - LLM integration framework
- **DeepSeek-V3 / DeepSeek-R1** - AI models via OpenAI-compatible API
- **uv** - Package manager

## Development Commands

```bash
# Install/Update dependencies
uv sync

# Run the courtroom simulation
python src/test.py

# Serve via LangGraph API (for web interface)
langgraph dev
```

## Architecture Overview

### Core Components

**State Management (`src/state.py`)**
- `CourtState`: Central state object managing the entire trial
- `case_info`: Case metadata including prosecutor/defendant details, court info
- `Evidence`: Evidence structure with id, name, content, and provider
- Enums: `PhaseEnum` (OPENING, INVESTIGATION, DEBATE, VERDICT), `ProviderEnum`, `Evidence_Show_Enum`

**Multi-Agent System (`src/agents/`)**
- **Judge**: Controls proceedings, identifies issues, delivers verdict
- **Prosecutor**: Presents charges, questions defendant, presents evidence
- **Defendant**: Responds to charges, presents defense, cross-examines
- **Clerk**: Announces rules and procedures

**Workflow (`src/test.py`)**
- LangGraph StateGraph with 46 nodes
- Conditional edges for dynamic routing based on trial state
- Uses `Command` pattern for agent-driven navigation

**LLM Configuration (`src/llmconfig.py`)**
- `models` dict containing "DeepSeek_V3" and "DeepSeek_R1" instances
- Loads config from `.env` file

### Trial Flow

1. **Opening Phase**: Clerk announces rules → Judge opens court → Verifies defendant identity
2. **Investigation Phase**: Prosecutor reads indictment → Q&A rounds (3 rounds) → Evidence presentation (3 rounds)
3. **Debate Phase**: Final arguments → Focus identification → Debate on focus points (2 rounds)
4. **Verdict Phase**: Judge delivers final judgment

### Key Architecture Patterns

- **Command-based routing**: Agents use `Command` to dynamically control graph navigation (see `defense_defense_object_control`, `pros_evidence_decision`, etc.)
- **Round counters**: `pros_question_rounds`, `pros_evidence_rounds`, `pros_focus_rounds` control iteration limits
- **State persistence**: LangGraph checkpoint system maintains conversation history
- **Agent specialization**: Each agent has specific prompts and responsibilities

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
