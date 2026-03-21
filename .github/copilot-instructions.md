# GitHub Copilot Instructions — Video Workflow Agent

## Project Overview
This is a **multi-step agentic pipeline** that:
1. Takes a video (screen recording or talking-head)
2. Transcribes it via Whisper (local or OpenAI API)
3. Extracts semantically unique frames (scene changes, ignoring mouse movement)
4. Describes each frame using a vision LLM with chained context (each frame sees the previous frame's description)
5. Passes through a 4-agent LangGraph pipeline:
   - **Requirements Streamliner** → distills workflow intent from transcript + frame descriptions
   - **Researcher** → uses Tavily web search to generate recommendations
   - **Alignment PM** → checks if research aligns with requirements (confident/not-confident)
   - **Synthesis** → final structured output
6. If Alignment is NOT confident → loops back to Researcher with feedback notes
7. All agent outputs are persisted to `state_outputs/<run_id>/<agent>.sh` (shell-sourceable) and `<agent>.md` (human-readable markdown)
8. UI is built in **Streamlit**, orchestration in **LangGraph**

## LLM Provider
This project uses the **OpenAI SDK** for all LLM calls. Two providers are supported:

- **LM Studio** (default): OpenAI-compatible local server at `LM_STUDIO_BASE_URL` (default `http://localhost:1234/v1`)
- **OpenAI**: Standard OpenAI API using `OPENAI_API_KEY`

All provider logic lives in `utils/llm_client.py`. Use the `chat()` function for simple completions and `get_client()`/`get_model()` for tool-use loops.

### ⚠️ Thinking-Model Compatibility
Models like **Qwen3** (including `qwen3.5-9b`) use extended thinking mode by default in LM Studio. This puts all reasoning in `reasoning_content` and may return empty `content` if `max_tokens` is exhausted during thinking. Always use:
- `max_tokens=8000` minimum for paragraph-length outputs
- `max_tokens=16000` for tool-use loops in research_agent
- The `chat()` function in `llm_client.py` automatically falls back to `reasoning_content` when `content` is empty

## Key Conventions

### State Files (.sh + .md format)
Every agent MUST write its output via `utils/state_manager.write_agent_state()`.
This writes two files per run:
- `state_outputs/<run_id>/<agent>.sh` — shell-sourceable metadata
- `state_outputs/<run_id>/<agent>.md` — human-readable full output (for requirements, research, alignment, synthesis agents)

```bash
export AGENT='requirements'
export STATUS='complete'           # complete | running | failed | waiting
export TIMESTAMP='2025-01-01T00:00:00Z'
export ITERATION='0'
export OUTPUT_SUMMARY='One-line summary'
export OUTPUT_FULL='Multi-line full output with escaped single quotes'
```

Never write these files directly — always use `utils/state_manager.py`.

### Per-Run Folders
Each pipeline run creates a unique folder: `state_outputs/run_YYYYMMDD_HHMMSS/`
The `run_id` and `run_dir` are stored in `WorkflowState` and passed through every node.

### LangGraph State (graph/state.py)
The `WorkflowState` TypedDict is the single source of truth flowing through nodes.
Never mutate state outside of node return values.

### Agent Nodes (graph/nodes.py)
Each node function signature: `def node_name(state: WorkflowState) -> dict`
Return only the keys being updated, not the full state.
Always call `write_agent_state()` at the end of each node.

### Alignment Loop
- If `alignment_confident == False` AND `iteration_count < MAX_ITERATIONS (3)`:
  - Route back to `research_node`
  - Append alignment PM feedback to `state["alignment_notes"]`
- If confident OR max iterations reached → route to `synthesis_node`

### Frame Extraction Logic
- Use OpenCV for frame differencing
- Threshold: mean absolute difference > `FRAME_DIFF_THRESHOLD` (default 30)
- Skip frames where changed region fits within `MOUSE_REGION_SIZE` × `MOUSE_REGION_SIZE` (mouse movement heuristic)
- Optionally downscale to `VIDEO_MAX_WIDTH` before processing
- Save keyframes to `<run_dir>/frames/` as JPEG

### Streamlit UI Flow
- Sidebar: provider selector + pipeline status badges (one per agent, color-coded)
- Main: step-by-step collapsible outputs per agent
- Show iteration counter for the alignment loop
- Allow re-run from any step by clearing downstream .sh/.md files
- `_run_pipeline_thread()` takes `(video_path, provider, run_dir, run_id)` and uses `WorkflowState` directly (no redundant `initial_state` call)

## File Map
```
app.py                          ← Streamlit entry point
graph/
  state.py                      ← WorkflowState TypedDict + constants
  nodes.py                      ← All LangGraph node functions
  workflow.py                   ← StateGraph definition + compilation
agents/
  transcriber.py                ← Whisper transcription (local or OpenAI API)
  frame_extractor.py            ← OpenCV frame extraction + downscale
  requirements_agent.py         ← LLM requirements streamliner
  research_agent.py             ← LLM researcher with Tavily search (tool-use loop)
  alignment_agent.py            ← LLM alignment/PM judge
  synthesis_agent.py            ← LLM final synthesis
utils/
  llm_client.py                 ← Unified OpenAI SDK client factory
  state_manager.py              ← Read/write .sh + .md state files
  video_utils.py                ← Video helpers (duration, fps, audio extraction)
tests/
  run_e2e.py                    ← CLI end-to-end runner
  test_*.py                     ← pytest unit tests (54 tests)
  video/                        ← Test video assets
state_outputs/                  ← Runtime: per-run timestamped folders
  run_YYYYMMDD_HHMMSS/
    *.sh                        ← Shell-sourceable state per agent
    *.md                        ← Markdown output per agent
    frames/                     ← Extracted keyframe JPEGs
requirements.txt
.env.example
```

## Dependencies (requirements.txt)
- streamlit>=1.35
- langgraph>=0.2
- openai>=1.30.0
- openai-whisper
- opencv-python
- tavily-python
- python-dotenv
- Pillow
- numpy
- pytest>=8.0.0
- pytest-mock>=3.12.0

## Environment Variables (.env)
```
DEFAULT_PROVIDER=lm_studio       # lm_studio | openai
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3.5-9b
LM_STUDIO_VISION_MODEL=qwen/qwen3.5-9b
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o
OPENAI_VISION_MODEL=gpt-4o
TAVILY_API_KEY=your_key
WHISPER_MODEL=base          # base | small | medium | large
MAX_ALIGNMENT_ITERATIONS=3
FRAME_DIFF_THRESHOLD=30
MOUSE_REGION_SIZE=100
VIDEO_MAX_WIDTH=480         # 0 = no downscale
```

## Coding Style
- Type hints everywhere (Python 3.11+ `str | None` union syntax)
- Docstrings on every public function
- Structured logging via `logging` module (not print)
- All LLM prompts defined as module-level constants, not inline strings
- Error handling: agents catch exceptions and write STATUS="failed" to .sh files
- Tests: pytest + pytest-mock in `tests/` directory
