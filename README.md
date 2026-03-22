# FlowScope — AI Workflow Optimisation Agent

> Show a video of your business workflow. Get a decision-ready report with multiple solution paths, cost estimates, and an implementation roadmap.

FlowScope is an open-source multi-agent AI pipeline designed for **business owners and operators** who want to identify and fix operational bottlenecks. Upload a screen recording or talking-head walkthrough of any workflow and the system returns a structured optimisation report — covering everything from quick configuration tweaks to full tool replacement or custom automation.

---

## How It Works

```
Video Upload
    │
    ▼
┌─────────────────┐
│  1. Transcribe  │  Whisper transcribes the audio (local or OpenAI API)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Extract      │  OpenCV extracts unique keyframes (ignores mouse movement)
│    Frames       │  Each frame is described by a vision LLM with chained context
│                 │  Descriptions are grouped into 25-frame narrative summaries
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Requirements │  LLM analyses the workflow using transcript + 25-frame chunk
│    Analyst      │  summaries: identifies business context, maps each step,
│                 │  scores bottlenecks (severity 1-5), proposes strategies
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Researcher   │  Searches the web (Tavily) for the most relevant approaches
│                 │  for THIS specific workflow. Common types explored where
│                 │  applicable: Optimise Current │ Alternative SaaS │ No-Code
│                 │  Automation │ AI Agent │ Custom Build — but not constrained
│                 │  to these. Produces comparison tables with trade-offs/sources
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Alignment PM │  Reviews research for genuine breadth and fit for THIS
│                 │  workflow, business alignment, and actionability.
│                 │  Evaluates whether the right approaches were explored (not
│                 │  whether all 5 paradigms were covered). Confident → proceed.
│                 │  Not confident → sends feedback back to Researcher (up to 3 loops)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. Synthesis    │  Produces the final Decision-Ready Report with dynamically
│                 │  named solution paths tailored to the workflow (e.g.
│                 │  "Automate with n8n", "Migrate to Linear") + Decision Matrix
│                 │  with workflow-relevant criteria + Roadmap + Quick Wins
└─────────────────┘
```

All agent outputs are saved to `state_outputs/<run_id>/` in a nested structure, with shell-sourceable `state.sh` and human-readable `output.md` files per agent.

---

## Example Output

Given a video of a Notion-based product roadmap workflow, FlowScope identified:

- **4 bottlenecks** (Manual Data Sync severity 5/5, Hierarchical Navigation 4/5)
- **5 solution paths** with cost estimates ranging from $0 (Notion formula hack) to $80k/yr (Linear Enterprise migration)
- **Quick win**: n8n webhook ($20/mo) to auto-update task status from GitHub commits
- **Recommended path**: Linear migration with n8n bridge, saving ~40% engineering admin time

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| ffmpeg | any | Audio extraction from video |
| LM Studio **or** OpenAI API key | — | LLM inference |
| Tavily API key | — | Web search for research agent |

### Python packages

```
streamlit>=1.35
langgraph>=0.2
openai>=1.30.0
openai-whisper
opencv-python
tavily-python
python-dotenv
Pillow
numpy
pytest>=8.0.0
pytest-mock>=3.12.0
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/srbsa/flowscope.git
cd flowscope
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```dotenv
# Choose your LLM provider
DEFAULT_PROVIDER=lm_studio      # or: openai

# LM Studio (local inference — free)
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3.5-9b
LM_STUDIO_VISION_MODEL=qwen/qwen3.5-9b

# OpenAI (cloud)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_VISION_MODEL=gpt-4o

# Tavily web search (free tier available at https://tavily.com)
TAVILY_API_KEY=tvly-...

# Whisper model size: tiny | base | small | medium | large
WHISPER_MODEL=base
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a video, and click **Run Pipeline**.

---

## LLM Provider Options

### Option A — LM Studio (local, free)

1. Download [LM Studio](https://lmstudio.ai)
2. Download a model — recommended: **Qwen3 8B** or **Qwen3 14B**
3. Start the local server (default: `http://localhost:1234/v1`)
4. Set `DEFAULT_PROVIDER=lm_studio` in `.env`

> **Thinking-model note:** Qwen3 uses extended thinking mode which consumes tokens on internal reasoning. FlowScope automatically disables thinking for the synthesis step to prevent output truncation. For other agents, `max_tokens=10000` is used.

### Option B — OpenAI

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Set `OPENAI_API_KEY=sk-...` in `.env`
3. Set `DEFAULT_PROVIDER=openai`
4. Recommended models: `gpt-4o` (text + vision)

---

## Output Structure

Each pipeline run creates a timestamped folder:

```
state_outputs/
  run_20260320_184906/
    frames/                     ← extracted keyframe JPEGs
    transcriber/
      state.sh                  ← machine-readable state (shell-sourceable)
    frame_extractor/
      state.sh
      output.md                 ← frame-by-frame descriptions + 25-frame chunk summaries
    requirements/
      state.sh
      output.md                 ← bottleneck analysis, automation opportunities
    research/
      state.sh
      output.md                 ← multi-approach solution research with sources
    alignment/
      state.sh
      output.md                 ← confidence verdict + critique
    synthesis/
      state.sh
      output.md                 ← final Decision-Ready Report
```

`state.sh` files are shell-sourceable for scripting:

```bash
source state_outputs/run_20260320_184906/synthesis/state.sh
echo "$STATUS"       # complete
echo "$OUTPUT_FULL"  # full report text
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_PROVIDER` | `lm_studio` | LLM provider: `lm_studio` or `openai` |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `LM_STUDIO_MODEL` | `qwen/qwen3.5-9b` | Model for text agents |
| `LM_STUDIO_VISION_MODEL` | `qwen/qwen3.5-9b` | Model for frame descriptions |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI text model |
| `OPENAI_VISION_MODEL` | `gpt-4o` | OpenAI vision model |
| `TAVILY_API_KEY` | — | Required for research agent web search |
| `WHISPER_MODEL` | `base` | Whisper model size (larger = more accurate, slower) |
| `MAX_ALIGNMENT_ITERATIONS` | `3` | Max research→alignment loop retries |
| `MAX_RESEARCH_TOOL_ROUNDS` | `6` | Max web-search calls per research run |
| `FRAME_DIFF_THRESHOLD` | `30` | Pixel diff threshold for keyframe detection (0–255) |
| `MOUSE_REGION_SIZE` | `100` | Bounding box (px) to ignore as cursor movement |
| `VIDEO_MAX_WIDTH` | `480` | Downscale video width before processing (0 = off) |
| `MAX_FRAMES_DESCRIBED` | `100` | Max keyframes sent to the vision LLM for description |
| `FRAME_CHUNK_SIZE` | `25` | Frames per chunk summarisation group |

---

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

54 unit tests covering all agents, state management, LLM client, workflow routing, and frame extraction.

> **Note:** Tests use mocked LLM/Tavily calls and do not require API keys or a running LM Studio instance.

---

## Project Structure

```
app.py                          ← Streamlit entry point
graph/
  state.py                      ← WorkflowState TypedDict + constants
  nodes.py                      ← LangGraph node functions
  workflow.py                   ← StateGraph definition + compilation
agents/
  transcriber.py                ← Whisper transcription (local or API)
  frame_extractor.py            ← OpenCV keyframe extraction
  requirements_agent.py         ← Workflow analyst (bottleneck + opportunity mapping)
  research_agent.py             ← Multi-paradigm researcher with Tavily web search
  alignment_agent.py            ← Strategic alignment judge
  synthesis_agent.py            ← Final report generator
utils/
  llm_client.py                 ← Unified OpenAI SDK client (LM Studio + OpenAI)
  state_manager.py              ← Read/write nested agent state files
  video_utils.py                ← Video metadata and validation helpers
tests/
  conftest.py                   ← Shared fixtures
  test_agents.py                ← Agent unit tests
  test_nodes.py                 ← Node unit tests
  test_workflow.py              ← Routing and loop tests
  test_state_manager.py         ← State file read/write tests
  test_llm_client.py            ← LLM client tests
  test_frame_extractor.py       ← Frame extraction tests
  test_state.py                 ← WorkflowState tests
  video/                        ← Add your own test video here (not committed)
state_outputs/                  ← Runtime run folders (gitignored)
.env.example                    ← Copy to .env and fill in keys
```

---

## Security Notes

- **Never commit `.env`** — it is gitignored. Copy `.env.example` to `.env` locally.
- **Video files are gitignored** — screen recordings may contain proprietary business information. They are never committed.
- **Run outputs are gitignored** — `state_outputs/run_*/` contains transcripts and AI analysis of your video. These stay local.
- **Prompt injection risk** — web search results from Tavily and video transcript content are passed into LLM prompts. Malicious content in a video transcript or a search result could attempt to hijack agent instructions. For internal use this is acceptable. For a public-facing deployment, add content filtering on Tavily results before they reach the LLM.
- **ffmpeg** is called with `subprocess.run()` using a list (not a shell string) so the video path cannot inject shell commands.
- **API keys** are read from environment variables at call time, not hardcoded.

---

## Limitations

- **Local models (9B):** Thinking models like Qwen3-8B are capable but can truncate outputs on very long context. FlowScope works around this by disabling thinking mode for synthesis and capping research rounds.
- **Frame limit:** By default the first 100 keyframes are described via vision (cost/latency control). Override with `MAX_FRAMES_DESCRIBED` in `.env`. Descriptions are then grouped into 25-frame narrative chunks (`FRAME_CHUNK_SIZE`) before being passed to the requirements agent — this keeps context window usage in check even at the 100-frame limit. The Streamlit UI previews only the first 20 thumbnails regardless of this setting.
- **Video length:** Very long videos (>30 min) may produce transcripts that exceed the model context window. Split long recordings if needed.

---

## Contributing

Issues and PRs welcome. Please open an issue before starting large changes.

---

## License

MIT
