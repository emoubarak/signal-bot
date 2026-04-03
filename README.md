# Signal LLM Bot

A modular Signal bot with LLM integration (OpenAI). It analyzes messages, summarizes discussions, answers questions, and can adopt custom personalities.

## Prerequisites

- Python 3.9+
- Docker (for `bbernhard/signal-cli-rest-api`)
- A **second Signal number** (for the bot)
- Environment variables: `SIGNAL_SERVICE`, `PHONE_NUMBER`, `OPENAI_API_KEY`

## Quick Start

### 1. Link the bot account to `signal-cli-rest-api`

```bash
docker run -d \
  --name signal-rest \
  -p 8080:8080 \
  -v signal-data:/home/.local/share/signal-cli \
  -e MODE=json-rpc \
  bbernhard/signal-cli-rest-api:latest
```

Open http://127.0.0.1:8080/v1/qrcodelink?device_name=local and scan the QR code with your Signal app (using the bot's account).

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Install dependencies and run

```bash
pip install -r requirements.txt
python bot.py
```

## Available Commands

| Command | Description |
|---------|-------------|
| `/analyse` | Logical and structured analysis |
| `/resume` | Summarize messages |
| `/roast` | Roast the message author |
| `/question` | Neutral, factual Q&A |
| `/debat` | Generate two opposing viewpoints |
| `/perso` | Custom personality (see below) |
| `/help` | Show all available commands |

### How Commands Work

- **Reply to a message** → targets THAT specific message
- `/command N` → targets the last N messages
- `/command` (no argument) → uses full history (up to limit)

### `/perso` - Custom Personality

Create any personality on the fly:

```
/perso a drunk pirate: what's the meaning of life?
/perso a cynical philosopher: explain quantum physics
/perso an enthusiastic sports commentator: analyze this argument
```

## Adding Custom Prompts

You can create permanent custom prompts that become new commands.

### 1. Create `prompts_custom.py`

Copy the example file:

```bash
cp prompts_custom.example.py prompts_custom.py
```

### 2. Add your prompts

Each prompt variable becomes a command (underscores → hyphens):

```python
# prompts_custom.py

MY_PROMPT = """
Your prompt description here.
Instructions for the LLM...
"""

ANOTHER_PROMPT = """
Another personality or utility...
"""
```

This creates:
- `/my-prompt`
- `/another-prompt`

### 3. Restart the bot

Custom prompts are automatically loaded at startup and appear in `/help`.

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_SERVICE` | required | Signal CLI REST API address (e.g., `127.0.0.1:8080`) |
| `PHONE_NUMBER` | required | Bot's Signal number (e.g., `+33700000000`) |
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `OPENAI_TEMPERATURE` | `1.2` | Response creativity |
| `OPENAI_MAX_OUTPUT_TOKENS` | `800` | Max response length |
| `MAX_CONTEXT_MESSAGES` | `50` | Max messages in history |
| `HARD_INPUT_CHAR_LIMIT` | `16000` | Max input size |

## Architecture

```
signal-bot/
├── bot.py              # Main entry point
├── command.py          # Command handler + custom prompt loader
├── prompts.py          # Core prompts
├── prompts_custom.py   # Your custom prompts (optional)
├── config.py           # Configuration
├── llm/
│   └── openai_client.py
├── utils/
│   └── history.py
└── patches/            # Signalbot library patches
```

## License

MIT