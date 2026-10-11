# Signal LLM Bot

A bot that lives in your Signal groups. Call it by its name and say what you want: a summary of what was said, an explanation, a roast, a picture, a change to a photo, what a voice note says. No commands. The model reads the conversation, decides what to do, and uses its tools.

Runs on [OpenRouter](https://openrouter.ai) with `anthropic/claude-haiku-5.5` for chat and tool use, `bytedance-seed/seedream-5-0-flash` for images, `google/gemini-3.1-flash-lite` for voice notes. All three can be changed.

## What it does

Say the bot's name (here `Jarvis`) anywhere in a message, mention it, reply to one of its messages, or write to it in private.

| You say | It does |
|---|---|
| « Jarvis, résume ce qui s'est dit depuis hier soir » | Reads the history it needs and summarises, with names and times |
| « jarvis explique-moi ce qu'il veut dire » (replying to a message) | Answers about that message |
| « Jarvis roast-le » (replying to someone) | Roasts the author of that message |
| « jarvis il a dit quoi ? » (replying to a voice note) | Quotes the transcription |
| « Jarvis fais une image d'un chat pirate en 16:9 » | Generates the image |
| « jarvis mets un chapeau sur cette photo » (replying to a photo, or attaching one) | Edits the photo |
| « Jarvis mélange nos deux photos » | Picks the images it saw in the conversation and combines them |
| « Jarvis réponds en philosophe » | Adopts one of your personas (see below) |
| « Jarvis t'as coûté combien ce mois-ci ? » | Reports its spend |
| « Jarvis passe sur openai/gpt-5-mini » (admins) | Switches the chat model |

Voice notes are transcribed when they arrive, so « résume » covers them like any other message. If the name only comes up in passing (« Jarvis a planté hier »), the bot stays quiet.

Personas: copy `prompts_custom.example.py` to `prompts_custom.py`. Every uppercase string is a style the bot can be asked to adopt by name.

## Setup

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg (voice notes and image conversion), Docker, a phone number for the bot and an OpenRouter key.

1. Run [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) (0.101 or newer) in json-rpc mode and link the bot's number:

   ```bash
   docker run -d --name signal-rest -p 8080:8080 \
     -v signal-data:/home/.local/share/signal-cli \
     -e MODE=json-rpc bbernhard/signal-cli-rest-api:latest
   ```

   Open http://127.0.0.1:8080/v1/qrcodelink?device_name=bot and scan the QR code from the Signal app signed in with the bot's number.

2. Configure and run:

   ```bash
   cp .env.example .env   # SIGNAL_SERVICE, PHONE_NUMBER, BOT_NAME, OPENROUTER_API_KEY
   uv sync
   uv run python bot.py
   ```

3. Add the bot to a group and call it by name.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `SIGNAL_SERVICE` | required | signal-cli-rest-api address, e.g. `127.0.0.1:8080` |
| `PHONE_NUMBER` | required | The bot's number |
| `OPENROUTER_API_KEY` | required | OpenRouter key |
| `BOT_NAME` | `Bot` | The bot's name; saying it in a message addresses the bot |
| `BOT_ALIASES` | none | Other names that address the bot, comma-separated |
| `OPENROUTER_MODEL` | `anthropic/claude-haiku-5.5` | Chat model, any OpenRouter slug with text output and tool use |
| `IMAGE_MODEL` | `bytedance-seed/seedream-5-0-flash` | Image model, from the OpenRouter Image API |
| `TRANSCRIBE_MODEL` | `google/gemini-3.1-flash-lite` | Audio-input model for voice notes |
| `ALLOWED_GROUPS` | all | Group names or ids the bot listens to, comma-separated |
| `ALLOWED_NUMBERS` | all | Numbers allowed to write to it in private |
| `ADMIN_NUMBERS` | anyone | Who may ask it to switch model |
| `RATE_LIMIT_PER_HOUR` | `20` | Requests per person per hour (0 = unlimited) |
| `IMAGES_PER_DAY` | `10` | Images per person per day |
| `CONTEXT_MESSAGES` | `30` | Recent messages given to the model with every request |
| `MAX_OUTPUT_TOKENS` | `1000` | Reply budget, doubled once if the reply is cut |
| `TIMEZONE` | `Europe/Paris` | For the times shown to the model |
| `DATA_DIR` | `data` | SQLite database and attachments |
| `DESCRIBE_IMAGES` | `false` | Describe every incoming image with the chat model (one vision call each) |
| `ATTACHMENT_RETENTION_DAYS` | `30` | Attachments older than this are deleted at startup |

Indicative costs (OpenRouter, October 2026): a request with Claude Haiku 5.5 ≈ $0.0003, a one-minute voice note ≈ $0.001, an image or an edit with Seedream 5.0 Flash $0.018.

## How it works

- Every message of every allowed chat goes into SQLite (`data/bot.sqlite3`) with its Signal identity (author, timestamp, replied-to message). Names come from Signal profiles, never numbers. History survives restarts.
- Voice notes are converted with ffmpeg and transcribed on arrival; the text is stored with a 🎤 prefix. Images are kept in `data/attachments/` so they can be edited later; view-once media is not kept.
- When the bot is addressed, the model receives: the last `CONTEXT_MESSAGES` messages with times and names, the reply chain with the bot as user/assistant turns, the message replied to as `[CIBLE]`, the attached images, and four tools: `get_history` (more or older messages), `generate_image` (prompt, format, and `img1`, `img2`… identifiers of images seen in the conversation as references), `get_spend`, `set_chat_model`. It can call tools up to five times before answering.
- Replies cut by the token limit are retried once with a bigger budget. Model errors are reported in the chat. Rate limits are per person; the image limit is also given to the model so it refuses politely.
- Messages typed from the bot's own phone are recognised and never answered, except when they call the bot by name in a group.

## Layout

```
bot.py                  entry point
llmbot/
  config.py             settings from the environment
  store.py              SQLite: messages, usage, settings
  openrouter.py         chat with tools, audio input, image generation (httpx)
  media.py              attachments on disk, ffmpeg conversions
  agent.py              tools and the model loop
  handler.py            the Signal handler: record, addressing, reply
  context.py            how messages are shown to the model
  prompts.py            shared prompt pieces
  personas.py           custom personas from prompts_custom.py
  syncflag.py           tells the bot's own (sync) messages apart
tests/                  pytest, with signal-cli-rest-api and OpenRouter stubbed
```

Tests: `uv run pytest`.

## License

MIT
