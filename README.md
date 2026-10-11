# Signal LLM Bot

A Signal bot for group chats, powered by [OpenRouter](https://openrouter.ai). Call it by its name and say what you want in plain language: a summary of the conversation, an explanation, a roast, an image ("make a picture of…", "put a hat on this photo" while replying to the photo), what a voice note says. The model decides what to do and calls the right tool. No commands to learn.

Default models: `anthropic/claude-haiku-5.5` for chat and tool use, `bytedance-seed/seedream-5-0-flash` for images, `google/gemini-3.1-flash-lite` for voice notes. All three are configurable, and the chat model can be switched by asking the bot.

## How it works

- Every message of every allowed chat is stored in SQLite (`data/bot.sqlite3`) with its Signal identity (author, timestamp), so the bot knows who said what, when, and in reply to what, and still does after a restart.
- Voice notes are transcribed on arrival (converted with ffmpeg, sent to an audio-capable model) and stored as text, prefixed 🎤. Images are saved to `data/attachments/` so they can be edited later.
- Natural-language mode (`llmbot/agent.py`): when the bot is called by name (`BOT_NAME`, `BOT_ALIASES`, accent-insensitive), mentioned, replied to, or written to in private, the model gets the last 30 messages, the reply chain as user/assistant turns, the quoted message as `[CIBLE]`, attached images, and four tools: `get_history` (more or older messages), `generate_image` (with `img1`, `img2`… identifiers of images seen in the conversation as references, so "combine our two photos" works), `get_spend`, `set_chat_model`. Up to 5 tool rounds. If the name is only mentioned in passing, the model answers `[rien]` and the bot stays silent.
- Replies cut by the token limit are retried once with a bigger budget. Model errors are reported in the chat. Per-person rate limits (`RATE_LIMIT_PER_HOUR`, `IMAGES_PER_DAY`) and allow-lists keep the bill in check; ask the bot what it has cost.

## Examples

Say the bot's name (here `Jarvis`) anywhere in the message:

- « Jarvis, résume ce qui s'est dit depuis hier soir »
- « jarvis il a dit quoi dans son vocal ? » (replying to the voice note)
- « Jarvis roast-le » (replying to someone's message)
- « Jarvis fais une image d'un chat pirate en 16:9 »
- « jarvis mets un chapeau sur cette photo » (replying to the photo, or attaching it)
- « Jarvis mélange nos deux photos » (the model picks the images it saw in the conversation)
- « Jarvis t'as coûté combien ce mois-ci ? », « Jarvis passe sur openai/gpt-5-mini » (admins)
- Mentioning the bot, replying to one of its messages, or writing to it in private works the same, no name needed.

Personas: copy `prompts_custom.example.py` to `prompts_custom.py`; every uppercase string is a style the bot can be asked to adopt by name (« Jarvis, réponds en philosophe »).

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg (for voice notes), Docker, a second phone number for the bot, an OpenRouter key.

1. Run [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) (0.101 or newer) in json-rpc mode and link the bot's number:

   ```bash
   docker run -d --name signal-rest -p 8080:8080 \
     -v signal-data:/home/.local/share/signal-cli \
     -e MODE=json-rpc bbernhard/signal-cli-rest-api:latest
   ```

   Open http://127.0.0.1:8080/v1/qrcodelink?device_name=bot and scan the QR code from the Signal app logged in with the bot's number.

2. Configure and run:

   ```bash
   cp .env.example .env   # fill SIGNAL_SERVICE, PHONE_NUMBER, OPENROUTER_API_KEY
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
| `OPENROUTER_MODEL` | `anthropic/claude-haiku-5.5` | Chat model (any text-output slug) |
| `IMAGE_MODEL` | `bytedance-seed/seedream-5-0-flash` | Image model (OpenRouter Image API) |
| `TRANSCRIBE_MODEL` | `google/gemini-3.1-flash-lite` | Audio-input model for voice notes |
| `BOT_NAME` | `Bot` | The bot's name: saying it in a message addresses the bot |
| `BOT_ALIASES` | none | Other names that address the bot, comma-separated |
| `ALLOWED_GROUPS` | all | Comma-separated group names or ids |
| `ALLOWED_NUMBERS` | all | Comma-separated numbers allowed in private chat |
| `ADMIN_NUMBERS` | anyone | Who may ask the bot to switch model |
| `RATE_LIMIT_PER_HOUR` | `20` | Requests per person per hour (0 = unlimited) |
| `IMAGES_PER_DAY` | `10` | Images per person per day |
| `MAX_CONTEXT_MESSAGES` | `50` | Upper bound on messages given to the model |
| `MAX_INPUT_CHARS` | `24000` | Oldest context lines are dropped beyond this |
| `MAX_OUTPUT_TOKENS` | `1000` | Reply budget (doubled once if the reply is cut) |
| `TIMEZONE` | `Europe/Paris` | For the timestamps shown to the model |
| `DATA_DIR` | `data` | SQLite database and attachments |
| `DESCRIBE_IMAGES` | `false` | Describe every incoming image with the chat model (one vision call each) |
| `ATTACHMENT_RETENTION_DAYS` | `30` | Attachments older than this are deleted at startup |

Indicative costs (OpenRouter, October 2026): a summary with Claude Haiku 5.5 ≈ $0.0001, a one-minute voice note ≈ $0.001, an image or an edit with Seedream 5.0 Flash = $0.018.

## Layout

```
bot.py                  entry point
llmbot/
  config.py             Settings from environment
  store.py              SQLite: messages, usage, settings
  openrouter.py         chat, audio input, image generation (httpx)
  media.py              attachments on disk, ffmpeg conversion
  prompts.py            shared prompt pieces
  personas.py           custom personas from prompts_custom.py
  context.py            formatting of the conversation for the model
  handler.py            the Signal handler: record, addressing, reply
  agent.py              natural-language mode: tools and the model loop
  syncflag.py           tells the bot's own (sync) messages apart
tests/                  pytest, signal-cli-rest-api and OpenRouter stubbed
```

Tests: `uv run pytest`.

## License

MIT
