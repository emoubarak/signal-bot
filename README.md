# Signal LLM Bot

A bot that lives in your Signal groups. Call it by its name and say what you want: a summary of what was said, an explanation, a roast, a picture, a change to a photo, what a voice note says. No commands. The model reads the conversation, decides what to do, and uses its tools.

Runs on [OpenRouter](https://openrouter.ai) with `deepseek/deepseek-v4.1-flash` for chat and tool use, `meta/muse-image` for images, `google/gemini-3.1-flash-lite` for voice notes. All three can be changed.

Chat can also run on a **local model** instead: set `LOCAL_BASE_URL` to any OpenAI-compatible server (llama-swap, llama-server, vLLM, Ollama) and chat and tool use go there, with no API key and nothing leaving the machine. Images and voice notes have no local equivalent in a text server and stay on OpenRouter, or are switched off by emptying `IMAGE_MODEL` and `TRANSCRIBE_MODEL`.

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

### Running on a local model

Point `LOCAL_BASE_URL` at any OpenAI-compatible server to keep chat and tool use on your own machine:

```bash
LOCAL_BASE_URL=http://100.111.115.112:8089/v1   # llama-swap, llama-server, vLLM, Ollama…
CHAT_MODEL=qwen3.8-flash-next-ablit              # whatever the server serves; a llama-swap selector loads it on demand
IMAGE_MODEL=                                    # empty: a text server has no /images endpoint
TRANSCRIBE_MODEL=                               # empty: it ignores audio input too
OPENROUTER_API_KEY=
```

The model must support tool calls, which is what the bot drives its history, image and spend tools with. With `IMAGE_MODEL` or `TRANSCRIBE_MODEL` empty the bot stops offering that capability and says so when asked, rather than promising something it cannot do. Keep `OPENROUTER_API_KEY` and the two models set to keep images and voice notes while chat stays local.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `SIGNAL_SERVICE` | required | signal-cli-rest-api address, e.g. `127.0.0.1:8080` |
| `PHONE_NUMBER` | required | The bot's number |
| `OPENROUTER_API_KEY` | required unless `LOCAL_BASE_URL` | OpenRouter key: chat without a local model, images, voice notes |
| `LOCAL_BASE_URL` | none | OpenAI-compatible chat endpoint, e.g. `http://127.0.0.1:8089/v1`. Set = chat runs locally, no key needed |
| `BOT_NAME` | `Bot` | The bot's name; saying it in a message addresses the bot |
| `BOT_ALIASES` | none | Other names that address the bot, comma-separated |
| `CHAT_MODEL` | `deepseek/deepseek-v4.1-flash` | Chat model: an OpenRouter slug, or a name the local server serves (a llama-swap selector works). `OPENROUTER_MODEL` is still read as the older name |
| `IMAGE_MODEL` | `meta/muse-image` | Image model, from the OpenRouter Image API. Empty = image generation off |
| `TRANSCRIBE_MODEL` | `google/gemini-3.1-flash-lite` | Audio-input model for voice notes. Empty = transcription off |
| `ALLOWED_GROUPS` | all | Group names or ids the bot listens to, comma-separated |
| `ALLOWED_NUMBERS` | all | Numbers allowed to write to it in private |
| `OWNER_NUMBER` | none | Your number: gets a private message on every restart saying which model it came back on |
| `ANNOUNCE_STARTUP` | `true` | Announce every restart in the allowed groups and to `OWNER_NUMBER` |
| `RATE_LIMIT_PER_HOUR` | `20` | Requests per person per hour (0 = unlimited) |
| `IMAGES_PER_DAY` | `10` | Images per person per day |
| `CONTEXT_MESSAGES` | `30` | Recent messages given to the model with every request |
| `MAX_OUTPUT_TOKENS` | `1000` | Reply budget, doubled once if the reply is cut |
| `TIMEZONE` | `Europe/Paris` | For the times shown to the model |
| `DATA_DIR` | `data` | SQLite database and attachments |
| `DESCRIBE_IMAGES` | `false` | Describe every incoming image with the chat model (one vision call each) |
| `ATTACHMENT_RETENTION_DAYS` | `30` | Attachments older than this are deleted at startup |

Indicative costs (OpenRouter, October 2026): a request with DeepSeek V4.1 Flash ≈ $0.0001 (the cheapest provider is picked), a one-minute voice note ≈ $0.001, an image or an edit with Muse Image $0.010. A local chat model costs nothing and its requests are reported at $0.

### Server and auto-deploy

`compose.yaml` runs signal-cli-rest-api and the bot (`docker compose up -d`; the bot's data stays in `data/`). Set `SIGNAL_BIND` in `.env` to the address the link page should listen on (default `127.0.0.1`), then link the account at `http://<SIGNAL_BIND>:8086/v1/qrcodelink?device_name=bot`.

To deploy every push to `main`, add `deploy/autodeploy.sh` to the server's crontab:

```
*/5 * * * * $HOME/repos/signal-bot/deploy/autodeploy.sh >> $HOME/repos/signal-bot/deploy/autodeploy.log 2>&1
```

It fast-forwards to `origin/main` and rebuilds the bot; if the new version does not stay up for 30 seconds it goes back to the previous commit and skips the broken one until `main` moves again.

## How it works

- Every message of every allowed chat goes into SQLite (`data/bot.sqlite3`) with its Signal identity (author, timestamp, replied-to message). Names come from Signal profiles, never numbers. History survives restarts.
- Voice notes are converted with ffmpeg and transcribed on arrival; the text is stored with a 🎤 prefix. Images are kept in `data/attachments/` so they can be edited later; view-once media is not kept.
- When the bot is addressed, the model receives: the last `CONTEXT_MESSAGES` messages with times and names, the reply chain with the bot as user/assistant turns, the message replied to as `[CIBLE]`, the attached images, and three tools: `get_history` (more or older messages), `generate_image` (prompt, format, and `img1`, `img2`… identifiers of images seen in the conversation as references), `get_spend`. Models are set in `.env` only. It can call tools up to five times before answering. `generate_image` is only offered when an image model is configured, so a bot without one says so instead of promising a picture.
- Replies cut by the token limit are retried once with a bigger budget. Model errors are reported in the chat. Rate limits are per person; the image limit is also given to the model so it refuses politely.
- Messages typed from the bot's own phone are recognised and never answered, except when they call the bot by name in a group.
- On every start the bot says « J'ai redémarré, blipbloup, je suis le modèle X en local » in each allowed group and in private to `OWNER_NUMBER`. `X` is the model that really answers the probe, not the configured name: on a llama-swap selector that means the model it loaded (`qwen3.8-flash-next-q2_0`), which is what you want to read after a restart. If the model does not answer in time the configured name is used, and the announcement still goes out. Set `ANNOUNCE_STARTUP=false` to silence it.

## Layout

```
bot.py                  entry point
llmbot/
  config.py             settings from the environment
  store.py              SQLite: messages, usage, settings
  openrouter.py         chat with tools (OpenRouter or a local server), audio input, image generation (httpx)
  media.py              attachments on disk, ffmpeg conversions
  agent.py              tools and the model loop
  handler.py            the Signal handler: record, addressing, reply
  context.py            how messages are shown to the model
  prompts.py            shared prompt pieces
  personas.py           custom personas from prompts_custom.py
  syncflag.py           tells the bot's own (sync) messages apart
  startup.py            the restart announcement: which model it came back on
tests/                  pytest, with signal-cli-rest-api and OpenRouter stubbed
```

Tests: `uv run pytest`.

## License

MIT
