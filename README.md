# Signal LLM Bot

A bot that lives in your Signal groups. Call it by its name and say what you want: a summary of what was said, an explanation, a roast, a picture, a change to a photo, what a voice note says. No commands. The model reads the conversation, decides what to do, and uses its tools.

Runs on [OpenRouter](https://openrouter.ai) with `deepseek/deepseek-v4.1-flash` for chat and tool use, `meta/muse-image` for images, `google/gemini-3.1-flash-lite` for voice notes. All three can be changed.

Chat can also run on a **local model** instead: set `LOCAL_BASE_URL` to any OpenAI-compatible server (llama-swap, llama-server, vLLM, Ollama) and chat and tool use go there, with no API key and nothing leaving the machine. Images and voice notes have no local equivalent in a text server and stay on OpenRouter, or are switched off by emptying `IMAGE_MODEL` and `TRANSCRIBE_MODEL`.

## What it does

Say the bot's name (here `Jarvis`) anywhere in a message, mention it, reply to one of its messages, or write to it in private.

| You say | It does |
|---|---|
| "Jarvis, summarize what was said since last night" | Reads the history it needs and summarises, with names and times |
| "jarvis explain what he means" (replying to a message) | Answers about that message |
| "Jarvis roast him" (replying to someone) | Roasts the author of that message |
| "jarvis what did he say?" (replying to a voice note) | Quotes the transcription |
| "Jarvis make an image of a pirate cat in 16:9" | Generates the image |
| "jarvis put a hat on this photo" (replying to a photo, or attaching one) | Edits the photo |
| "Jarvis combine our two photos" | Picks the images it saw in the conversation and combines them |
| "Jarvis answer as a philosopher" | Adopts one of your personas (see below) |
| "Jarvis how much did you cost this month?" | Reports its spend |
| "Jarvis who won last night?", "Jarvis what is the weather in Lyon tomorrow?" | Searches the web or asks the weather service, then answers from what it found |
| "Jarvis who talked about a restaurant in Lyon?" | Searches the whole history by keywords, not only the latest messages |
| "Jarvis remind us to call the plumber tomorrow at 9" | Posts the reminder in the chat at that time, even after a restart. "What reminders are waiting?" lists them, and one can be cancelled |
| "Jarvis make a meme about Karim being late" | Picks a classic template (Drake, Distracted Boyfriend…), writes the captions and sends the meme |
| "Jarvis who pays, between Karim, Lina and me?" | Draws at random (coin, dice, one of several options) instead of pretending to |
| "Jarvis read this" + a link | Reads the page and summarises it |
| "Jarvis we are 7, the bill is 214.60, how much each?" | Computes it exactly instead of guessing |
| `/clear` (the owner, in any chat) | Forgets what was said before, so an old conversation stops colouring its answers |

Voice notes are transcribed when they arrive, so "summarize" covers them like any other message. If the name only comes up in passing ("Jarvis crashed yesterday"), the bot stays quiet.

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

The model must support tool calls, which is what the bot drives its history, image and spend tools with. With `IMAGE_MODEL` or `TRANSCRIBE_MODEL` empty the bot stops offering that capability and says so when asked, rather than promising something it cannot do. Keep `OPENROUTER_API_KEY` and the two models set to keep images and voice notes while chat stays local. Reasoning is switched off for a local model (`enable_thinking: false`), which cuts an answer from 7-14 s to 2-4 s on Qwen3.8; a model whose template ignores that option is unaffected.

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
| `OWNER_NUMBER` | none | Your number: the only one who can `/clear`, and who gets a private message on every restart saying which model it came back on |
| `ANNOUNCE_STARTUP` | `true` | Announce every restart in private to `OWNER_NUMBER` |
| `RATE_LIMIT_PER_HOUR` | `20` | Requests per person per hour (0 = unlimited) |
| `IMAGES_PER_DAY` | `10` | Images per person per day |
| `CONTEXT_MESSAGES` | `30` | Recent messages given to the model with every request |
| `MAX_OUTPUT_TOKENS` | `1000` | Reply budget, doubled once if the reply is cut |
| `TIMEZONE` | `Europe/Paris` | For the times shown to the model |
| `DATA_DIR` | `data` | SQLite database and attachments |
| `DESCRIBE_IMAGES` | `false` | Describe every incoming image with the chat model (one vision call each) |
| `ADMIN_TOKEN` | empty | Password of the admin page; the page is off without it |
| `ADMIN_PORT` / `ADMIN_HOST` | `8090` / `127.0.0.1` | Where the admin page listens (in Docker, compose sets the host; `ADMIN_BIND` picks the address on the machine) |
| `DISABLED_TOOLS` | empty | Tools switched off, comma separated (the admin page edits this) |
| `WEB_TOOLS` | `true` | Offer the free tools: web search, reading a link, weather, time in another timezone, calculator |
| `KLIPY_API_KEY` | empty | Free key from [partner.klipy.com](https://partner.klipy.com): turns on `send_gif`, reaction GIFs (empty = off) |
| `ATTACHMENT_RETENTION_DAYS` | `30` | Attachments older than this are deleted at startup |

Indicative costs (OpenRouter, October 2026): a request with DeepSeek V4.1 Flash ≈ $0.0001 (the cheapest provider is picked), a one-minute voice note ≈ $0.001, an image or an edit with Muse Image $0.010. A local chat model costs nothing and its requests are reported at $0.

### Admin page

A web page to change the bot's settings without editing `.env`: the provider (a local server or OpenRouter) and its models and keys, which tools the bot may use (each one can be switched off, and tried from the page), the Signal account, the bot's names, the groups it answers in, its limits. It also shows where the bot stands: model, Signal connection, messages stored, cost over 24 h and 30 days.

It is **off unless `ADMIN_TOKEN` is set** in `.env`; that value is the password. Then open `http://127.0.0.1:8090` on the host. To reach it from your other devices, set `ADMIN_BIND` in `.env` to the host's Tailscale address (the compose file publishes the page on that address only). The page speaks plain HTTP, so keep it on localhost or a private network, never on the open internet.

- **What applies at once, and what needs a restart.** Every field says which. Provider, models, keys, tools, names, limits and the timezone apply on save. The phone number, the allowed groups and people, the startup message and the attachment retention apply at the next start; the page then says a restart is pending. Its "Restart" button stops the bot the way a deploy does (answers in progress are finished first), and Docker brings it back.
- **Where it is saved.** In the bot's own database (`data/bot.sqlite3`), over the `.env`: a value saved from the page wins at every start until you reset it from the page ("Revenir au .env"). The page shows which values come from where. `ADMIN_TOKEN`, `ADMIN_HOST`, `ADMIN_PORT` and the Signal service address stay in `.env` and cannot be changed from the page.
- **Secrets.** Keys are stored in that database in clear, like in `.env`, and are never sent back to the browser: the page only says whether one is set and its last four characters. An empty key box keeps the saved key; deleting a key is its own button.
- **Safety.** Sign-in is a session cookie (`HttpOnly`, `SameSite=Strict`); five wrong passwords lock the address for a minute; changes need a header a foreign website cannot send; the page has a strict content policy.
- **Signal account.** The page checks that signal-cli answers and that the number is registered, and can show the QR code to link a device. The page starts even when signal-cli is down, so it can be used to fix the account.

### Server and auto-deploy

On a PC with Podman instead of Docker (systemd user units, like the rest of the machine's services), follow `deploy/local/README.md`: Quadlet units, a timer that applies every push to `main` with an automatic rollback, and the admin page on port 8450.

`compose.yaml` runs signal-cli-rest-api and the bot (`docker compose up -d`; the bot's data stays in `data/`). Set `SIGNAL_BIND` in `.env` to the address the link page should listen on (default `127.0.0.1`), then link the account at `http://<SIGNAL_BIND>:8086/v1/qrcodelink?device_name=bot`.

To deploy every push to `main`, add `deploy/autodeploy.sh` to the server's crontab:

```
*/5 * * * * $HOME/repos/signal-bot/deploy/autodeploy.sh >> $HOME/repos/signal-bot/deploy/autodeploy.log 2>&1
```

It fast-forwards to `origin/main` and rebuilds the bot; if the new version does not stay up for 30 seconds it goes back to the previous commit and skips the broken one until `main` moves again.

## How it works

- Every message of every allowed chat goes into SQLite (`data/bot.sqlite3`) with its Signal identity (author, timestamp, replied-to message). Names come from Signal profiles, never numbers. History survives restarts.
- Voice notes are converted with ffmpeg and transcribed on arrival; the text is stored with a 🎤 prefix. Images are kept in `data/attachments/` so they can be edited later; view-once media is not kept.
- When the bot is addressed, the model receives: the last `CONTEXT_MESSAGES` messages with times and names, the reply chain with the bot as user/assistant turns, the message replied to as `[TARGET]`, the attached images, and its tools: `get_history` (more or older messages), `generate_image` (prompt, format, and `img1`, `img2`… identifiers of images seen in the conversation as references), `get_spend`, and, unless `WEB_TOOLS=false`, six free ones that need no account: `web_search` (DuckDuckGo through the `ddgs` package, unofficial so it can be rate limited), `read_page` (public http(s) pages only: localhost, private and link-local addresses are refused, redirects are re-checked), `get_weather` ([Open-Meteo](https://open-meteo.com), free for non-commercial use), `get_time`, `calculate` (arithmetic only, no code is evaluated, results are size-bounded) and `pick_random`, plus `make_meme` (a captioned classic template from [memegen.link](https://memegen.link), no key, one per reply, off with `WEB_TOOLS=false`; the captions are checked against the template before generating: as many lines as it takes, 60 characters each, so nothing is silently dropped). Independently of `WEB_TOOLS`, the bot has `search_messages` (every word must match, accents and case ignored, never past a `/clear`), `remind` and `list_reminders` (reminders are stored in the database, checked every 15 s, at most 20 waiting per chat and a year ahead; one that came due while the bot was off is posted at the next start with an apology; delivery is tried five times). With `KLIPY_API_KEY` it also gets `send_gif`, one animated GIF per reply from [Klipy](https://klipy.com) (the Tenor API closed on 30 June 2026); the GIF is sent as is, so it stays animated, and it does not use the daily image allowance. Models are set in `.env` only. It can call tools up to five times before answering. `generate_image` is only offered when an image model is configured, so a bot without one says so instead of promising a picture.
- Replies cut by the token limit are retried once with a bigger budget. Model errors are reported in the chat. Rate limits are per person; the image limit is also given to the model so it refuses politely.
- Messages typed from the bot's own phone are recognised and never answered, except when they call the bot by name in a group.
- On every start the bot says "I restarted, beep-boop, I am model X locally" in private to `OWNER_NUMBER` only, never in groups. On OpenRouter `X` is the model that really answers a one-token probe; if it does not answer in time the configured name is used, and the announcement still goes out. A local model is announced by its configured name without any request, so llama-swap does not load it at boot: it loads on the first real message. Set `ANNOUNCE_STARTUP=false` to silence it.
- `OWNER_NUMBER` can type `/clear` in a chat to reset its context: nothing said before is shown to the model any more, and `get_history` cannot reach past it either, so the bot stops being biased by an old conversation. It is per chat (a `/clear` in one group leaves the others alone), it survives restarts, and it deletes nothing: the messages stay in SQLite, so cost and history are intact and the floor can be moved again with another `/clear`. The bot confirms in the chat so it is visible that the command took.

## Layout

```
bot.py                  entry point
llmbot/
  config.py             settings from the environment
  store.py              SQLite: messages, usage, settings
  openrouter.py         chat with tools (OpenRouter or a local server), audio input, image generation (httpx)
  media.py              attachments on disk, ffmpeg conversions
  agent.py              tools and the model loop
  webtools.py           the free tools: search, read a page, weather, time, calculator, dice, GIF
  admin.py              the admin page's server (stdlib, own thread) and its API
  reminders.py          setting reminders and posting them when due
  runtime_config.py     which settings the page can change, validation, how they are saved
  static/admin.html     the admin page itself (one file, no build step)
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
