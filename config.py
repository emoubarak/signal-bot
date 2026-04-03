import os

# Centralized configuration
BOT_CONFIG = {
    "openai": {
        # Recommended model: fast, cheap, sufficient quality for analysis/summary
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "1.2")),
        "max_output_tokens": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "800")),
        # timeouts/retries are handled at client level
    },
    "limits": {
        # Simple safeguards
        "max_context_messages": int(os.environ.get("MAX_CONTEXT_MESSAGES", "50")),  # upper bound
        "hard_input_char_limit": int(os.environ.get("HARD_INPUT_CHAR_LIMIT", "16000")),
    },
    "runtime": {
        # To ignore empty messages, etc.
        "ignore_empty": True,
    },
}