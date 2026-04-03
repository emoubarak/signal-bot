import os
from typing import Optional
from openai import AsyncOpenAI

# Default async client
# (Modern SDK exposes OpenAI / AsyncOpenAI; see official README)
_client_async: Optional[AsyncOpenAI] = None


def _get_async_client() -> AsyncOpenAI:
    global _client_async
    if _client_async is None:
        # SDK reads OPENAI_API_KEY automatically if not provided
        _client_async = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _client_async


async def run_chat(
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 800,
) -> str:
    """
    Simple async call to Chat Completions.
    Returns the text of the first response.
    """
    client = _get_async_client()
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()