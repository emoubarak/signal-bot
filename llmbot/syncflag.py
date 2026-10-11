"""Know when a DataMessage came from the bot's own account (a sync message).

For a private sync message signalbot replaces the source by the *destination*
(so replies go to the right chat) and keeps no flag, which makes the owner's
own messages look like the contact's. We remember the timestamps of sync
envelopes as they are parsed and expose `is_sync(timestamp)`.
"""
from __future__ import annotations

from collections import deque

from signalbot.messages import data_message as _dm

_SYNC_TS: deque[int] = deque(maxlen=2000)
_SYNC_SET: set[int] = set()
_original = _dm.parse_source_from_envelope
# The bot's own uuid, learnt from its sync envelopes (needed to recognise
# mentions when the number is hidden)
own_uuid: str | None = None


def _recording_parse_source(message_envelope):
    global own_uuid
    if getattr(message_envelope, "sync_message", None) is not None:
        ts = message_envelope.timestamp
        own_uuid = getattr(message_envelope, "source_uuid", None) or own_uuid
        if len(_SYNC_TS) == _SYNC_TS.maxlen:
            _SYNC_SET.discard(_SYNC_TS[0])
        _SYNC_TS.append(ts)
        _SYNC_SET.add(ts)
    return _original(message_envelope)


def install() -> None:
    if _dm.parse_source_from_envelope is not _recording_parse_source:
        _dm.parse_source_from_envelope = _recording_parse_source


def is_sync(timestamp: int) -> bool:
    return timestamp in _SYNC_SET


install()
