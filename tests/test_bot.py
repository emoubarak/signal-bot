from __future__ import annotations

import asyncio

from signalbot.test_utils.chat_testing import _new_timestamp

from llmbot.store import KIND_BOT, KIND_IMAGE, KIND_VOICE
from tests.conftest import (
    CHAT_ID, LEA, PAUL, Harness, bot_quote, edit_message, group_message, image_attachment, quote_of,
    sync_message, voice_attachment,
)


def tool_outputs(call: dict) -> list[str]:
    return [m["content"] for m in call["messages"] if m["role"] == "tool"]


async def test_names_recorded_and_reply_targets_quoted_message(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(group_message(PAUL, "La pizza à l'ananas c'est le meilleur truc du monde", ts=t1),
                      group_message(LEA, "n'importe quoi"))
    await harness.run(group_message(LEA, "jarvis roast ça", quote=quote_of(PAUL, t1, "La pizza à l'ananas c'est le meilleur truc du monde")))
    user = harness.llm.last_user_text()
    assert "[CIBLE]" in user and "Paul : La pizza à l'ananas" in user
    assert "Léa : n'importe quoi" in harness.llm.last_system()  # recent window as context
    assert "+336" not in user and "+336" not in harness.llm.last_system()  # names, never numbers
    sent = harness.sent()
    assert len(sent) == 1 and sent[0].message == "réponse du modèle"
    assert sent[0].quote_author == LEA[1]  # the reply quotes the request


async def test_bot_answers_are_part_of_the_context(harness: Harness):
    await harness.run(group_message(PAUL, "on mange où ce soir ?"), group_message(LEA, "jarvis une idée ?"))
    await harness.run(group_message(PAUL, "jarvis et toi tu viens ?"))
    system = harness.llm.last_system()
    assert "Léa : jarvis une idée ?" in system and "Jarvis (répond à Léa) : réponse du modèle" in system


async def test_calling_the_bot_by_name_runs_the_agent_with_history_tool(harness: Harness):
    harness.llm.script = [{"name": "get_history", "arguments": {"count": 50}}, "Voilà le résumé."]
    await harness.run(
        group_message(PAUL, "on part à 9h"),
        group_message(LEA, "ok pour moi"),
        group_message(PAUL, "jarvis, tu peux résumer la conv depuis ce matin ?"),
    )
    assert harness.sent_texts() == ["Voilà le résumé."]
    first, second = harness.llm.chat_calls[-2], harness.llm.chat_calls[-1]
    assert first["tools"] and any(t["function"]["name"] == "generate_image" for t in first["tools"])
    assert "Paul : on part à 9h" in first["messages"][0]["content"]
    assert "Léa : ok pour moi" in tool_outputs(second)[0]


async def test_name_with_accent_and_silence(harness: Harness):
    harness.llm.script = ["[rien]"]
    await harness.run(group_message(PAUL, "JÁRVIS a encore planté hier lol"))
    assert len(harness.llm.chat_calls) == 1 and harness.sent() == []
    await harness.run(group_message(PAUL, "jarvisse n'est pas un mot"), group_message(LEA, "mon jarvis@home"),
                      group_message(LEA, "/resume"))
    assert len(harness.llm.chat_calls) == 1  # no word match, no slash commands any more


async def test_voice_note_is_transcribed_and_understood(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(group_message(PAUL, None, attachments=[voice_attachment()], ts=t1))
    await harness.run(group_message(LEA, "jarvis il dit quoi dans son vocal ?", quote=quote_of(PAUL, t1, "")))
    assert harness.llm.transcribe_calls and harness.llm.transcribe_calls[0]["format"] == "aac"
    stored = harness.store.get(CHAT_ID, t1)
    assert stored.kind == KIND_VOICE and stored.text == "salut c'est un vocal"
    assert "🎤 salut c'est un vocal" in harness.llm.last_user_text()  # the quoted voice note is the target


async def test_non_voice_audio_is_not_transcribed(harness: Harness):
    t1 = _new_timestamp()
    att = {"id": "song.mp3", "contentType": "audio/mpeg", "isVoiceNote": False, "size": 10}
    await harness.run(group_message(PAUL, None, attachments=[att], ts=t1))
    assert harness.llm.transcribe_calls == []
    assert harness.store.get(CHAT_ID, t1).text == "[fichier audio/mpeg]"


async def test_image_generation_then_edit_by_replying_to_the_result(harness: Harness):
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "un chat sur la lune", "aspect_ratio": "16:9"}}, "Voilà"]
    await harness.run(group_message(PAUL, "Jarvis fais une image d'un chat sur la lune en 16:9"))
    call = harness.llm.image_calls[0]
    assert call["aspect_ratio"] == "16:9" and call["prompt"] == "un chat sur la lune" and call["references"] == []
    sent = harness.sent()[0]
    assert sent.message == "Voilà" and sent.base64_attachments[0].startswith("data:image/png;base64,")
    bot_ts = harness.bot_timestamps(CHAT_ID)[-1]
    assert harness.store.get(CHAT_ID, bot_ts).kind == KIND_IMAGE

    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "le même chat en bleu", "reference_images": ["img1"]}}, ""]
    await harness.run(group_message(LEA, "jarvis rends-le bleu", quote=bot_quote(bot_ts, "Voilà")))
    edit = harness.llm.image_calls[1]
    assert edit["aspect_ratio"] == "auto" and len(edit["references"]) == 1 and edit["references"][0][1] == "image/png"
    assert harness.sent()[-1].message == ""  # image with an empty caption is still sent


async def test_image_sent_with_the_request_is_a_reference(harness: Harness):
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "avec un chapeau", "reference_images": ["img1"]}}, "ok"]
    await harness.run(group_message(PAUL, "jarvis mets un chapeau sur cette photo", attachments=[image_attachment()]))
    assert len(harness.llm.image_calls[0]["references"]) == 1


async def test_agent_generates_an_image_from_several_references(harness: Harness):
    await harness.run(group_message(PAUL, "ma photo", attachments=[image_attachment()]),
                      group_message(LEA, "la mienne", attachments=[image_attachment()]))
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "les deux ensemble", "reference_images": ["img1", "img2"]}}, "Et voilà"]
    await harness.run(group_message(PAUL, "Jarvis mets nos deux photos ensemble"))
    assert len(harness.llm.image_calls[0]["references"]) == 2


async def test_unknown_image_id_is_reported_to_the_model(harness: Harness):
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "x", "reference_images": ["img9"]}}, "je n'ai pas cette image"]
    await harness.run(group_message(PAUL, "jarvis retouche la photo d'hier"))
    assert harness.llm.image_calls == []
    assert "Images inconnues : img9" in tool_outputs(harness.llm.chat_calls[-1])[0]


async def test_agent_image_limit(mocker, tmp_path):
    h = Harness(mocker, tmp_path, images_per_day=1)
    h.llm.script = [{"name": "generate_image", "arguments": {"prompt": "un chat"}}, "ok",
                    {"name": "generate_image", "arguments": {"prompt": "un chien"}}, "désolé, limite atteinte"]
    await h.run(group_message(PAUL, "jarvis dessine un chat"))
    await h.run(group_message(PAUL, "jarvis dessine un chien"))
    assert len(h.llm.image_calls) == 1
    assert "Refusé" in tool_outputs(h.llm.chat_calls[-1])[0]


async def test_private_chat_is_a_conversation_with_thread(harness: Harness):
    await harness.run(group_message(PAUL, "salut, tu fais quoi ?", private=True))
    assert harness.llm.chat_calls[0]["messages"][-1]["role"] == "user"
    assert "Tu es Jarvis" in harness.llm.last_system()
    bot_ts = harness.bot_timestamps(PAUL[1])[-1]
    assert harness.store.get(PAUL[1], bot_ts).kind == KIND_BOT

    await harness.run(group_message(PAUL, "et ensuite ?", private=True, quote=bot_quote(bot_ts, "réponse du modèle")))
    roles = [m["role"] for m in harness.llm.chat_calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_group_message_mentioning_the_bot_gets_an_answer(harness: Harness):
    await harness.run(
        group_message(PAUL, "blabla"),
        group_message(LEA, "@J t'en penses quoi ?", mentions=[{"name": "J", "number": harness.settings.phone_number,
                                                             "uuid": "u", "start": 0, "length": 2}]),
    )
    assert len(harness.llm.chat_calls) == 1 and "Paul : blabla" in harness.llm.last_system()
    assert len(harness.sent()) == 1


async def test_plain_group_messages_stay_silent(harness: Harness):
    await harness.run(group_message(PAUL, "juste un message"), group_message(LEA, "/foo bar"))
    assert harness.sent() == [] and harness.llm.chat_calls == []


async def test_rate_limit(mocker, tmp_path):
    h = Harness(mocker, tmp_path, rate_limit_per_hour=1)
    await h.run(group_message(PAUL, "jarvis a ?"), group_message(PAUL, "jarvis b ?"), group_message(LEA, "jarvis c ?"))
    texts = h.sent_texts()
    assert texts[0] == "réponse du modèle" and texts[1].startswith("Doucement") and texts[2] == "réponse du modèle"


async def test_edit_updates_stored_text(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(group_message(PAUL, "verison avec faute", ts=t1), edit_message(PAUL, "version corrigée", t1))
    assert harness.store.get(CHAT_ID, t1).text == "version corrigée"


async def test_messages_from_the_bots_own_phone(harness: Harness):
    # In private: the owner's message is stored, not answered, not attributed to the contact
    t1 = _new_timestamp()
    await harness.run(sync_message("jarvis je te réponds moi-même", to_user=PAUL, ts=t1))
    assert harness.llm.chat_calls == [] and harness.sent() == []
    assert harness.store.get(PAUL[1], t1).author_name == "Jarvis"
    # In a group: calling the bot by name from its own phone works, plain messages don't trigger it
    await harness.run(group_message(PAUL, "salut"), sync_message("coucou"), sync_message("jarvis résume"))
    assert len(harness.llm.chat_calls) == 1 and "Paul : salut" in harness.llm.last_system()


async def test_image_in_progress_does_not_hold_up_other_answers(harness: Harness):
    harness.llm.image_gate = asyncio.Event()
    # the second request takes the next scripted reply while the first waits on its image
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "un chat"}}, "salut paul", "voilà"]
    await harness.run(group_message(PAUL, "jarvis fais un chat"), drain=False)
    await asyncio.sleep(0)  # the image request is now waiting on the image
    await harness.run(group_message(PAUL, "jarvis salut"), drain=False)
    for _ in range(20):
        await asyncio.sleep(0)
    assert harness.sent_texts() == ["salut paul"]
    harness.llm.image_gate.set()
    await harness.handler.drain()
    assert harness.sent_texts() == ["salut paul", "voilà"]
