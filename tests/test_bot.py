from __future__ import annotations

from signalbot.test_utils.chat_testing import _new_timestamp

from llmbot.store import KIND_BOT, KIND_IMAGE, KIND_VOICE
from tests.conftest import (
    CHAT_ID, LEA, PAUL, Harness, bot_quote, edit_message, group_message, image_attachment, quote_of,
    sync_message, voice_attachment,
)


async def test_names_recorded_and_reply_targets_quoted_message(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(
        group_message(PAUL, "La pizza à l'ananas c'est le meilleur truc du monde", ts=t1),
        group_message(LEA, "n'importe quoi"),
        group_message(LEA, "/roast", quote=quote_of(PAUL, t1, "La pizza à l'ananas c'est le meilleur truc du monde")),
    )
    payload = harness.llm.last_user_text()
    assert "[CIBLE]" in payload
    assert "Paul : La pizza à l'ananas" in payload
    # the quoted message is the target; what came after it is not context
    assert "n'importe quoi" not in payload
    assert "+336" not in payload  # names, never numbers
    sent = harness.sent()
    assert len(sent) == 1 and sent[0].message == "réponse du modèle"
    assert sent[0].quote_author == LEA[1]  # the reply quotes the command


async def test_commands_and_bot_replies_are_kept_out_of_context(harness: Harness):
    # separate runs so the bot's answer is dated between the two commands, as in real life
    await harness.run(group_message(PAUL, "on mange où ce soir ?"),
                      group_message(LEA, "/question 5 c'est quoi la capitale du Pérou ?"))
    await harness.run(group_message(PAUL, "/analyse 5"))
    payload = harness.llm.last_user_text()
    assert "/question" not in payload
    assert "on mange où ce soir" in payload
    assert "Jarvis (répond à Léa) : réponse du modèle" in payload  # the bot's own answer is part of the conversation


async def test_resume_covers_only_what_came_after_the_previous_resume(harness: Harness):
    await harness.run(
        group_message(PAUL, "message A"),
        group_message(LEA, "message B"),
        group_message(LEA, "/resume"),
        group_message(PAUL, "message C"),
        group_message(LEA, "message D"),
        group_message(PAUL, "/resume"),
    )
    first, second = harness.llm.chat_calls[0], harness.llm.chat_calls[1]
    assert "message A" in first["messages"][-1]["content"]
    payload = second["messages"][-1]["content"]
    assert "message C" in payload and "message D" in payload
    assert "message A" not in payload and "message B" not in payload
    assert "[CIBLE]" not in payload  # a summary has no single target


async def test_resume_with_nothing_new(harness: Harness):
    await harness.run(group_message(PAUL, "yo"), group_message(LEA, "/resume"), group_message(LEA, "/resume"))
    assert harness.sent_texts()[-1].startswith("Rien de nouveau")


async def test_needs_target_picks_last_message_of_someone_else(harness: Harness):
    await harness.run(
        group_message(PAUL, "je suis le meilleur"),
        group_message(LEA, "ok"),
        group_message(LEA, "/roast"),
    )
    payload = harness.llm.last_user_text()
    assert "[CIBLE]\n" in payload
    assert payload.split("[CIBLE]\n")[1].startswith("[") and "Paul : je suis le meilleur" in payload.split("[CIBLE]\n")[1]


async def test_voice_note_is_transcribed_and_understood(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(
        group_message(PAUL, None, attachments=[voice_attachment()], ts=t1),
        group_message(LEA, "/transcris", quote=quote_of(PAUL, t1, "")),
        group_message(LEA, "/resume 5"),
    )
    assert harness.llm.transcribe_calls and harness.llm.transcribe_calls[0]["format"] == "aac"
    stored = harness.store.get(CHAT_ID, t1)
    assert stored.kind == KIND_VOICE and stored.text == "salut c'est un vocal"
    assert harness.sent_texts()[0] == "🎤 Paul : salut c'est un vocal"
    assert "🎤 salut c'est un vocal" in harness.llm.last_user_text()


async def test_image_generation_then_edit_by_replying_to_the_result(harness: Harness):
    await harness.run(group_message(PAUL, "/image 16:9 un chat sur la lune"))
    call = harness.llm.image_calls[0]
    assert call["aspect_ratio"] == "16:9" and call["prompt"] == "un chat sur la lune" and call["references"] == []
    sent = harness.sent()[0]
    assert sent.base64_attachments[0].startswith("data:image/png;base64,")
    bot_ts = harness.bot_timestamps(CHAT_ID)[-1]
    stored = harness.store.get(CHAT_ID, bot_ts)
    assert stored.kind == KIND_IMAGE and stored.images()

    await harness.run(group_message(LEA, "/image rends-le bleu", quote=bot_quote(bot_ts, "")))
    edit = harness.llm.image_calls[1]
    assert edit["prompt"] == "rends-le bleu" and edit["aspect_ratio"] == "auto"
    assert len(edit["references"]) == 1 and edit["references"][0][1] == "image/png"


async def test_edit_requires_a_reference_image(harness: Harness):
    await harness.run(group_message(PAUL, "/edit plus de contraste"))
    assert harness.llm.image_calls == []
    assert harness.sent_texts()[0].startswith("Réponds à une image")


async def test_image_sent_with_the_command_is_used_as_reference(harness: Harness):
    await harness.run(group_message(PAUL, "/image mets un chapeau", attachments=[image_attachment()]))
    assert len(harness.llm.image_calls[0]["references"]) == 1


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
        group_message(LEA, "@Jarvis t'en penses quoi ?", mentions=[{"name": "Jarvis", "number": harness.settings.phone_number,
                                                               "uuid": "u", "start": 0, "length": 4}]),
    )
    assert len(harness.llm.chat_calls) == 1
    assert "Paul : blabla" in harness.llm.last_system()
    assert len(harness.sent()) == 1


async def test_plain_group_messages_and_unknown_commands_stay_silent(harness: Harness):
    await harness.run(group_message(PAUL, "juste un message"), group_message(LEA, "/foo bar"))
    assert harness.sent() == [] and harness.llm.chat_calls == []


async def test_rate_limit(mocker, tmp_path):
    h = Harness(mocker, tmp_path, rate_limit_per_hour=1)
    await h.run(group_message(PAUL, "/question a ?"), group_message(PAUL, "/question b ?"), group_message(LEA, "/question c ?"))
    texts = h.sent_texts()
    assert texts[0] == "réponse du modèle"
    assert texts[1].startswith("Doucement")
    assert texts[2] == "réponse du modèle"  # per person


async def test_edit_updates_stored_text(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(group_message(PAUL, "verison avec faute", ts=t1), edit_message(PAUL, "version corrigée", t1))
    assert harness.store.get(CHAT_ID, t1).text == "version corrigée"


async def test_model_switch_and_help(harness: Harness):
    await harness.run(group_message(PAUL, "/modele openai/gpt-5-mini"), group_message(PAUL, "/question test"))
    assert harness.llm.chat_calls[-1]["model"] == "openai/gpt-5-mini"
    await harness.run(group_message(PAUL, "/modele nope/nope"), group_message(PAUL, "/modele reset"), group_message(PAUL, "/aide"))
    texts = harness.sent_texts()
    assert any(t.startswith("Modèle inconnu") for t in texts)
    assert "/image" in texts[-1] and "/resume" in texts[-1]


async def test_messages_from_the_bots_own_phone(harness: Harness):
    # In private: the owner's message is stored, not answered, not attributed to the contact
    t1 = _new_timestamp()
    await harness.run(sync_message("je te réponds moi-même", to_user=PAUL, ts=t1))
    assert harness.llm.chat_calls == [] and harness.sent() == []
    assert harness.store.get(PAUL[1], t1).author_name == "Jarvis"
    # In a group: commands typed from the bot's phone work
    await harness.run(group_message(PAUL, "salut"), sync_message("/resume 5"))
    assert len(harness.llm.chat_calls) == 1 and "Paul : salut" in harness.llm.last_user_text()


async def test_resume_ignores_its_own_previous_answer_and_case(harness: Harness):
    await harness.run(group_message(PAUL, "A"), group_message(LEA, "/Resume"), group_message(LEA, "/resume"))
    assert harness.sent_texts()[-1].startswith("Rien de nouveau")
    await harness.run(group_message(PAUL, "B"), group_message(LEA, "/resume"))
    payload = harness.llm.last_user_text()
    assert "Paul : B" in payload and "réponse du modèle" not in payload and "Paul : A" not in payload


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
    assert "Paul : on part à 9h" in first["messages"][0]["content"]  # recent window in the system prompt
    tool_msg = [m for m in second["messages"] if m["role"] == "tool"][0]
    assert "Léa : ok pour moi" in tool_msg["content"]


async def test_name_with_accent_and_silence(harness: Harness):
    harness.llm.script = ["[rien]"]
    await harness.run(group_message(PAUL, "JÁRVIS a encore planté hier lol"))
    assert len(harness.llm.chat_calls) == 1 and harness.sent() == []
    await harness.run(group_message(PAUL, "jarvisse n'est pas un mot"), group_message(LEA, "mon jarvis@home"))
    assert len(harness.llm.chat_calls) == 1  # not a word match


async def test_agent_generates_an_image_from_several_references(harness: Harness):
    t1, t2 = _new_timestamp(), _new_timestamp()
    await harness.run(group_message(PAUL, "ma photo", attachments=[image_attachment()], ts=t1),
                      group_message(LEA, "la mienne", attachments=[image_attachment()], ts=t2))
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "les deux personnes ensemble",
                                                                   "reference_images": ["img1", "img2"]}}, "Et voilà"]
    await harness.run(group_message(PAUL, "Jarvis mets nos deux photos ensemble"))
    call = harness.llm.image_calls[0]
    assert len(call["references"]) == 2 and call["aspect_ratio"] == "auto"
    sent = harness.sent()[-1]
    assert sent.message == "Et voilà" and sent.base64_attachments[0].startswith("data:image/png;base64,")
    assert harness.store.get(CHAT_ID, harness.bot_timestamps(CHAT_ID)[-1]).kind == KIND_IMAGE


async def test_agent_image_limit(mocker, tmp_path):
    h = Harness(mocker, tmp_path, images_per_day=1)
    h.llm.script = [{"name": "generate_image", "arguments": {"prompt": "un chat"}}, "ok",
                    {"name": "generate_image", "arguments": {"prompt": "un chien"}}, "désolé, limite atteinte"]
    await h.run(group_message(PAUL, "jarvis dessine un chat"))
    await h.run(group_message(PAUL, "jarvis dessine un chien"))
    assert len(h.llm.image_calls) == 1
    assert "Refusé" in [m for m in h.llm.chat_calls[-1]["messages"] if m["role"] == "tool"][0]["content"]


async def test_reply_to_bot_targets_quoted_message(harness: Harness):
    t1 = _new_timestamp()
    await harness.run(group_message(PAUL, "la terre est plate", ts=t1))
    await harness.run(group_message(LEA, "jarvis roast ça", quote=quote_of(PAUL, t1, "la terre est plate")))
    assert "[CIBLE]" in harness.llm.last_user_text() and "Paul : la terre est plate" in harness.llm.last_user_text()
