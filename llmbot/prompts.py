"""Shared prompt pieces. Personas go in `prompts_custom.py` (see prompts_custom.example.py)."""

# Appended to every system prompt.
COMMON_RULES = """
Règles générales :
- Réponds dans la langue de la conversation (le plus souvent le français).
- Tu écris pour Signal : texte brut, pas de Markdown, pas de titres, pas de tableaux, pas de gras. Des tirets simples si une liste s'impose.
- Sois court : quelques phrases, sauf si la demande exige plus.
- Pas de formule d'accueil ni de conclusion polie, va droit au but.
- Ce qu'on te donne est structuré par des blocs : [CONVERSATION] est le contexte (date, heure, auteur, texte), [CIBLE] est le message dont il est question. Ne mentionne jamais ces blocs ni les identifiants img…, réponds naturellement.
- Un message précédé de 🎤 est la transcription d'un vocal, un message précédé de 🖼️ est une image (avec sa légende ou sa description).
"""

TRANSCRIBE = """Transcris fidèlement cet enregistrement audio, dans la langue parlée, avec la ponctuation. Ne traduis pas, ne résume pas, n'ajoute aucun commentaire. Si l'audio est inaudible ou vide, réponds exactement : [inaudible]"""

DESCRIBE_IMAGE = """Décris cette image en une phrase courte et factuelle (ce qu'on voit, texte visible s'il y en a). Pas d'introduction, juste la phrase."""
