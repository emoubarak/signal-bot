# Faire tourner le bot sur un PC avec Podman (Quadlet)

Même méthode que Kuma : `systemd --user` pilote les conteneurs, ils redémarrent seuls au boot (`loginctl enable-linger`).

    mkdir -p ~/.config/containers/systemd ~/.local/share/signal-bot
    git clone git@github.com:emoubarak/signal-bot.git ~/.local/share/signal-bot/src   # la copie qui tourne
    cp deploy/local/*.container ~/.config/containers/systemd/
    cp deploy/local/signal-bot-update.* ~/.config/systemd/user/
    # les réglages : un fichier .env classique (voir .env.example), jamais dans le dépôt
    install -m 600 /chemin/vers/bot.env signal-bot-secrets/bot.env
    systemctl --user daemon-reload
    ~/.local/share/signal-bot/src/deploy/local/update.sh      # construit l'image
    systemctl --user enable --now signal-rest.service signal-bot.service signal-bot-update.timer

- Journaux : `podman logs -f signal-bot` ; état : `systemctl --user status signal-bot signal-rest`.
- Page d'administration : `http://<adresse Tailscale>:8450` (mot de passe : `ADMIN_TOKEN` de `bot.env`).
- Sauvegarde : `podman volume export signal-data > signal-data.tar` (le compte Signal lié) et `signal-bot-data`
  (la base du bot). Sans `signal-data`, il faut relier l'appareil.
- Le minuteur applique chaque push sur `main` (voir `update.sh` : retour arrière automatique).
