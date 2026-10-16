#!/bin/bash
# Mise à jour de la copie qui tourne sur ce PC : quand origin/main a bougé, tire, reconstruit l'image et
# redémarre le bot. Si la nouvelle version ne reste pas démarrée 30 s, revient à la précédente et ignore la
# cassée jusqu'au prochain commit. Équivalent de deploy/autodeploy.sh (Docker, serveur) pour Podman + Quadlet.
# Lancée par signal-bot-update.timer ; la copie est un clone à part (~/.local/share/signal-bot/src), pour ne pas
# se battre avec le dossier de travail.
set -u
SRC="${SIGNAL_BOT_SRC:-$HOME/.local/share/signal-bot/src}"
cd "$SRC" || exit 1
exec 9>/tmp/signal-bot-update.lock
flock -n 9 || exit 0
log() { echo "$(date -Is) $*"; }

git fetch -q origin main || { log "fetch failed"; exit 1; }
old=$(git rev-parse HEAD)
new=$(git rev-parse origin/main)
first=0
podman image exists localhost/signal-bot:latest || first=1
[ "$old" = "$new" ] && [ "$first" = 0 ] && exit 0
[ "$new" = "$(cat .failed 2>/dev/null)" ] && exit 0

log "deploy ${old:0:7} -> ${new:0:7}"
git merge -q --ff-only origin/main || { log "fast-forward impossible"; exit 1; }
podman build -q -t localhost/signal-bot:new . >/dev/null || {
  log "build failed, staying on ${old:0:7}"; git reset -q --hard "$old"; echo "$new" > .failed; exit 1; }
podman tag localhost/signal-bot:latest localhost/signal-bot:previous 2>/dev/null
podman tag localhost/signal-bot:new localhost/signal-bot:latest
systemctl --user restart signal-bot.service
sleep 30
if [ "$(systemctl --user is-active signal-bot.service)" = "active" ] && [ "$(systemctl --user show -p NRestarts --value signal-bot.service)" = "0" ]; then
  log "ok"
  exit 0
fi
log "${new:0:7} does not stay up, back to ${old:0:7}"
podman logs --tail 20 signal-bot 2>&1 | sed 's/^/  | /'
echo "$new" > .failed
git reset -q --hard "$old"
if podman image exists localhost/signal-bot:previous; then podman tag localhost/signal-bot:previous localhost/signal-bot:latest; fi
systemctl --user restart signal-bot.service
exit 1
