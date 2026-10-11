#!/bin/bash
# Run from cron on the server: when origin/main has moved, fast-forward and
# rebuild the bot. If the new version does not stay up for 30 s, go back to
# the previous commit and skip the broken one until main moves again.
set -u
cd "$(dirname "$0")/.." || exit 1
exec 9>/tmp/signal-bot-autodeploy.lock
flock -n 9 || exit 0
log() { echo "$(date -Is) $*"; }

git fetch -q origin main || { log "fetch failed"; exit 1; }
old=$(git rev-parse HEAD)
new=$(git rev-parse origin/main)
[ "$old" = "$new" ] && exit 0
[ "$new" = "$(cat deploy/.failed 2>/dev/null)" ] && exit 0

log "deploy ${old:0:7} -> ${new:0:7}"
git merge -q --ff-only origin/main || { log "fast-forward impossible (local changes?)"; exit 1; }
if docker compose up -d --build bot >/dev/null 2>&1 && sleep 30 &&
   [ "$(docker inspect -f '{{.State.Running}} {{.RestartCount}}' signal-bot 2>/dev/null)" = "true 0" ]; then
  log "ok"
  exit 0
fi
log "${new:0:7} does not stay up, back to ${old:0:7}"
docker logs --tail 20 signal-bot 2>&1 | sed 's/^/  | /'
echo "$new" > deploy/.failed
git reset -q --hard "$old"
docker compose up -d --build bot >/dev/null 2>&1
exit 1
