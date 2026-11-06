// Makes the settings page installable as an app. Nothing is cached: the page shows the
// bot's live state and settings, so every request goes to the bot; offline, the browser
// shows its own error.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {});
