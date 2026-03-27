/**
 * service_worker.js — EC224 Homework Portal
 * Minimal PWA service worker:
 *   - Install-to-homescreen support
 *   - Offline splash page only (no caching of homework data)
 *   - Safe: never intercepts API or Sheets calls
 */

const CACHE_NAME = "ec224-v11";
const OFFLINE_URL = "/offline.html";

// Assets to cache for offline splash
const PRECACHE = [OFFLINE_URL];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Remove old caches
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  // Only handle navigation requests — let everything else pass through
  if (event.request.mode !== "navigate") return;

  event.respondWith(
    fetch(event.request).catch(() =>
      caches.match(OFFLINE_URL)
    )
  );
});
