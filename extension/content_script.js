/**
 * BioPrint Chrome Extension Content Script
 * Injected into login pages to monitor behavioral keystroke and mouse telemetry.
 */

(function () {
  console.log("[BioPrint] Behavioral Biometrics Shield Initialized.");

  let keystrokes = [];
  let mouseEvents = [];
  let isListening = false;
  let lastMouseTime = 0;

  function initBioPrint() {
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    if (passwordInputs.length === 0) return;

    // Inject floating badge
    injectStatusBadge();

    // Attach listeners
    passwordInputs.forEach(input => {
      input.addEventListener("focus", startTracking);
      input.addEventListener("blur", () => {
        updateBadge("Ready", "#10b981");
      });
    });

    window.addEventListener("keydown", (e) => {
      if (!isListening) return;
      keystrokes.push({
        type: "keydown",
        key: e.key,
        code: e.code,
        timestamp: performance.now(),
        isTrusted: Boolean(e.isTrusted)
      });
    }, true);

    window.addEventListener("keyup", (e) => {
      if (!isListening) return;
      keystrokes.push({
        type: "keyup",
        key: e.key,
        code: e.code,
        timestamp: performance.now(),
        isTrusted: Boolean(e.isTrusted)
      });
    }, true);

    window.addEventListener("mousemove", (e) => {
      if (!isListening) return;
      const now = performance.now();
      if (now - lastMouseTime < 16) return;
      lastMouseTime = now;
      mouseEvents.push({
        type: "mousemove",
        x: e.clientX,
        y: e.clientY,
        timestamp: now,
        isTrusted: Boolean(e.isTrusted)
      });
    }, true);
  }

  function startTracking() {
    isListening = true;
    keystrokes = [];
    mouseEvents = [];
    updateBadge("Analyzing Cadence...", "#06b6d4");
  }

  function injectStatusBadge() {
    if (document.getElementById("bioprint-badge")) return;

    const badge = document.createElement("div");
    badge.id = "bioprint-badge";
    badge.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999999;
      background: #0f172a;
      color: #38bdf8;
      border: 1px solid #06b6d4;
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      gap: 6px;
      pointer-events: none;
      transition: all 0.3s ease;
    `;
    badge.innerHTML = `
      <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981;"></span>
      <span id="bioprint-status-text">BioPrint Protected</span>
    `;
    document.body.appendChild(badge);
  }

  function updateBadge(text, dotColor) {
    const textEl = document.getElementById("bioprint-status-text");
    const badge = document.getElementById("bioprint-badge");
    if (textEl && badge) {
      textEl.textContent = `BioPrint: ${text}`;
      const dot = badge.querySelector("span");
      if (dot) dot.style.background = dotColor;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initBioPrint);
  } else {
    initBioPrint();
  }
})();
