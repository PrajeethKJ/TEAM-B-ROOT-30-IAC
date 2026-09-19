/**
 * BioPrint Biometric Telemetry Sensor & Event Collector
 * Captures high-resolution keystroke timing (dwell/flight) and mouse kinematics (curvature/velocity/click).
 */

class BioPrintCollector {
  constructor() {
    this.keystrokes = [];
    this.mouseEvents = [];
    this.isCollecting = false;
    this.lastMouseMoveTime = 0;
    this.mouseSampleIntervalMs = 16; // ~60 Hz sampling for smooth trajectory analysis
    this.focusStartTime = 0;

    this._boundKeyDown = this._handleKeyDown.bind(this);
    this._boundKeyUp = this._handleKeyUp.bind(this);
    this._boundMouseMove = this._handleMouseMove.bind(this);
    this._boundMouseDown = this._handleMouseDown.bind(this);
    this._boundMouseUp = this._handleMouseUp.bind(this);
  }

  startCollection(targetInputElements = []) {
    this.reset();
    this.isCollecting = true;
    this.focusStartTime = performance.now();

    // Attach listeners
    window.addEventListener("keydown", this._boundKeyDown, true);
    window.addEventListener("keyup", this._boundKeyUp, true);
    window.addEventListener("mousemove", this._boundMouseMove, true);
    window.addEventListener("mousedown", this._boundMouseDown, true);
    window.addEventListener("mouseup", this._boundMouseUp, true);
  }

  stopCollection() {
    this.isCollecting = false;
    window.removeEventListener("keydown", this._boundKeyDown, true);
    window.removeEventListener("keyup", this._boundKeyUp, true);
    window.removeEventListener("mousemove", this._boundMouseMove, true);
    window.removeEventListener("mousedown", this._boundMouseDown, true);
    window.removeEventListener("mouseup", this._boundMouseUp, true);
  }

  reset() {
    this.keystrokes = [];
    this.mouseEvents = [];
    this.focusStartTime = performance.now();
  }

  _handleKeyDown(e) {
    if (!this.isCollecting) return;
    this.keystrokes.push({
      type: "keydown",
      key: e.key,
      code: e.code,
      timestamp: performance.now(),
      isTrusted: Boolean(e.isTrusted)
    });
  }

  _handleKeyUp(e) {
    if (!this.isCollecting) return;
    this.keystrokes.push({
      type: "keyup",
      key: e.key,
      code: e.code,
      timestamp: performance.now(),
      isTrusted: Boolean(e.isTrusted)
    });
  }

  _handleMouseMove(e) {
    if (!this.isCollecting) return;
    const now = performance.now();
    if (now - this.lastMouseMoveTime < this.mouseSampleIntervalMs) {
      return; // throttle to ~60Hz
    }
    this.lastMouseMoveTime = now;

    this.mouseEvents.push({
      type: "mousemove",
      x: e.clientX,
      y: e.clientY,
      timestamp: now,
      isTrusted: Boolean(e.isTrusted)
    });
  }

  _handleMouseDown(e) {
    if (!this.isCollecting) return;
    this.mouseEvents.push({
      type: "mousedown",
      x: e.clientX,
      y: e.clientY,
      button: e.button,
      timestamp: performance.now(),
      isTrusted: Boolean(e.isTrusted)
    });
  }

  _handleMouseUp(e) {
    if (!this.isCollecting) return;
    this.mouseEvents.push({
      type: "mouseup",
      x: e.clientX,
      y: e.clientY,
      button: e.button,
      timestamp: performance.now(),
      isTrusted: Boolean(e.isTrusted)
    });
  }

  getTelemetry() {
    return {
      keystrokes: [...this.keystrokes],
      mouse: [...this.mouseEvents],
      client_metadata: {
        webdriver: Boolean(navigator.webdriver),
        screen_width: window.screen.width,
        screen_height: window.screen.height,
        user_agent: navigator.userAgent,
        collection_duration_ms: performance.now() - this.focusStartTime
      }
    };
  }
}

// Global instance
window.bioprintCollector = new BioPrintCollector();
