/**
 * BioPrint: Telemetry Visual HUD & Canvas Radar
 * Renders real-time biometric confidence gauge, radar telemetry, and explainability cards.
 */

class TelemetryHUD {
  constructor() {
    this.radarCanvas = document.getElementById("radarCanvas");
    this.radarCtx = this.radarCanvas ? this.radarCanvas.getContext("2d") : null;
    this.circumference = 2 * Math.PI * 55; // r=55 for 130px svg circle
  }

  updateScore(score, decision, latencyMs = 0) {
    const scoreValEl = document.getElementById("scoreValue");
    const progCircle = document.getElementById("scoreProgCircle");
    const badgeEl = document.getElementById("decisionBadge");
    const latencyEl = document.getElementById("latencyValue");

    if (scoreValEl) scoreValEl.textContent = `${Math.round(score)}%`;
    if (latencyEl) latencyEl.textContent = `${latencyMs}ms`;

    // Calculate stroke offset
    if (progCircle) {
      const offset = this.circumference - (score / 100) * this.circumference;
      progCircle.style.strokeDashoffset = offset;

      if (decision === "AUTHENTICATED") {
        progCircle.style.stroke = "#10b981"; // Emerald
      } else if (decision === "BLOCKED_IMPOSTOR") {
        progCircle.style.stroke = "#f59e0b"; // Amber
      } else if (decision === "BLOCKED_BOT") {
        progCircle.style.stroke = "#ef4444"; // Red
      } else {
        progCircle.style.stroke = "#06b6d4"; // Cyan
      }
    }

    if (badgeEl) {
      badgeEl.className = "decision-badge";
      if (decision === "AUTHENTICATED") {
        badgeEl.classList.add("decision-pass");
        badgeEl.textContent = "VERIFIED GENUINE";
      } else if (decision === "BLOCKED_IMPOSTOR") {
        badgeEl.classList.add("decision-blocked");
        badgeEl.textContent = "BLOCKED: BEHAVIORAL MISMATCH";
      } else if (decision === "BLOCKED_BOT") {
        badgeEl.classList.add("decision-bot");
        badgeEl.textContent = "BLOCKED: AUTOMATED FRAUD";
      } else {
        badgeEl.classList.add("decision-waiting");
        badgeEl.textContent = "AWAITING INPUT";
      }
    }
  }

  updateGauges(signalGauges) {
    if (!signalGauges) return;

    this._setGauge("gaugeDwell", signalGauges.keystroke_dwell_match || 0);
    this._setGauge("gaugeFlight", signalGauges.flight_cadence_match || 0);
    this._setGauge("gaugeCursor", signalGauges.cursor_trajectory_match || 0);
    this._setGauge("gaugeAuthenticity", signalGauges.human_authenticity || 0);
  }

  _setGauge(id, val) {
    const valEl = document.getElementById(`${id}Val`);
    const barEl = document.getElementById(`${id}Bar`);
    if (valEl) valEl.textContent = `${Math.round(val)}%`;
    if (barEl) {
      barEl.style.width = `${Math.max(0, Math.min(100, val))}%`;
      if (val < 50) {
        barEl.style.background = "#ef4444";
      } else if (val < 70) {
        barEl.style.background = "#f59e0b";
      } else {
        barEl.style.background = "#06b6d4";
      }
    }
  }

  updateExplainability(explanation) {
    const box = document.getElementById("explainBox");
    if (!box || !explanation) return;

    let html = `<h4>${explanation.headline || "Diagnostic Summary"}</h4>`;
    html += `<p style="margin-bottom: 0.5rem;">${explanation.summary || ""}</p>`;

    if (explanation.bullet_points && explanation.bullet_points.length > 0) {
      html += "<ul>";
      for (const pt of explanation.bullet_points) {
        html += `<li>${pt}</li>`;
      }
      html += "</ul>";
    }

    box.innerHTML = html;
  }

  drawRadar(featureMetrics = null) {
    if (!this.radarCtx || !this.radarCanvas) return;
    const ctx = this.radarCtx;
    const width = this.radarCanvas.width;
    const height = this.radarCanvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 25;

    ctx.clearRect(0, 0, width, height);

    const labels = [
      "Dwell Stability",
      "Flight Rhythm",
      "Speed (CPS)",
      "Curvature",
      "Velocity Var",
      "Click Dwell"
    ];
    const totalAxes = labels.length;

    // Draw background concentric webs
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let r = 0.25; r <= 1.0; r += 0.25) {
      ctx.beginPath();
      for (let i = 0; i < totalAxes; i++) {
        const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
        const x = centerX + Math.cos(angle) * radius * r;
        const y = centerY + Math.sin(angle) * radius * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw radial axes and labels
    ctx.font = "9px -apple-system, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();

      // Label
      const lx = centerX + Math.cos(angle) * (radius + 16);
      const ly = centerY + Math.sin(angle) * (radius + 16);
      ctx.fillText(labels[i], lx, ly);
    }

    // Default or passed values (normalized 0 to 1)
    let values = [0.8, 0.85, 0.75, 0.9, 0.7, 0.85];
    if (featureMetrics) {
      values = [
        Math.max(0.1, 1.0 - (featureMetrics.dwell_time || 0) * 0.2),
        Math.max(0.1, 1.0 - (featureMetrics.flight_time || 0) * 0.2),
        Math.max(0.1, 1.0 - (featureMetrics.typing_speed || 0) * 0.2),
        Math.max(0.1, 1.0 - (featureMetrics.mouse_curvature || 0) * 0.2),
        Math.max(0.1, 1.0 - (featureMetrics.mouse_velocity || 0) * 0.2),
        Math.max(0.1, 1.0 - (featureMetrics.click_dwell || 0) * 0.2),
      ];
    }

    // Draw data polygon
    ctx.beginPath();
    ctx.fillStyle = "rgba(6, 182, 212, 0.25)";
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;

    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const val = Math.max(0.1, Math.min(1.0, values[i]));
      const x = centerX + Math.cos(angle) * (radius * val);
      const y = centerY + Math.sin(angle) * (radius * val);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

window.telemetryHUD = new TelemetryHUD();

