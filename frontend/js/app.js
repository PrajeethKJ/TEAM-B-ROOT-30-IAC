/**
 * BioPrint: Main Application Controller
 * Handles UI tabs, enrollment workflow, login verification, simulator executions, and WebSocket telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Tab switching
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.style.display = "none");

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.style.display = "block";

      if (tab.dataset.tab === "audit-tab") {
        fetchAuditLogs();
      }
    });
  });

  // Initialize Canvas Radar
  window.telemetryHUD.drawRadar();

  // Load existing enrolled profiles into dropdowns
  loadProfiles();

  // Setup WebSocket Telemetry
  setupWebSocket();

  // Attach Biometric Collector to Login Form inputs
  setupLoginForm();

  // Setup Enrollment Form
  setupEnrollmentFlow();

  // Setup Simulator Buttons
  setupSimulatorButtons();
});

// WebSocket Setup
function setupWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  try {
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data && data.metrics) {
          window.telemetryHUD.updateScore(data.confidence_score, data.decision, data.latency_ms);
          window.telemetryHUD.updateExplainability(data.explanation);
          if (data.explanation && data.explanation.signal_gauges) {
            window.telemetryHUD.updateGauges(data.explanation.signal_gauges);
          }
          if (data.metrics.feature_z_breakdown) {
            window.telemetryHUD.drawRadar(data.metrics.feature_z_breakdown);
          }
        }
      } catch (e) {
        console.error("WS Parse Error", e);
      }
    };
  } catch (e) {
    console.warn("WebSocket connection not available", e);
  }
}

// Load profiles
async function loadProfiles() {
  try {
    const res = await fetch("/api/profiles");
    const data = await res.json();
    const select = document.getElementById("loginUsernameSelect");
    const simSelect = document.getElementById("simTargetUser");

    if (select && data.profiles) {
      select.innerHTML = '<option value="">-- Select Enrolled User --</option>';
      if (simSelect) simSelect.innerHTML = "";

      data.profiles.forEach(user => {
        const opt = document.createElement("option");
        opt.value = user;
        opt.textContent = user;
        select.appendChild(opt);

        if (simSelect) {
          const optSim = document.createElement("option");
          optSim.value = user;
          optSim.textContent = user;
          simSelect.appendChild(optSim);
        }
      });
    }
  } catch (e) {
    console.warn("Could not load profiles", e);
  }
}

// Login Form Handling
function setupLoginForm() {
  const loginForm = document.getElementById("dummyLoginForm");
  const usernameInput = document.getElementById("loginUsername");
  const passwordInput = document.getElementById("loginPassword");
  const select = document.getElementById("loginUsernameSelect");

  if (select && usernameInput) {
    select.addEventListener("change", () => {
      if (select.value) {
        usernameInput.value = select.value;
      }
    });
  }

  // Start biometrics capture when user focuses password or form
  [usernameInput, passwordInput].forEach(inp => {
    if (!inp) return;
    inp.addEventListener("focus", () => {
      if (!window.bioprintCollector.isCollecting) {
        window.bioprintCollector.startCollection();
      }
    });
  });

  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const username = usernameInput.value.trim();
      const password = passwordInput.value;

      if (!username || !password) {
        alert("Please provide both username and password.");
        return;
      }

      // Collect captured telemetry
      const telemetry = window.bioprintCollector.getTelemetry();
      window.bioprintCollector.stopCollection();

      const payload = {
        username: username,
        password: password,
        keystrokes: telemetry.keystrokes,
        mouse: telemetry.mouse,
        client_metadata: telemetry.client_metadata
      };

      try {
        const response = await fetch("/api/authenticate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const result = await response.json();
        handleAuthResult(result);
      } catch (err) {
        console.error("Auth Request Failed", err);
        alert("Authentication request failed. Check server status.");
      }
    });
  }
}

function handleAuthResult(result) {
  if (result.status === "error") {
    alert(`Authentication Error: ${result.message}`);
    return;
  }

  window.telemetryHUD.updateScore(result.confidence_score, result.decision, result.latency_ms);
  window.telemetryHUD.updateExplainability(result.explanation);

  if (result.explanation && result.explanation.signal_gauges) {
    window.telemetryHUD.updateGauges(result.explanation.signal_gauges);
  }

  if (result.metrics && result.metrics.feature_z_breakdown) {
    window.telemetryHUD.drawRadar(result.metrics.feature_z_breakdown);
  }
}

// Enrollment Flow
let enrollmentSessions = [];
let targetEnrollmentSamples = 3;
let currentSampleIndex = 1;

function setupEnrollmentFlow() {
  const enrollForm = document.getElementById("enrollmentForm");
  const usernameInput = document.getElementById("enrollUsername");
  const passwordInput = document.getElementById("enrollPassword");
  const repeatInput = document.getElementById("enrollPasswordConfirm");
  const sampleCounterEl = document.getElementById("sampleCounter");
  const enrollProgress = document.getElementById("enrollProgress");
  const nextSampleBtn = document.getElementById("nextSampleBtn");

  if (!enrollForm) return;

  repeatInput.addEventListener("focus", () => {
    window.bioprintCollector.startCollection();
  });

  enrollForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    const confirmPw = repeatInput.value;

    if (!username || !password) {
      alert("Please fill all enrollment fields.");
      return;
    }
    if (password !== confirmPw) {
      alert("Passwords do not match!");
      return;
    }

    const telemetry = window.bioprintCollector.getTelemetry();
    window.bioprintCollector.stopCollection();

    if (telemetry.keystrokes.length < 3) {
      alert("Please type naturally into the password fields so your motor dynamics can be recorded.");
      return;
    }

    enrollmentSessions.push(telemetry);

    if (currentSampleIndex < targetEnrollmentSamples) {
      currentSampleIndex++;
      sampleCounterEl.textContent = `Sample ${currentSampleIndex} of ${targetEnrollmentSamples}`;
      enrollProgress.style.width = `${(currentSampleIndex / targetEnrollmentSamples) * 100}%`;
      repeatInput.value = "";
      alert(`Calibration sample ${currentSampleIndex - 1} captured! Please type the password once more for sample ${currentSampleIndex}.`);
      repeatInput.focus();
    } else {
      // Complete enrollment
      sampleCounterEl.textContent = "Finalizing Biometric Baseline Model...";
      try {
        const res = await fetch("/api/enroll", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: username,
            password: password,
            sessions: enrollmentSessions
          })
        });

        const resData = await res.json();
        if (res.ok) {
          alert(`Success! Biometric profile for '${username}' has been enrolled and locked.`);
          enrollmentSessions = [];
          currentSampleIndex = 1;
          sampleCounterEl.textContent = `Sample 1 of ${targetEnrollmentSamples}`;
          enrollProgress.style.width = "33%";
          enrollForm.reset();
          loadProfiles();
          // Switch to login tab
          document.querySelector('[data-tab="login-tab"]').click();
        } else {
          alert(`Enrollment failed: ${resData.detail || "Error"}`);
        }
      } catch (err) {
        console.error("Enrollment error", err);
        alert("Enrollment request failed.");
      }
    }
  });
}

// Attack Simulator Presets
function setupSimulatorButtons() {
  const btnBot1 = document.getElementById("btnSimBot1");
  const btnBot2 = document.getElementById("btnSimBot2");
  const btnImpostor = document.getElementById("btnSimImpostor");
  const simUserSelect = document.getElementById("simTargetUser");
  const simPasswordInput = document.getElementById("simPassword");

  async function executeSimulation(payload) {
    try {
      const response = await fetch("/api/authenticate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      handleAuthResult(result);
    } catch (e) {
      console.error("Simulation failed", e);
      alert("Simulation request failed.");
    }
  }

  if (btnBot1) {
    btnBot1.addEventListener("click", () => {
      const user = simUserSelect.value;
      const pw = simPasswordInput.value || "password123";
      if (!user) { alert("Please select a target user!"); return; }
      const payload = window.attackSimulator.generateBotPayload(user, pw);
      executeSimulation(payload);
    });
  }

  if (btnBot2) {
    btnBot2.addEventListener("click", () => {
      const user = simUserSelect.value;
      const pw = simPasswordInput.value || "password123";
      if (!user) { alert("Please select a target user!"); return; }
      const payload = window.attackSimulator.generateLinearBotPayload(user, pw);
      executeSimulation(payload);
    });
  }

  if (btnImpostor) {
    btnImpostor.addEventListener("click", () => {
      const user = simUserSelect.value;
      const pw = simPasswordInput.value || "password123";
      if (!user) { alert("Please select a target user!"); return; }
      const payload = window.attackSimulator.generateImpostorHumanPayload(user, pw);
      executeSimulation(payload);
    });
  }
}

// Fetch Audit Logs
async function fetchAuditLogs() {
  try {
    const res = await fetch("/api/audit-logs");
    const data = await res.json();
    const tbody = document.getElementById("auditTableBody");
    if (!tbody || !data.logs) return;

    tbody.innerHTML = "";
    if (data.logs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-dim);">No login events logged yet.</td></tr>';
      return;
    }

    data.logs.forEach(log => {
      const tr = document.createElement("tr");
      const dateStr = new Date(log.timestamp * 1000).toLocaleTimeString();

      let badgeClass = "badge-online";
      if (log.decision === "BLOCKED_BOT") badgeClass = "decision-bot";
      else if (log.decision === "BLOCKED_IMPOSTOR") badgeClass = "decision-blocked";
      else badgeClass = "decision-pass";

      tr.innerHTML = `
        <td style="font-family: var(--font-mono);">${dateStr}</td>
        <td><strong>${log.username}</strong></td>
        <td><span class="badge ${badgeClass}">${log.decision}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 700;">${Math.round(log.confidence_score)}%</td>
        <td style="font-family: var(--font-mono);">${log.latency_ms}ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.warn("Could not fetch audit logs", e);
  }
}
