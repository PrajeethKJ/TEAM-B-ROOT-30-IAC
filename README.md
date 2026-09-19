# BioPrint: Behavior-Based Login Security
> **Passwordless-Proof Identity Through Behavioral Biometrics**  
> *ROOT 36 Hackathon (IAC 8.0), IIT Palakkad — 19 to 20 September 2026*  
> **Team:** TEAM-B &bull; **Repository:** [https://github.com/PrajeethKJ/TEAM-B-ROOT-30-IAC.git](https://github.com/PrajeethKJ/TEAM-B-ROOT-30-IAC.git)

---

## 🛡️ Executive Summary
Passwords can be phished, guessed, leaked, or stolen. Even when an adversary has acquired the user's exact plaintext credentials, **BioPrint** ensures unauthorized access is strictly blocked by validating **how the operator interacts with the machine**, rather than just *what string they type*.

BioPrint captures high-dimensional neuromotor telemetry during form entry:
* **Keystroke Dynamics:** Key hold dwell times, inter-key flight latencies (Up-to-Down, Down-to-Down), digraph profiles, and typing rhythm cadence.
* **Pointer & Mouse Kinematics:** Trajectory curvature entropy ($\kappa > 1.10$ for humans vs $\kappa \approx 1.000$ for bots), micro-tremor angular jitter, velocity profiles, and click hold durations.
* **Autonomous Bot Shield:** Flags synthetic DOM injections (`isTrusted == false`), linear macros, and inhuman speeds ($> 22\text{ CPS}$) as fraud.
* **Zero OTP Fallback:** Completely passwordless-proof verification without SMS, email, or secondary authenticators.

---

## 📋 Deliverables & Compliance Matrix

| Deliverable Requirement | Location in Repository | Compliance Status |
| :--- | :--- | :--- |
| **Git Repository & History** | Root repository with 36h active commit history | ✅ **Complete** |
| **Executable File / Extension** | `run_bioprint.bat` (1-click desktop runner) + `extension/` (Manifest V3) | ✅ **Complete** (0 penalty) |
| **1–2 Page Report** | [`report/BioPrint_Technical_Report.pdf`](report/BioPrint_Technical_Report.pdf) | ✅ **Complete** (0 penalty) |
| **Dummy Login Page** | `frontend/index.html` on `http://localhost:8000` | ✅ **Complete** (0 penalty) |

---

## 🚀 Quickstart: Running on Localhost

### Prerequisites
* Python 3.10+ installed
* Google Chrome (or modern Chromium browser)

### 1-Click Launch (Windows)
Double-click **`run_bioprint.bat`**. This will:
1. Start the high-performance FastAPI biometrics engine on `http://localhost:8000`.
2. Automatically launch your browser to the BioPrint Portal.

### Manual Launch via Terminal
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the BioPrint server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 🧩 Chrome Extension Setup (Manifest V3)
BioPrint includes a Manifest V3 Chrome Extension that injects behavioral biometric telemetry into any external web login form.

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked**.
4. Select the `extension/` directory from this repository:
   `B:\BE_EE\ROOT '30 IAC\ROOT '30 IAC Coding\extension`
5. The **BioPrint** shield badge will now automatically activate on any login form across the web.

---

## 🎯 Live Demo Walkthrough for Judges (3–4 Minutes)

### Step 1: Baseline Enrollment (30 seconds)
1. In the web portal (`http://localhost:8000`), click the **Biometric Enrollment Lab** tab.
2. Enter a username (e.g. `alice`) and master password (e.g. `secret123`).
3. Retype the password naturally in the confirmation box and click **Record Calibration Sample**.
4. Repeat 3 times as prompted. BioPrint constructs Alice's unique baseline motor profile.

### Step 2: Genuine User Login (30 seconds)
1. Switch to the **Dummy Login Portal** tab.
2. Select or enter `alice` and type `secret123` at your natural typing rhythm.
3. Click **Authenticate & Verify Motor Dynamics**.
4. **Result:** Instant green badge: **VERIFIED GENUINE** ($> 85\%$ confidence). The radar chart displays aligned kinematics, and latency is displayed ($< 35\text{ ms}$).

### Step 3: Impostor Rejection (Correct Password, Wrong Human) (45 seconds)
1. Have a teammate (or judge) sit at the keyboard.
2. Enter `alice` and the exact same password: `secret123`.
3. Because the teammate possesses a different typing cadence and mouse trajectory, BioPrint's hybrid Mahalanobis engine detects the divergence.
4. **Result:** Blocked with amber badge: **BLOCKED: BEHAVIORAL MISMATCH** ($< 45\%$ confidence). The explainability card explains the exact motor signals that triggered the block.

### Step 4: Automated Bot Attack Intercepted (30 seconds)
1. Switch to the **Attack & Threat Simulator** tab.
2. Select target account `alice`.
3. Click **Simulate Bot Attack** (or **Linear Macro Bot**).
4. **Result:** Red security alarm: **BLOCKED: AUTOMATED FRAUD** ($0\%$ confidence). The explainability log cites `isTrusted == false` synthetic event delivery or zero-variance timing.

---

## ⚙️ Algorithmic Architecture

```
+-------------------+      +-----------------------+      +-------------------------+
| Raw Browser Input | ---> | Bot & Script Detector | ---> | Feature Extractor (8-D) |
+-------------------+      +-----------------------+      +-------------------------+
                                       |                                |
                                 (Reject Bot)                           v
                                                  +---------------------------------------------+
                                                  | Multi-Model Hybrid Classification Engine    |
                                                  | 1. Regularized Mahalanobis Distance         |
                                                  | 2. Z-Score Manhattan Deviation              |
                                                  | 3. One-Class Isolation Forest Model         |
                                                  | 4. Digraph Latency Profile Correlation      |
                                                  +---------------------------------------------+
                                                                        |
                                                                        v
                                                  +---------------------------------------------+
                                                  | Confidence Fusion & Decision Threshold      |
                                                  | Score >= 60.0% -> ACCEPT                    |
                                                  | Score <  60.0% -> BLOCK                     |
                                                  +---------------------------------------------+
```

### Mathematical Distance Metric
BioPrint computes the **Regularized Mahalanobis Distance**:
$$D_M(x, \mu) = \sqrt{(x - \mu)^T \left(\Sigma + \lambda I\right)^{-1} (x - \mu)}$$
where $\mu$ is the enrolled mean vector, $\Sigma$ is the empirical covariance, and $\lambda I$ is ridge regularization to ensure numerical stability during small-sample matrix inversion.

---

## 🧪 Automated Test Suite
To run the automated biometric test suite:
```bash
python tests/run_tests.py
```
**Tests Covered:**
1. Keystroke dwell & flight time extraction accuracy.
2. Mouse curvature & angular jitter calculations.
3. Detection of synthetic `isTrusted == false` DOM events.
4. Detection of zero-variance scripted delay loops.
5. Genuine user verification vs. human impostor rejection.

---

## 👥 Authors
* **TEAM-B** &bull; ROOT 36 Hackathon (IAC 8.0)  
* IIT Palakkad &bull; September 19–20, 2026

