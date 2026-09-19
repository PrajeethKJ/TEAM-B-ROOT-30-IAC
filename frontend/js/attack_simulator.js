/**
 * BioPrint: Built-In Attack & Threat Simulator
 * Allows live demonstration of Scripted Bots, Replay Attacks, and Human Impostors for judges.
 */

class AttackSimulator {
  constructor() {
    this.isSimulating = false;
  }

  generateBotPayload(username, password) {
    // Attack 1: Automated headless script with synthetic events and zero-jitter delays
    const t0 = performance.now();
    const keystrokes = [];
    const fixedDelay = 45.0; // exact identical delay

    let t = t0;
    for (const char of password) {
      keystrokes.push({
        type: "keydown",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: false // Dispatched by synthetic automation script!
      });
      t += 50.0;
      keystrokes.push({
        type: "keyup",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: false
      });
      t += fixedDelay;
    }

    return {
      username: username,
      password: password,
      keystrokes: keystrokes,
      mouse: [], // No physical mouse trajectory
      client_metadata: {
        webdriver: true, // Automation flag active
        screen_width: 1920,
        screen_height: 1080,
        user_agent: "Mozilla/5.0 (HeadlessChrome/120.0)",
        collection_duration_ms: t - t0
      }
    };
  }

  generateLinearBotPayload(username, password) {
    // Attack 2: Straight-line Bresenham cursor path + rapid inhuman typing (25 CPS)
    const t0 = performance.now();
    const keystrokes = [];
    let t = t0;

    for (const char of password) {
      keystrokes.push({
        type: "keydown",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: true
      });
      t += 25.0; // Rapid 25ms dwell
      keystrokes.push({
        type: "keyup",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: true
      });
      t += 15.0; // Rapid 15ms flight -> ~25 CPS!
    }

    // Perfectly straight linear mouse path (0 curvature, 0 jitter)
    const mouseEvents = [];
    const steps = 15;
    const startX = 200, startY = 200;
    const endX = 600, endY = 600;

    for (let i = 0; i <= steps; i++) {
      const frac = i / steps;
      mouseEvents.push({
        type: "mousemove",
        x: startX + (endX - startX) * frac,
        y: startY + (endY - startY) * frac,
        timestamp: t0 + i * 16.6,
        isTrusted: true
      });
    }

    // Click submit
    mouseEvents.push({
      type: "mousedown",
      x: endX,
      y: endY,
      button: 0,
      timestamp: t0 + steps * 16.6 + 5,
      isTrusted: true
    });
    mouseEvents.push({
      type: "mouseup",
      x: endX,
      y: endY,
      button: 0,
      timestamp: t0 + steps * 16.6 + 10,
      isTrusted: true
    });

    return {
      username: username,
      password: password,
      keystrokes: keystrokes,
      mouse: mouseEvents,
      client_metadata: {
        webdriver: false,
        screen_width: 1920,
        screen_height: 1080,
        user_agent: navigator.userAgent,
        collection_duration_ms: t - t0
      }
    };
  }

  generateImpostorHumanPayload(username, password) {
    // Attack 3: Genuine password entered by a different human (heavy dwell, slow hunt-and-peck cadence)
    const t0 = performance.now();
    const keystrokes = [];
    let t = t0;

    for (const char of password) {
      const dwell = 220.0 + (Math.random() * 40 - 20); // Heavy 220ms dwell vs typical 75ms
      const flight = 350.0 + (Math.random() * 80 - 40); // Slow hunt-and-peck search pause

      keystrokes.push({
        type: "keydown",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: true
      });
      t += dwell;
      keystrokes.push({
        type: "keyup",
        key: char,
        code: `Key${char.toUpperCase()}`,
        timestamp: t,
        isTrusted: true
      });
      t += flight;
    }

    // Erratic, meandering cursor path
    const mouseEvents = [];
    let curX = 300, curY = 400;
    for (let i = 0; i < 20; i++) {
      curX += (Math.random() * 60 - 20);
      curY += (Math.random() * 50 - 15);
      mouseEvents.push({
        type: "mousemove",
        x: curX,
        y: curY,
        timestamp: t0 + i * 40,
        isTrusted: true
      });
    }

    mouseEvents.push({
      type: "mousedown",
      x: curX,
      y: curY,
      button: 0,
      timestamp: t0 + 850,
      isTrusted: true
    });
    mouseEvents.push({
      type: "mouseup",
      x: curX,
      y: curY,
      button: 0,
      timestamp: t0 + 1050,
      isTrusted: true
    });

    return {
      username: username,
      password: password,
      keystrokes: keystrokes,
      mouse: mouseEvents,
      client_metadata: {
        webdriver: false,
        screen_width: window.screen.width,
        screen_height: window.screen.height,
        user_agent: navigator.userAgent,
        collection_duration_ms: t - t0
      }
    };
  }
}

window.attackSimulator = new AttackSimulator();

