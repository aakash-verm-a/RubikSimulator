/* ============================================================================
 * 4-Arm Rubik's Cube Robot Simulator Frontend
 *  - Webcam photo capture & OpenCV HSV color recognition (/api/detect_face)
 *  - Interactive 3D Cube + 4 Articulated Robot Arms (HTML5 Canvas 3D)
 *  - Step-by-step & sub-step playback of Arduino CFOP + servo sequences
 * ========================================================================== */

const COLOR_HEX = {
  R: '#ef4444',
  O: '#f97316',
  Y: '#facc15',
  G: '#22c55e',
  B: '#3b82f6',
  W: '#f8fafc',
};

const FACE_INDEX = { FRONT: 0, UP: 1, LEFT: 2, RIGHT: 3, DOWN: 4, BACK: 5 };
const FACE_ORDER_SCAN = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'FRONT', 'BACK'];

const NEUTRAL_SERVOS = {
  Up_Right: 82, Down_Right: 130,
  Up_Left: 83,  Down_Left: 130,
  Up_Front: 80, Down_Front: 130,
  Up_Back: 80,  Down_Back: 130,
};

// Application state
const state = {
  scanSequence: [],
  activeFace: 'RIGHT',
  scannedFaces: new Set(),
  selectedColor: 'R',
  facesVisual: {
    FRONT: Array(9).fill('R'),
    UP:    Array(9).fill('Y'),
    LEFT:  Array(9).fill('B'),
    RIGHT: Array(9).fill('G'),
    DOWN:  Array(9).fill('W'),
    BACK:  Array(9).fill('O'),
  },
  uncertainStickers: {},
  // Simulation playback
  plan: null,
  moveIdx: 0,
  subIdx: 0,
  playing: false,
  speed: 2.0,
  // Live 3D cube matrix (Arduino order: 6 faces x 9 chars)
  cubeMatrix: null,
  servos: { ...NEUTRAL_SERVOS },
  // Animation state
  anim: {
    active: false,
    progress: 0,
    macro: null,
    keyframeIdx: 0,
    kfElapsed: 0,
    rotAngle: 0, // radians
  },
  // Dual opposite-corner isometric 3D cameras (true isometric pitch = atan(1/sqrt(2)) ≈ 0.6155 rad)
  cams: {
    A: { yaw: -Math.PI / 4,     pitch:  0.6155, zoom: 0.96, dragging: false, lastX: 0, lastY: 0 }, // FRONT + UP + RIGHT
    B: { yaw:  3 * Math.PI / 4, pitch: -0.6155, zoom: 0.96, dragging: false, lastX: 0, lastY: 0 }, // BACK + DOWN + LEFT
  },
  mediaStream: null,
  liveDetectTimer: null,
  detectBusy: false,
  lastLiveDetection: null,
  stableKey: '',
  stableCount: 0,
  autoCaptureCooldownUntil: 0,
};

// ---------------------------------------------------------------------------
// Helpers: Arduino <-> Visual Grid conversion
// ---------------------------------------------------------------------------

function visualToArduino(vis) {
  return vis.slice(6, 9).concat(vis.slice(3, 6), vis.slice(0, 3)).join('');
}

function arduinoToVisual(str) {
  const arr = Array.isArray(str) ? str : str.split('');
  return arr.slice(6, 9).concat(arr.slice(3, 6), arr.slice(0, 3));
}

function facesVisualToCubeMatrix(fv) {
  const mat = Array(6).fill(null);
  for (const [name, idx] of Object.entries(FACE_INDEX)) {
    mat[idx] = visualToArduino(fv[name]).split('');
  }
  return mat;
}

function setStatus(msg, type = '') {
  const el = document.getElementById('statusBanner');
  el.innerHTML = msg;
  el.className = 'status-banner' + (type ? ' ' + type : '');
}

// ---------------------------------------------------------------------------
// Webcam & Scanner UI
// ---------------------------------------------------------------------------

async function initApp() {
  const res = await fetch('/api/scan_sequence');
  const data = await res.json();
  state.scanSequence = data.scan_sequence;
  state.cubeMatrix = facesVisualToCubeMatrix(state.facesVisual);

  bindEvents();
  renderFaceEditor();
  renderNetPreview();
  renderServoTelemetry();
  updateScanCard();
  requestAnimationFrame(renderLoop);
}

function bindEvents() {
  // Face tabs
  document.querySelectorAll('.face-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      state.activeFace = btn.dataset.face;
      updateScanCard();
      renderFaceEditor();
    });
  });

  // Color palette
  document.querySelectorAll('.swatch').forEach(btn => {
    btn.addEventListener('click', () => {
      state.selectedColor = btn.dataset.color;
      document.querySelectorAll('.swatch').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
    });
  });

  // Webcam controls
  document.getElementById('btnStartCam').addEventListener('click', toggleWebcam);
  document.getElementById('btnCapture').addEventListener('click', captureFromWebcam);
  document.getElementById('fileUpload').addEventListener('change', handleFileUpload);

  // Top bar actions
  document.getElementById('btnScramble').addEventListener('click', doRandomScramble);
  document.getElementById('btnResetSolved').addEventListener('click', doResetSolved);
  document.getElementById('btnSolve').addEventListener('click', doSolve);

  // Playback controls
  document.getElementById('btnPlay').addEventListener('click', togglePlay);
  document.getElementById('btnRestart').addEventListener('click', restartPlayback);
  document.getElementById('btnNext').addEventListener('click', stepNextMove);
  document.getElementById('btnPrev').addEventListener('click', stepPrevMove);
  document.getElementById('btnSubStep').addEventListener('click', stepNextSubStep);
  document.getElementById('speedSlider').addEventListener('input', e => {
    state.speed = parseFloat(e.target.value);
    document.getElementById('speedLabel').textContent = state.speed.toFixed(1) + 'x';
  });

  // Dual 3D Canvas independent orbit & zoom
  function bindViewportOrbit(canvasId, camKey) {
    const canvas = document.getElementById(canvasId);
    const cam = state.cams[camKey];
    canvas.addEventListener('mousedown', e => {
      cam.dragging = true;
      cam.lastX = e.clientX;
      cam.lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { cam.dragging = false; });
    window.addEventListener('mousemove', e => {
      if (!cam.dragging) return;
      const dx = e.clientX - cam.lastX;
      const dy = e.clientY - cam.lastY;
      cam.lastX = e.clientX;
      cam.lastY = e.clientY;
      cam.yaw += dx * 0.008;
      cam.pitch = Math.max(-1.25, Math.min(1.25, cam.pitch + dy * 0.008));
    });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      cam.zoom = Math.max(0.55, Math.min(1.7, cam.zoom - e.deltaY * 0.001));
    }, { passive: false });
  }
  bindViewportOrbit('simCanvasA', 'A');
  bindViewportOrbit('simCanvasB', 'B');

  document.getElementById('btnResetViews').addEventListener('click', () => {
    state.cams.A.yaw = -Math.PI / 4;     state.cams.A.pitch =  0.6155; state.cams.A.zoom = 0.96;
    state.cams.B.yaw =  3 * Math.PI / 4; state.cams.B.pitch = -0.6155; state.cams.B.zoom = 0.96;
  });
}

async function toggleWebcam() {
  const btn = document.getElementById('btnStartCam');
  const video = document.getElementById('webcamVideo');
  const btnCap = document.getElementById('btnCapture');
  const pill = document.getElementById('detectPill');

  if (state.mediaStream) {
    stopLiveDetection();
    state.mediaStream.getTracks().forEach(t => t.stop());
    state.mediaStream = null;
    video.srcObject = null;
    btn.textContent = '📷 Start Webcam';
    btnCap.disabled = true;
    pill.textContent = 'Camera Off';
    pill.className = 'detect-pill';
    clearDetectOverlay();
    return;
  }

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' },
      audio: false,
    });
    video.srcObject = state.mediaStream;
    btn.textContent = '⏹ Stop Webcam';
    btnCap.disabled = false;
    pill.textContent = 'Searching for cube...';
    pill.className = 'detect-pill';
    setStatus('Webcam active — hold the <strong>' + state.activeFace + '</strong> face anywhere in view. The cube will be detected automatically!');
    startLiveDetection();
  } catch (err) {
    setStatus('Could not access webcam (' + err.message + '). You can upload a photo or use <strong>Random Scramble</strong>.', 'error');
  }
}

function isLocalHost() {
  const h = window.location.hostname;
  return h === 'localhost' || h === '127.0.0.1' || h === '[::1]';
}

function startLiveDetection() {
  stopLiveDetection();
  state.stableKey = '';
  state.stableCount = 0;
  // Snappy 320ms on localhost; gentler 850ms on serverless hosts (e.g. Vercel)
  const intervalMs = isLocalHost() ? 320 : 850;
  state.liveDetectTimer = setInterval(runLiveDetectionTick, intervalMs);
}

function stopLiveDetection() {
  if (state.liveDetectTimer) {
    clearInterval(state.liveDetectTimer);
    state.liveDetectTimer = null;
  }
  state.detectBusy = false;
  state.lastLiveDetection = null;
}

function grabWebcamDataUrl(quality = 0.72) {
  const video = document.getElementById('webcamVideo');
  if (!video.videoWidth || !video.videoHeight) return null;
  const canvas = document.getElementById('captureCanvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', quality);
}

async function runLiveDetectionTick() {
  if (!state.mediaStream || state.detectBusy) return;
  const dataUrl = grabWebcamDataUrl(0.65);
  if (!dataUrl) return;

  state.detectBusy = true;
  try {
    const res = await fetch('/api/detect_face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, include_preview: false }),
    });
    const data = await res.json();
    state.lastLiveDetection = data.detected ? data : null;
    drawDetectOverlay(data);

    const pill = document.getElementById('detectPill');
    if (data.detected) {
      const key = data.visual_grid.join('');
      if (key === state.stableKey) {
        state.stableCount++;
      } else {
        state.stableKey = key;
        state.stableCount = 1;
      }

      const autoOn = document.getElementById('chkAutoCapture').checked;
      const now = Date.now();
      if (autoOn && state.stableCount >= 3 && now >= state.autoCaptureCooldownUntil) {
        pill.textContent = 'Cube Locked ✓ Capturing...';
        pill.className = 'detect-pill found';
        state.autoCaptureCooldownUntil = now + 2200;
        state.stableCount = 0;
        // Run full capture with annotated preview
        await sendImageForDetection(grabWebcamDataUrl(0.88));
      } else {
        pill.textContent = autoOn
          ? `Cube Detected (${Math.min(state.stableCount, 3)}/3 steady)`
          : `Cube Detected (${data.method})`;
        pill.className = 'detect-pill found';
      }
    } else {
      state.stableKey = '';
      state.stableCount = 0;
      pill.textContent = 'Searching for cube...';
      pill.className = 'detect-pill';
    }
  } catch (_) {
    // Ignore transient network errors during live polling
  } finally {
    state.detectBusy = false;
  }
}

function clearDetectOverlay() {
  const overlay = document.getElementById('detectOverlay');
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawDetectOverlay(data) {
  const overlay = document.getElementById('detectOverlay');
  const video = document.getElementById('webcamVideo');
  if (!overlay || !video) return;
  const w = video.clientWidth || 320;
  const h = video.clientHeight || 240;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  if (!data || !data.detected || !data.quad_norm) return;
  const q = data.quad_norm.map(([nx, ny]) => [nx * w, ny * h]);
  // Bilinear interpolation inside detected quad [TL, TR, BR, BL]
  const bilerp = (u, v) => [
    (1 - u) * (1 - v) * q[0][0] + u * (1 - v) * q[1][0] + u * v * q[2][0] + (1 - u) * v * q[3][0],
    (1 - u) * (1 - v) * q[0][1] + u * (1 - v) * q[1][1] + u * v * q[2][1] + (1 - u) * v * q[3][1],
  ];

  // Outer detected face polygon
  ctx.beginPath();
  ctx.moveTo(q[0][0], q[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(q[i][0], q[i][1]);
  ctx.closePath();
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Subdivided 3x3 grid lines following cube perspective
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
  ctx.lineWidth = 1.2;
  for (const t of [1 / 3, 2 / 3]) {
    const [x0, y0] = bilerp(t, 0);
    const [x1, y1] = bilerp(t, 1);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const [a0, b0] = bilerp(0, t);
    const [a1, b1] = bilerp(1, t);
    ctx.beginPath(); ctx.moveTo(a0, b0); ctx.lineTo(a1, b1); ctx.stroke();
  }

  // Detected sticker color badges at each cell center
  ctx.font = 'bold 11px Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const [cx, cy] = bilerp((c + 0.5) / 3, (r + 0.5) / 3);
      const col = data.visual_grid[r * 3 + c];
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = COLOR_HEX[col] || '#94a3b8';
      ctx.fill();
      ctx.strokeStyle = '#0b0f17';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = (col === 'R' || col === 'B') ? '#fff' : '#0b0f17';
      ctx.fillText(col, cx, cy + 0.5);
    }
  }
}

async function captureFromWebcam() {
  const dataUrl = grabWebcamDataUrl(0.9);
  if (!dataUrl) return;
  await sendImageForDetection(dataUrl);
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    // Resize uploaded photos to max 800px so high-res phone photos stay well under Vercel's 4.5 MB limit
    const img = new Image();
    img.onload = async () => {
      const maxDim = 800;
      let w = img.width, h = img.height;
      if (w > maxDim || h > maxDim) {
        const ratio = Math.min(maxDim / w, maxDim / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
      }
      const canvas = document.getElementById('captureCanvas');
      canvas.width = w;
      canvas.height = h;
      canvas.getContext('2d').drawImage(img, 0, 0, w, h);
      await sendImageForDetection(canvas.toDataURL('image/jpeg', 0.85));
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

async function sendImageForDetection(dataUrl) {
  setStatus('Detecting cube face & colors automatically...');
  try {
    const res = await fetch('/api/detect_face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, include_preview: true }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Detection failed');
    if (!data.detected) {
      setStatus(data.message || 'Cube face not detected — hold the cube clearly in front of the camera.', 'error');
      return;
    }

    const face = state.activeFace;
    state.facesVisual[face] = data.visual_grid;
    state.uncertainStickers[face] = data.exact_matches.map(ok => !ok);
    state.scannedFaces.add(face);
    state.cubeMatrix = facesVisualToCubeMatrix(state.facesVisual);

    if (data.annotated_preview) {
      document.getElementById('cropPreviewRow').style.display = 'flex';
      document.getElementById('detectedPreview').src = data.annotated_preview;
      document.getElementById('detectMethodLabel').textContent = `Method: ${data.method}`;
    }

    const uncertainCount = data.exact_matches.filter(ok => !ok).length;
    setStatus(
      `Auto-detected <strong>${face}</strong> face via <code>${data.method}</code>. ` +
      (uncertainCount > 0 ? `${uncertainCount} sticker(s) used HSV fallback (dashed border) — click to tweak if needed.` : 'All 9 stickers matched HSV ranges!'),
      uncertainCount > 0 ? '' : 'ok'
    );

    renderFaceEditor();
    renderNetPreview();

    // Auto-advance to next unscanned face in the robot's scan sequence
    const nextUnscanned = FACE_ORDER_SCAN.find(f => !state.scannedFaces.has(f));
    if (nextUnscanned) {
      state.activeFace = nextUnscanned;
      updateScanCard();
      renderFaceEditor();
    }
  } catch (err) {
    setStatus('Detection error: ' + err.message, 'error');
  }
}

function updateScanCard() {
  document.querySelectorAll('.face-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.face === state.activeFace);
    btn.classList.toggle('scanned', state.scannedFaces.has(btn.dataset.face));
  });
  document.getElementById('scanProgress').textContent = `${state.scannedFaces.size} / 6 scanned`;

  const entry = state.scanSequence.find(s => s.face === state.activeFace) || state.scanSequence[0];
  if (!entry) return;
  document.getElementById('scanFacePill').textContent = entry.face;
  document.getElementById('scanStepTitle').textContent = `Scan Step ${entry.step} of 6`;
  document.getElementById('scanInstruction').textContent = entry.instruction;
  document.getElementById('scanRobotAction').textContent = 'Robot: ' + entry.robot_action;
}

function renderFaceEditor() {
  const grid = document.getElementById('faceGrid');
  grid.innerHTML = '';
  const colors = state.facesVisual[state.activeFace];
  const flags = state.uncertainStickers[state.activeFace] || Array(9).fill(false);

  colors.forEach((col, i) => {
    const cell = document.createElement('div');
    cell.className = `sticker c-${col}` + (flags[i] ? ' uncertain' : '');
    cell.textContent = col;
    cell.title = `Sticker ${i + 1} (click to set to ${state.selectedColor})`;
    cell.addEventListener('click', () => {
      state.facesVisual[state.activeFace][i] = state.selectedColor;
      if (state.uncertainStickers[state.activeFace]) {
        state.uncertainStickers[state.activeFace][i] = false;
      }
      state.cubeMatrix = facesVisualToCubeMatrix(state.facesVisual);
      renderFaceEditor();
      renderNetPreview();
    });
    grid.appendChild(cell);
  });
}

function renderNetPreview() {
  const net = document.getElementById('netPreview');
  net.innerHTML = '';
  const layout = [
    null, 'UP', null, null,
    'LEFT', 'FRONT', 'RIGHT', 'BACK',
    null, 'DOWN', null, null,
  ];
  const mat = state.cubeMatrix || facesVisualToCubeMatrix(state.facesVisual);
  layout.forEach(faceName => {
    if (!faceName) {
      net.appendChild(document.createElement('div'));
      return;
    }
    const vis = arduinoToVisual(mat[FACE_INDEX[faceName]]);
    const div = document.createElement('div');
    div.className = 'mini-face';
    div.title = faceName;
    vis.forEach(c => {
      const cell = document.createElement('div');
      cell.className = `mini-cell c-${c}`;
      div.appendChild(cell);
    });
    net.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// Scramble, Reset & Solve API
// ---------------------------------------------------------------------------

async function doRandomScramble() {
  stopPlayback();
  const res = await fetch('/api/scramble', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves: 18 }),
  });
  const data = await res.json();
  state.facesVisual = data.faces_visual;
  state.uncertainStickers = {};
  state.scannedFaces = new Set(FACE_ORDER_SCAN);
  state.cubeMatrix = facesVisualToCubeMatrix(state.facesVisual);
  state.plan = null;
  updatePlaybackButtons();
  updateScanCard();
  renderFaceEditor();
  renderNetPreview();
  setStatus(`Generated 18-move scramble: <code>${data.scramble}</code>. Click <strong>Solve &amp; Simulate</strong>!`, 'ok');
}

async function doResetSolved() {
  stopPlayback();
  const res = await fetch('/api/solved_state');
  const data = await res.json();
  state.facesVisual = data.faces_visual;
  state.uncertainStickers = {};
  state.scannedFaces = new Set();
  state.cubeMatrix = facesVisualToCubeMatrix(state.facesVisual);
  state.plan = null;
  state.servos = { ...NEUTRAL_SERVOS };
  updatePlaybackButtons();
  updateScanCard();
  renderFaceEditor();
  renderNetPreview();
  renderServoTelemetry();
  setStatus('Reset to solved cube state.');
}

async function doSolve() {
  stopPlayback();
  setStatus('Running ported Arduino CFOP solver & computing 4-arm servo plan...');
  const res = await fetch('/api/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ faces_visual: state.facesVisual }),
  });
  const data = await res.json();
  if (!data.solved) {
    setStatus('Solver error: ' + (data.error || 'Could not solve this cube state.'), 'error');
    return;
  }

  state.plan = data;
  state.moveIdx = 0;
  state.subIdx = 0;
  state.cubeMatrix = data.initial_state.map(r => r.slice());
  state.servos = { ...NEUTRAL_SERVOS };

  // Update stats
  document.getElementById('statLogical').textContent = data.stats.logical_moves;
  document.getElementById('statPhysical').textContent = data.stats.physical_macro_steps;
  document.getElementById('statMulti').textContent = data.stats.multi_step_moves;
  document.getElementById('statTime').textContent = data.stats.estimated_physical_seconds + 's';

  renderTimeline();
  updatePlaybackButtons();
  updateHudAndMacros();
  renderNetPreview();
  renderServoTelemetry();

  setStatus(
    `Solved! <strong>${data.stats.logical_moves}</strong> logical moves (${data.stats.moves_saved_by_optimizer} saved by optimizer) → <strong>${data.stats.physical_macro_steps}</strong> physical 4-arm operations (~${data.stats.estimated_physical_seconds}s real time). Press <strong>Play</strong>!`,
    'ok'
  );
}

// ---------------------------------------------------------------------------
// Playback & Animation Engine
// ---------------------------------------------------------------------------

function updatePlaybackButtons() {
  const hasPlan = Boolean(state.plan && state.plan.timeline.length > 0);
  document.getElementById('btnRestart').disabled = !hasPlan;
  document.getElementById('btnPrev').disabled = !hasPlan;
  document.getElementById('btnPlay').disabled = !hasPlan;
  document.getElementById('btnNext').disabled = !hasPlan;
  document.getElementById('btnSubStep').disabled = !hasPlan;
  document.getElementById('btnPlay').textContent = state.playing ? '⏸ Pause' : '▶ Play';
}

function stopPlayback() {
  state.playing = false;
  state.anim.active = false;
  state.anim.macro = null;
  state.anim.rotAngle = 0;
  updatePlaybackButtons();
}

function togglePlay() {
  if (!state.plan) return;
  if (state.playing) {
    state.playing = false;
    updatePlaybackButtons();
    return;
  }
  if (state.moveIdx >= state.plan.timeline.length) {
    restartPlayback();
  }
  state.playing = true;
  updatePlaybackButtons();
  if (!state.anim.active) {
    startCurrentSubStepAnimation();
  }
}

function restartPlayback() {
  if (!state.plan) return;
  stopPlayback();
  state.moveIdx = 0;
  state.subIdx = 0;
  state.cubeMatrix = state.plan.initial_state.map(r => r.slice());
  state.servos = { ...NEUTRAL_SERVOS };
  updateHudAndMacros();
  renderNetPreview();
  renderServoTelemetry();
  highlightTimeline();
}

function jumpToMove(idx) {
  if (!state.plan) return;
  stopPlayback();
  state.moveIdx = Math.max(0, Math.min(state.plan.timeline.length, idx));
  state.subIdx = 0;
  if (state.moveIdx < state.plan.timeline.length) {
    state.cubeMatrix = state.plan.timeline[state.moveIdx].state_before.map(r => r.slice());
  } else {
    state.cubeMatrix = state.plan.final_state.map(r => r.slice());
  }
  state.servos = { ...NEUTRAL_SERVOS };
  updateHudAndMacros();
  renderNetPreview();
  renderServoTelemetry();
  highlightTimeline();
}

function stepNextMove() {
  if (!state.plan || state.moveIdx >= state.plan.timeline.length) return;
  jumpToMove(state.moveIdx + 1);
}

function stepPrevMove() {
  if (!state.plan) return;
  jumpToMove(Math.max(0, state.moveIdx - 1));
}

function stepNextSubStep() {
  if (!state.plan || state.moveIdx >= state.plan.timeline.length) return;
  state.playing = false;
  updatePlaybackButtons();
  startCurrentSubStepAnimation();
}

function startCurrentSubStepAnimation() {
  if (!state.plan || state.moveIdx >= state.plan.timeline.length) {
    state.playing = false;
    updatePlaybackButtons();
    updateHudAndMacros();
    return;
  }
  const moveObj = state.plan.timeline[state.moveIdx];
  const macro = moveObj.macro_steps[state.subIdx];
  state.cubeMatrix = macro.state_before.map(r => r.slice());
  state.anim = {
    active: true,
    macro,
    keyframeIdx: 0,
    kfElapsed: 0,
    rotAngle: 0,
  };
  updateHudAndMacros();
  highlightTimeline();
}

function advanceAnimation(dtMs) {
  if (!state.anim.active || !state.anim.macro) return;
  const macro = state.anim.macro;
  const kfs = macro.keyframes;
  state.anim.kfElapsed += dtMs * state.speed;

  const kf = kfs[state.anim.keyframeIdx];
  const frac = Math.min(1, state.anim.kfElapsed / Math.max(1, kf.duration_ms));

  // Interpolate servo angles from previous keyframe (or neutral) to current keyframe
  const prevServos = state.anim.keyframeIdx === 0 ? NEUTRAL_SERVOS : kfs[state.anim.keyframeIdx - 1].servos;
  for (const key of Object.keys(NEUTRAL_SERVOS)) {
    state.servos[key] = Math.round(prevServos[key] + (kf.servos[key] - prevServos[key]) * frac);
  }

  // Smooth rotation angle during the rotate_face / rotate_cube keyframe
  if (kf.anim === 'rotate_face' || kf.anim === 'rotate_cube') {
    const eased = frac < 0.5 ? 2 * frac * frac : 1 - Math.pow(-2 * frac + 2, 2) / 2;
    state.anim.rotAngle = eased * (Math.PI / 2) * macro.dir;
  } else if (state.anim.keyframeIdx > 1) {
    state.anim.rotAngle = (Math.PI / 2) * macro.dir;
  } else {
    state.anim.rotAngle = 0;
  }

  document.getElementById('hudPhase').textContent = kf.phase;
  renderServoTelemetry();

  if (state.anim.kfElapsed >= kf.duration_ms) {
    state.anim.kfElapsed = 0;
    state.anim.keyframeIdx++;
    if (state.anim.keyframeIdx >= kfs.length) {
      // Sub-step completed: commit new cube matrix!
      state.cubeMatrix = macro.state_after.map(r => r.slice());
      state.anim.active = false;
      state.anim.rotAngle = 0;
      state.anim.macro = null;
      renderNetPreview();

      // Advance subIdx / moveIdx
      const moveObj = state.plan.timeline[state.moveIdx];
      state.subIdx++;
      if (state.subIdx >= moveObj.macro_steps.length) {
        state.subIdx = 0;
        state.moveIdx++;
      }

      updateHudAndMacros();
      highlightTimeline();

      if (state.playing && state.moveIdx < state.plan.timeline.length) {
        startCurrentSubStepAnimation();
      } else if (state.moveIdx >= state.plan.timeline.length) {
        state.playing = false;
        state.servos = { ...NEUTRAL_SERVOS };
        renderServoTelemetry();
        updatePlaybackButtons();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// HUD, Telemetry & Timeline Renderers
// ---------------------------------------------------------------------------

function updateHudAndMacros() {
  const hudStage = document.getElementById('hudStage');
  const hudMove = document.getElementById('hudMove');
  const hudBadge = document.getElementById('hudBadge');
  const hudDesc = document.getElementById('hudDesc');
  const hudProgress = document.getElementById('hudProgress');
  const macroList = document.getElementById('macroList');

  if (!state.plan) return;
  const total = state.plan.timeline.length;
  hudProgress.textContent = `${Math.min(state.moveIdx, total)} / ${total}`;

  if (state.moveIdx >= total) {
    hudStage.textContent = 'COMPLETE';
    hudMove.textContent = 'SOLVED ✓';
    hudBadge.innerHTML = '';
    hudDesc.textContent = 'All 4-arm robot operations completed. Cube is solved!';
    document.getElementById('hudPhase').textContent = 'Servos at Home';
    macroList.innerHTML = '<div class="macro-chip active"><span>✓ Cube Solved</span><span>Done</span></div>';
    return;
  }

  const mv = state.plan.timeline[state.moveIdx];
  hudStage.textContent = mv.stage.toUpperCase();
  hudMove.textContent = mv.notation;
  hudBadge.innerHTML = mv.is_multi_step
    ? `<span class="badge-multi">${mv.macro_steps.length}-STEP COMPOSITE</span>`
    : `<span class="badge-direct">1-STEP DIRECT</span>`;
  hudDesc.textContent = mv.explanation;

  macroList.innerHTML = '';
  mv.macro_steps.forEach((m, i) => {
    const chip = document.createElement('div');
    chip.className = 'macro-chip' + (i === state.subIdx ? ' active' : '');
    chip.innerHTML = `<span>${i + 1}. ${m.fn}</span><span>${m.duration_ms}ms</span>`;
    macroList.appendChild(chip);
  });
}

function renderServoTelemetry() {
  const grid = document.getElementById('servoGrid');
  grid.innerHTML = '';
  const order = [
    'Up_Right', 'Down_Right',
    'Up_Left',  'Down_Left',
    'Up_Front', 'Down_Front',
    'Up_Back',  'Down_Back',
  ];
  order.forEach(name => {
    const val = state.servos[name];
    const isDown = name.startsWith('Down_');
    const retracted = isDown && val < 95;
    const pct = Math.max(0, Math.min(100, (val / 180) * 100));
    const item = document.createElement('div');
    item.className = 'servo-item' + (retracted ? ' retracted' : '');
    item.innerHTML = `
      <div class="s-top">
        <span>${name}</span>
        <span>${val}°${isDown ? (retracted ? ' OPEN' : ' GRIP') : ''}</span>
      </div>
      <div class="servo-bar"><div style="width:${pct}%"></div></div>
    `;
    grid.appendChild(item);
  });
}

function renderTimeline() {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  if (!state.plan) return;
  state.plan.timeline.forEach((mv, idx) => {
    const pill = document.createElement('button');
    pill.className = 'move-pill' + (mv.is_multi_step ? ' multi' : '');
    pill.textContent = mv.notation;
    pill.title = `Move ${idx + 1}: ${mv.notation} (${mv.macro_steps.length} robot op${mv.macro_steps.length > 1 ? 's' : ''})`;
    pill.dataset.idx = idx;
    pill.addEventListener('click', () => jumpToMove(idx));
    tl.appendChild(pill);
  });
  highlightTimeline();
}

function highlightTimeline() {
  const tl = document.getElementById('timeline');
  document.querySelectorAll('#timeline .move-pill').forEach(p => {
    const i = parseInt(p.dataset.idx, 10);
    p.classList.toggle('active', i === state.moveIdx);
    p.classList.toggle('done', i < state.moveIdx);
    if (i === state.moveIdx && tl) {
      // Scroll ONLY the internal #timeline container without moving the outer page
      const top = p.offsetTop - tl.offsetTop;
      const bottom = top + p.offsetHeight;
      if (top < tl.scrollTop) {
        tl.scrollTop = Math.max(0, top - 4);
      } else if (bottom > tl.scrollTop + tl.clientHeight) {
        tl.scrollTop = bottom - tl.clientHeight + 4;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// 3D Software Renderer (Rubik's Cube + 4 Articulated Robot Arms)
// ---------------------------------------------------------------------------

function rotVecX(v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c];
}
function rotVecY(v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}
function rotVecZ(v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
}

function applyActiveTransform(pt, macro, angle) {
  if (!macro || angle === 0) return pt;
  if (macro.kind === 'cube_rotate') {
    if (macro.axis === 'X') return rotVecX(pt, -angle);
    if (macro.axis === 'Y') return rotVecY(pt, angle);
  }
  if (macro.kind === 'face_turn') {
    const f = macro.physical_face;
    if (f === 'RIGHT') return rotVecX(pt, -angle);
    if (f === 'LEFT')  return rotVecX(pt, angle);
    if (f === 'UP')    return rotVecY(pt, -angle);
    if (f === 'DOWN')  return rotVecY(pt, angle);
  }
  return pt;
}

function isStickerInActiveLayer(faceIdx, stickerIdx, macro) {
  if (!macro) return false;
  if (macro.kind === 'cube_rotate') return true;
  if (macro.kind !== 'face_turn') return false;

  const col = stickerIdx % 3;            // 0=left, 1=mid, 2=right
  const row = Math.floor(stickerIdx / 3); // 0=bottom, 1=mid, 2=top
  const pf = macro.physical_face;

  if (pf === 'RIGHT') {
    if (faceIdx === FACE_INDEX.RIGHT) return true;
    if (faceIdx === FACE_INDEX.FRONT || faceIdx === FACE_INDEX.UP || faceIdx === FACE_INDEX.DOWN) return col === 2;
    if (faceIdx === FACE_INDEX.BACK) return col === 0;
  }
  if (pf === 'LEFT') {
    if (faceIdx === FACE_INDEX.LEFT) return true;
    if (faceIdx === FACE_INDEX.FRONT || faceIdx === FACE_INDEX.UP || faceIdx === FACE_INDEX.DOWN) return col === 0;
    if (faceIdx === FACE_INDEX.BACK) return col === 2;
  }
  if (pf === 'UP') {
    if (faceIdx === FACE_INDEX.UP) return true;
    if (faceIdx === FACE_INDEX.FRONT || faceIdx === FACE_INDEX.RIGHT || faceIdx === FACE_INDEX.LEFT || faceIdx === FACE_INDEX.BACK) return row === 2;
  }
  if (pf === 'DOWN') {
    if (faceIdx === FACE_INDEX.DOWN) return true;
    if (faceIdx === FACE_INDEX.FRONT || faceIdx === FACE_INDEX.RIGHT || faceIdx === FACE_INDEX.LEFT || faceIdx === FACE_INDEX.BACK) return row === 0;
  }
  return false;
}

const FACE_NORMALS = [
  [0, 0, 1],   // FRONT (+Z)
  [0, 1, 0],   // UP (+Y)
  [-1, 0, 0],  // LEFT (-X)
  [1, 0, 0],   // RIGHT (+X)
  [0, -1, 0],  // DOWN (-Y)
  [0, 0, -1],  // BACK (-Z)
];

function faceCellQuad3D(faceIdx, stickerIdx, inset = 0.08) {
  // Returns 4 3D vertices for sticker cell (col 0..2, row 0..2) on cube of half-size 1.5
  const col = stickerIdx % 3;
  const row = Math.floor(stickerIdx / 3);
  const u0 = -1.5 + col + inset, u1 = -1.5 + col + (1 - inset);
  const v0 = -1.5 + row + inset, v1 = -1.5 + row + (1 - inset);
  const S = 1.5;

  switch (faceIdx) {
    case FACE_INDEX.FRONT: // +Z
      return [[u0, v0, S], [u1, v0, S], [u1, v1, S], [u0, v1, S]];
    case FACE_INDEX.BACK:  // -Z (left-to-right looking at back is +X to -X)
      return [[-u0, v0, -S], [-u1, v0, -S], [-u1, v1, -S], [-u0, v1, -S]];
    case FACE_INDEX.RIGHT: // +X (left-to-right is +Z to -Z)
      return [[S, v0, -u0], [S, v0, -u1], [S, v1, -u1], [S, v1, -u0]];
    case FACE_INDEX.LEFT:  // -X (left-to-right is -Z to +Z)
      return [[-S, v0, u0], [-S, v0, u1], [-S, v1, u1], [-S, v1, u0]];
    case FACE_INDEX.UP:    // +Y (row 0 is front +Z, row 2 is back -Z)
      return [[u0, S, -v0], [u1, S, -v0], [u1, S, -v1], [u0, S, -v1]];
    case FACE_INDEX.DOWN:  // -Y (row 2 is front +Z, row 0 is back -Z)
      return [[u0, -S, v0], [u1, -S, v0], [u1, -S, v1], [u0, -S, v1]];
  }
}

function buildArmPolygons(armName, downAngle, upAngle, homeUp, isActive) {
  // Linear slide: clamped (130°) => dist=1.78, retracted (60°) => dist=2.65
  const tGrip = Math.max(0, Math.min(1, (130 - downAngle) / 70));
  const dist = 1.72 + tGrip * 0.95;
  const wristRad = ((upAngle - homeUp) * Math.PI) / 180;

  // Define arm in local +X frame (claw at x=dist, shaft extending to x=dist+1.2), then orient to target axis
  const clawW = 0.72, clawH = 0.22, clawD = 0.25;
  const shaftR = 0.18;

  // Simple 3D box generator around local origin
  function boxFaces(cx, cy, cz, sx, sy, sz, color, stroke) {
    const x0 = cx - sx / 2, x1 = cx + sx / 2;
    const y0 = cy - sy / 2, y1 = cy + sy / 2;
    const z0 = cz - sz / 2, z1 = cz + sz / 2;
    const faces = [
      [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
      [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]],
      [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
      [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]],
      [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
      [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
    ];
    return faces.map(pts => ({ pts, color, stroke }));
  }

  const clawColor = isActive ? '#38bdf8' : (downAngle < 95 ? '#f59e0b' : '#475569');
  const shaftColor = isActive ? '#0284c7' : '#334155';
  const stroke = '#0f172a';

  let localPolys = [
    ...boxFaces(dist, 0, 0, clawD, clawH, clawW, clawColor, stroke),
    ...boxFaces(dist + 0.55, 0, 0, 1.0, shaftR, shaftR, shaftColor, stroke),
  ];

  // Rotate claw+shaft around local X by wristRad, then map +X to the arm's world axis:
  //   RIGHT -> +X, LEFT -> -X, FRONT(U) -> +Y, BACK(D) -> -Y
  return localPolys.map(poly => {
    const pts = poly.pts.map(p => {
      let q = rotVecX(p, wristRad);
      if (armName === 'RIGHT') return q;
      if (armName === 'LEFT')  return [-q[0], q[1], -q[2]];
      if (armName === 'FRONT') return [-q[1], q[0], q[2]];   // +X -> +Y (grips UP face)
      if (armName === 'BACK')  return [q[1], -q[0], q[2]];   // +X -> -Y (grips DOWN face)
      return q;
    });
    return { ...poly, pts };
  });
}

let lastFrameTime = performance.now();

function renderLoop(now) {
  const dt = Math.min(50, now - lastFrameTime);
  lastFrameTime = now;

  if (state.anim.active) {
    advanceAnimation(dt);
  }
  drawViewport('simCanvasA', state.cams.A, true);
  drawViewport('simCanvasB', state.cams.B, false);
  requestAnimationFrame(renderLoop);
}

function drawViewport(canvasId, cam, showFrontAxisTag) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const stage = canvas.parentElement;
  const targetW = Math.max(1, stage.clientWidth);
  const targetH = Math.max(1, stage.clientHeight);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const scale = Math.min(W, H) * 0.165 * cam.zoom;
  const cx = W / 2, cy = H / 2 + 8;
  const camDist = 11.5;

  function toCam(pt) {
    let p = rotVecY(pt, cam.yaw);
    return rotVecX(p, cam.pitch);
  }

  function projectCam(p) {
    const z = camDist - p[2];
    const f = (camDist / Math.max(2, z)) * scale;
    return { x: cx + p[0] * f, y: cy - p[1] * f, z: p[2] };
  }

  function project(pt) {
    return projectCam(toCam(pt));
  }

  const drawables = [];
  const mat = state.cubeMatrix || facesVisualToCubeMatrix(state.facesVisual);
  const activeMacro = state.anim.active ? state.anim.macro : null;
  const activeAngle = state.anim.active ? state.anim.rotAngle : 0;

  // 1. All 54 Rubik's Cube cubie faces: each tile renders its own black plastic cell
  //    + inset colored sticker atomically, with 3D outward-normal back-face culling
  //    so no backing plate can ever hide stickers on the same face.
  for (let f = 0; f < 6; f++) {
    for (let i = 0; i < 9; i++) {
      const inLayer = isStickerInActiveLayer(f, i, activeMacro);
      let tileQuad = faceCellQuad3D(f, i, 0.0);
      let stickerQuad = faceCellQuad3D(f, i, 0.085);
      let normal = FACE_NORMALS[f];

      if (inLayer && activeAngle !== 0) {
        tileQuad = tileQuad.map(p => applyActiveTransform(p, activeMacro, activeAngle));
        stickerQuad = stickerQuad.map(p => applyActiveTransform(p, activeMacro, activeAngle));
        normal = applyActiveTransform(normal, activeMacro, activeAngle);
      }

      const tileCam = tileQuad.map(toCam);
      const normCam = toCam(normal);
      const centerCam = [
        (tileCam[0][0] + tileCam[1][0] + tileCam[2][0] + tileCam[3][0]) * 0.25,
        (tileCam[0][1] + tileCam[1][1] + tileCam[2][1] + tileCam[3][1]) * 0.25,
        (tileCam[0][2] + tileCam[1][2] + tileCam[2][2] + tileCam[3][2]) * 0.25,
      ];
      // View vector from tile center to camera at (0, 0, camDist)
      const dot =
        normCam[0] * (0 - centerCam[0]) +
        normCam[1] * (0 - centerCam[1]) +
        normCam[2] * (camDist - centerCam[2]);
      if (dot <= 0) continue; // Back-face culled

      const ch = mat[f][i];
      drawables.push({
        type: 'tile',
        tileP2: tileCam.map(projectCam),
        stickerP2: stickerQuad.map(project),
        color: COLOR_HEX[ch] || '#94a3b8',
        avgZ: centerCam[2],
      });
    }
  }

  // 2. The 4 physical robot arms (RIGHT, LEFT, FRONT->UP, BACK->DOWN)
  const activeArms = new Set(activeMacro ? activeMacro.arms : []);
  const armSpecs = [
    { name: 'RIGHT', down: state.servos.Down_Right, up: state.servos.Up_Right, home: 82, labelPos: [3.2, 0, 0], label: 'RIGHT ARM (R)' },
    { name: 'LEFT',  down: state.servos.Down_Left,  up: state.servos.Up_Left,  home: 83, labelPos: [-3.2, 0, 0], label: 'LEFT ARM (L)' },
    { name: 'FRONT', down: state.servos.Down_Front, up: state.servos.Up_Front, home: 80, labelPos: [0, 3.15, 0], label: 'FRONT ARM (U)' },
    { name: 'BACK',  down: state.servos.Down_Back,  up: state.servos.Up_Back,  home: 80, labelPos: [0, -3.15, 0], label: 'BACK ARM (D)' },
  ];

  armSpecs.forEach(a => {
    const armPolys = buildArmPolygons(a.name, a.down, a.up, a.home, activeArms.has(a.name));
    armPolys.forEach(poly => {
      const p2 = poly.pts.map(project);
      const avgZ = p2.reduce((acc, q) => acc + q.z, 0) / p2.length;
      drawables.push({
        type: 'poly',
        p2,
        color: poly.color,
        stroke: poly.stroke || '#090d14',
        lineWidth: poly.lineWidth || 1.2,
        avgZ,
      });
    });
  });

  // Painter-sort back-to-front by average Z
  drawables.sort((a, b) => a.avgZ - b.avgZ);

  drawables.forEach(item => {
    if (item.type === 'tile') {
      // Draw black plastic tile cell
      ctx.beginPath();
      ctx.moveTo(item.tileP2[0].x, item.tileP2[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(item.tileP2[i].x, item.tileP2[i].y);
      ctx.closePath();
      ctx.fillStyle = '#0f172a';
      ctx.fill();
      ctx.strokeStyle = '#020617';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw colored sticker directly on top of its own tile
      ctx.beginPath();
      ctx.moveTo(item.stickerP2[0].x, item.stickerP2[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(item.stickerP2[i].x, item.stickerP2[i].y);
      ctx.closePath();
      ctx.fillStyle = item.color;
      ctx.fill();
      ctx.strokeStyle = '#090d14';
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(item.p2[0].x, item.p2[0].y);
      for (let i = 1; i < item.p2.length; i++) ctx.lineTo(item.p2[i].x, item.p2[i].y);
      ctx.closePath();
      ctx.fillStyle = item.color;
      ctx.fill();
      ctx.strokeStyle = item.stroke;
      ctx.lineWidth = item.lineWidth;
      ctx.stroke();
    }
  });

  // 3. Arm labels in 3D space
  ctx.font = 'bold 11px Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  armSpecs.forEach(a => {
    const pt = project(a.labelPos);
    const active = activeArms.has(a.name);
    ctx.fillStyle = active ? '#38bdf8' : '#94a3b8';
    ctx.fillText(a.label, pt.x, pt.y);
  });

  // Open camera axis indicator (FRONT/BACK open axis has no arm -> explains 3-step F/B moves)
  if (showFrontAxisTag) {
    const camPt = project([0, 0, 2.75]);
    ctx.fillStyle = '#c084fc';
    ctx.fillText('📷 OPEN FRONT AXIS', camPt.x, camPt.y);
  } else {
    const backPt = project([0, 0, -2.75]);
    ctx.fillStyle = '#c084fc';
    ctx.fillText('OPEN BACK AXIS (No Arm)', backPt.x, backPt.y);
  }
}

window.addEventListener('DOMContentLoaded', initApp);
