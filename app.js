/* GazeLab — pure front-end eye tracking with WebGazer
 * Features:
 *  - Webcam preview (mirrored)
 *  - Gaze cursor + heatmap
 *  - Simple calibration (user-guided samples & linear map)
 *  - Recording to CSV (timestamp,x,y,confidence,blinkFlag,headYaw,headPitch)
 *  - HUD (FPS, confidence, mode, rec)
 *  - Controls + keyboard shortcuts
 * Privacy: runs entirely in-browser. No uploads.
 */

/* -------------------- DOM -------------------- */
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const heatmap = document.getElementById('heatmap');
const landmarksCanvas = document.getElementById('landmarks');

const hudFPS = document.getElementById('hud-fps');
const hudConf = document.getElementById('hud-conf');
const hudMode = document.getElementById('hud-mode');
const hudRec = document.getElementById('hud-rec');
const lostBadge = document.getElementById('lost');

const permitCard = document.getElementById('permit');
const btnStart = document.getElementById('btn-start');
const btnRetry = document.getElementById('btn-retry');
const btnTroubleshoot = document.getElementById('btn-troubleshoot');
const permHelp = document.getElementById('perm-help');

const toggleLandmarks = document.getElementById('toggle-landmarks');
const toggleCursor = document.getElementById('toggle-cursor');
const toggleHeatmap = document.getElementById('toggle-heatmap');
const toggleTheme = document.getElementById('toggle-theme');

const smoothingSlider = document.getElementById('smoothing');
const dotSizeSlider = document.getElementById('dot-size');
const heatDecaySlider = document.getElementById('heat-decay');

const btnCalibrate = document.getElementById('btn-calibrate');
const btnCalibClear = document.getElementById('btn-calibrate-clear');

const btnRecord = document.getElementById('btn-record');
const btnExport = document.getElementById('btn-export');
const btnReset = document.getElementById('btn-reset');

const calibUI = document.getElementById('calib');
const calibDot = document.getElementById('calib-dot');
const calibExit = document.getElementById('calib-exit');
const calibCount = document.getElementById('calib-count');
const calibTotal = document.getElementById('calib-total');

/* -------------------- State -------------------- */
const ctx = overlay.getContext('2d');
const htx = heatmap.getContext('2d');
const ltx = landmarksCanvas.getContext('2d');

let camStream = null;
let running = false;
let showCursor = true;
let showHeat = false;
let showLandmarks = false;
let isRecording = false;

let emaSmoothing = parseFloat(smoothingSlider.value); // 0..1
let dotSize = parseInt(dotSizeSlider.value, 10);
let heatDecay = parseFloat(heatDecaySlider.value);

// EMA state
let filtX = null, filtY = null;

// FPS calc
let lastFrameTs = performance.now();
let fps = 0;

// Calibration
const DEFAULT_POINTS = [
  {x: 0.1, y: 0.1}, {x: 0.5, y: 0.1}, {x: 0.9, y: 0.1},
  {x: 0.1, y: 0.5}, {x: 0.5, y: 0.5}, {x: 0.9, y: 0.5},
  {x: 0.1, y: 0.9}, {x: 0.5, y: 0.9}, {x: 0.9, y: 0.9}
];
let calibPoints = [];
let calibIndex = 0;
let samples = []; // each: {features:[fx,fy], screen:[sx,sy]}
let mapper = null; // simple linear mapping params

// Recorder buffer
let logRows = [];
let logTimer = null;
const LOG_HZ = 20; // target sampling rate

// Heatmap fade loop
let heatAnim = null;

// Persistence
const LS_KEY = 'gazelab_calibration_v1';

/* -------------------- Utilities -------------------- */
function clamp(v, min, max){ return Math.min(max, Math.max(min, v)); }
function lerp(a,b,t){ return a + (b - a)*t; }
function fmt(n, d=2){ return Number.isFinite(n) ? n.toFixed(d) : '0.00'; }

function download(filename, text){
  const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* -------------------- Video / Canvas sizing -------------------- */
function resizeCanvases(){
  const rect = video.getBoundingClientRect();
  [overlay, heatmap, landmarksCanvas].forEach(c=>{
    c.width = Math.floor(rect.width);
    c.height = Math.floor(rect.height);
  });
}
window.addEventListener('resize', resizeCanvases);

/* -------------------- Camera -------------------- */
async function startCamera(){
  try{
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user' }, audio:false });
    video.srcObject = camStream;
    permitCard.style.display = 'none';
    running = true;
    resizeCanvases();
  }catch(err){
    console.warn('Camera error', err);
    permHelp.hidden = false;
  }
}
btnStart.onclick = startCamera;
btnRetry.onclick = startCamera;
btnTroubleshoot.onclick = () => {
  alert(`Troubleshooting tips:
- Ensure your browser (Chrome) has camera permission for this site.
- If previously denied, click the camera icon in the address bar and allow.
- Close any other app using your camera.
- Some enterprise devices block camera via policy.`);
};

/* -------------------- WebGazer init -------------------- */
async function initWebGazer(){
  // Configure webgazer
  webgazer
    .setRegression('ridge') // robust linear model
    .setTracker('clmtrackr') // face tracker
    .setGazeListener(onGaze)
    .showVideoPreview(false) // we supply our own <video>
    .showPredictionPoints(false)
    .showFaceOverlay(false)
    .showFaceFeedbackBox(false);

  await webgazer.begin();
  // Try to bind to our <video> element so permissions are unified
  if (video && video.srcObject) {
    // WebGazer manages its own stream; that's okay.
  }
}

/* -------------------- Gaze Processing -------------------- */
let latest = { x:null, y:null, conf:0, blink:false };
function onGaze(data, elapsedMs){
  // data: {x, y} in page coordinates; confidence not provided -> estimate
  if(!data){
    latest.blink = true;
    latest.conf = 0;
    return;
  }
  const rect = overlay.getBoundingClientRect();
  // Convert page coords to canvas coords
  const cx = clamp(data.x - rect.left, 0, rect.width);
  const cy = clamp(data.y - rect.top,  0, rect.height);

  // Simple confidence proxy: is point within viewport & face tracker state
  let conf = 1.0;
  if (data.x < rect.left || data.x > rect.right || data.y < rect.top || data.y > rect.bottom) conf = 0.2;

  // Apply calibration mapper if available
  const feat = normalizeFeatures(cx, cy, rect);
  let mapped = {x: cx, y: cy};
  if (mapper){
    mapped = applyMapper(mapper, feat, rect);
  }

  // EMA smoothing
  if (filtX == null){ filtX = mapped.x; filtY = mapped.y; }
  filtX = lerp(filtX, mapped.x, 1 - emaSmoothing);
  filtY = lerp(filtY, mapped.y, 1 - emaSmoothing);

  latest.x = filtX; latest.y = filtY; latest.conf = conf; latest.blink = conf < 0.35;
}

/* -------------------- Feature normalization & mapping -------------------- */
/* We do a simple linear mapping from normalized features -> screen coords.
 * Features: [fx, fy] where fx,fy are 0..1 within overlay rect.
 */
function normalizeFeatures(cx, cy, rect){
  return [ cx / rect.width, cy / rect.height ];
}
function fitLinearMapping(pairs){
  // Fit two independent linear models:
  // sx = ax*fx + bx*fy + cx
  // sy = ay*fx + by*fy + cy
  const X = [], Yx = [], Yy = [];
  for(const p of pairs){
    const [fx, fy] = p.features;
    X.push([fx, fy, 1]);
    Yx.push(p.screen[0]);
    Yy.push(p.screen[1]);
  }
  const Xt = mathTranspose(X);
  const XtX = mathMul(Xt, X);
  const XtXinv = mathInv3(XtX);
  const XtYx = mathMulVec(Xt, Yx);
  const XtYy = mathMulVec(Xt, Yy);
  const bx = mathMulMatVec(XtXinv, XtYx); // [ax,bx,cx]
  const by = mathMulMatVec(XtXinv, XtYy); // [ay,by,cy]
  return { bx, by }; // mapper
}
function applyMapper(mapper, features, rect){
  const [fx, fy] = features;
  const sx = mapper.bx[0]*fx + mapper.bx[1]*fy + mapper.bx[2];
  const sy = mapper.by[0]*fx + mapper.by[1]*fy + mapper.by[2];
  return {
    x: clamp(sx * rect.width, 0, rect.width),
    y: clamp(sy * rect.height, 0, rect.height)
  };
}

/* Minimal linear algebra helpers (3x3 inverse via adjugate) */
function mathTranspose(A){
  const m = A.length, n = A[0].length, T = Array.from({length:n}, ()=>Array(m).fill(0));
  for(let i=0;i<m;i++) for(let j=0;j<n;j++) T[j][i]=A[i][j];
  return T;
}
function mathMul(A,B){
  const m=A.length,n=A[0].length,p=B[0].length; const C=Array.from({length:m},()=>Array(p).fill(0));
  for(let i=0;i<m;i++) for(let k=0;k<p;k++) for(let j=0;j<n;j++) C[i][k]+=A[i][j]*B[j][k];
  return C;
}
function mathMulVec(A,v){
  const m=A.length,n=A[0].length; const out=Array(m).fill(0);
  for(let i=0;i<m;i++) for(let j=0;j<n;j++) out[i]+=A[i][j]*v[j];
  return out;
}
function mathInv3(M){
  // 3x3 inverse
  const [a,b,c] = M[0], [d,e,f] = M[1], [g,h,i] = M[2];
  const A =  (e*i - f*h), B = -(d*i - f*g), C =  (d*h - e*g);
  const D = -(b*i - c*h), E =  (a*i - c*g), F = -(a*h - b*g);
  const G =  (b*f - c*e), H = -(a*f - c*d), I =  (a*e - b*d);
  const det = a*A + b*B + c*C;
  if (Math.abs(det) < 1e-8) return [[1,0,0],[0,1,0],[0,0,1]];
  const invDet = 1/det;
  return [
    [A*invDet, D*invDet, G*invDet],
    [B*invDet, E*invDet, H*invDet],
    [C*invDet, F*invDet, I*invDet],
  ];
}

/* -------------------- Rendering -------------------- */
function drawLoop(){
  if (!running){ requestAnimationFrame(drawLoop); return; }

  // FPS
  const now = performance.now();
  const dt = now - lastFrameTs;
  if (dt > 0) fps = 1000 / dt;
  lastFrameTs = now;
  hudFPS.textContent = Math.round(fps);

  // Clear overlay
  ctx.clearRect(0,0,overlay.width,overlay.height);

  // Landmarks (placeholder visualization, using webgazer face feedback boxes not exposed)
  if (showLandmarks){
    // WebGazer doesn't give raw landmarks here; we can draw a gentle border to indicate "on"
    ltx.clearRect(0,0,landmarksCanvas.width,landmarksCanvas.height);
    ltx.strokeStyle = 'rgba(122,162,247,.6)';
    ltx.lineWidth = 2;
    ltx.strokeRect(8,8,landmarksCanvas.width-16,landmarksCanvas.height-16);
  }else{
    ltx.clearRect(0,0,landmarksCanvas.width,landmarksCanvas.height);
  }

  // Cursor & heatmap
  const conf = latest.conf || 0;
  hudConf.textContent = fmt(conf,2);
  lostBadge.hidden = conf >= 0.35;

  if (showHeat && Number.isFinite(latest.x) && Number.isFinite(latest.y)){
    // Add a splash to heatmap
    const r = Math.max(10, dotSize*2.0);
    const grad = htx.createRadialGradient(latest.x, latest.y, 0, latest.x, latest.y, r);
    grad.addColorStop(0, 'rgba(122,162,247,0.45)');
    grad.addColorStop(1, 'rgba(122,162,247,0)');
    htx.fillStyle = grad;
    htx.beginPath();
    htx.arc(latest.x, latest.y, r, 0, Math.PI*2);
    htx.fill();
  }

  if (showCursor && Number.isFinite(latest.x) && Number.isFinite(latest.y)){
    const col = conf >= 0.6 ? '#34d399' : (conf >= 0.35 ? '#f59e0b' : '#f87171');
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(12,14,20,0.5)';
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(latest.x, latest.y, dotSize, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }

  requestAnimationFrame(drawLoop);
}

/* Heatmap decay */
function startHeatFade(){
  if (heatAnim) cancelAnimationFrame(heatAnim);
  const step = () => {
    htx.fillStyle = `rgba(0,0,0,${1 - heatDecay})`;
    htx.globalCompositeOperation = 'destination-in';
    htx.fillRect(0,0,heatmap.width,heatmap.height);
    htx.globalCompositeOperation = 'source-over';
    heatAnim = requestAnimationFrame(step);
  };
  heatAnim = requestAnimationFrame(step);
}

/* -------------------- Calibration -------------------- */
function loadCalibration(){
  const raw = localStorage.getItem(LS_KEY);
  if (!raw) return;
  try{
    const data = JSON.parse(raw);
    if (data && data.mapper) mapper = data.mapper;
  }catch{}
}
function saveCalibration(){
  localStorage.setItem(LS_KEY, JSON.stringify({ mapper }));
}
function clearCalibration(){
  mapper = null;
  localStorage.removeItem(LS_KEY);
}

function startCalibration(points = DEFAULT_POINTS){
  calibPoints = points.slice(); // copy
  calibIndex = 0;
  samples = [];
  calibTotal.textContent = String(calibPoints.length);
  calibCount.textContent = '0';
  hudMode.textContent = 'Calibration';
  calibUI.classList.remove('hidden');
  calibUI.setAttribute('aria-hidden','false');
  placeCalibDot();
}
function placeCalibDot(){
  const rect = overlay.getBoundingClientRect();
  const p = calibPoints[calibIndex];
  const x = p.x * rect.width;
  const y = p.y * rect.height;
  calibDot.style.left = `${x}px`;
  calibDot.style.top = `${y}px`;
}
function captureCalibSample(){
  if (!calibPoints.length) return;
  const rect = overlay.getBoundingClientRect();
  const p = calibPoints[calibIndex];
  // Get current gaze as feature (normalized)
  if (latest.x == null || latest.y == null) return;
  const feat = normalizeFeatures(latest.x, latest.y, rect);
  samples.push({ features: feat, screen: [p.x, p.y] }); // screen normalized 0..1
  // Advance
  calibIndex++;
  calibCount.textContent = String(Math.min(calibIndex, calibPoints.length));
  if (calibIndex >= calibPoints.length){
    // Fit mapper
    if (samples.length >= 3){
      mapper = fitLinearMapping(samples);
      saveCalibration();
    }
    exitCalibration();
  }else{
    placeCalibDot();
  }
}
function exitCalibration(){
  calibUI.classList.add('hidden');
  calibUI.setAttribute('aria-hidden','true');
  hudMode.textContent = showHeat ? 'Heatmap' : (showCursor ? 'Cursor' : 'Idle');
}

/* -------------------- Recorder -------------------- */
function startRecording(){
  if (isRecording) return;
  isRecording = true;
  hudRec.textContent = '● ON';
  hudRec.style.color = '#34d399';
  logRows = [];
  logTimer = setInterval(()=>{
    const ts = Date.now();
    const x = Number.isFinite(latest.x) ? latest.x : '';
    const y = Number.isFinite(latest.y) ? latest.y : '';
    const conf = latest.conf ?? 0;
    const blink = latest.blink ? 1 : 0;
    // Head pose placeholders (not available from WebGazer):
    const headYaw = '';   // could be provided if integrating MediaPipe FaceLandmarker
    const headPitch = '';
    logRows.push([ts, x, y, conf, blink, headYaw, headPitch].join(','));
  }, 1000/LOG_HZ);
}
function stopRecording(){
  if (!isRecording) return;
  isRecording = false;
  hudRec.textContent = '● OFF';
  hudRec.style.color = '#f87171';
  clearInterval(logTimer);
  logTimer = null;
}
function exportCSV(){
  const header = 'timestamp,x,y,confidence,blinkFlag,headYaw,headPitch';
  const csv = [header, ...logRows].join('\n');
  download(`gazelab_${new Date().toISOString().replace(/[:.]/g,'-')}.csv`, csv);
}

/* -------------------- Events & UI -------------------- */
toggleLandmarks.onchange = () => { showLandmarks = toggleLandmarks.checked; landmarksCanvas.hidden = !showLandmarks; };
toggleCursor.onchange = () => { showCursor = toggleCursor.checked; hudMode.textContent = showHeat ? 'Heatmap' : (showCursor ? 'Cursor' : 'Idle'); };
toggleHeatmap.onchange = () => { showHeat = toggleHeatmap.checked; hudMode.textContent = showHeat ? 'Heatmap' : (showCursor ? 'Cursor' : 'Idle'); };

smoothingSlider.oninput = () => { emaSmoothing = parseFloat(smoothingSlider.value); };
dotSizeSlider.oninput = () => { dotSize = parseInt(dotSizeSlider.value,10); };
heatDecaySlider.oninput = () => { heatDecay = parseFloat(heatDecaySlider.value); };

btnCalibrate.onclick = () => startCalibration(DEFAULT_POINTS);
btnCalibClear.onclick = () => { clearCalibration(); alert('Calibration cleared.'); };

btnRecord.onclick = () => {
  if (isRecording) { stopRecording(); btnRecord.textContent = 'Start Recording'; }
  else { startRecording(); btnRecord.textContent = 'Stop Recording'; }
};
btnExport.onclick = exportCSV;

btnReset.onclick = () => {
  toggleLandmarks.checked = false; toggleLandmarks.onchange();
  toggleCursor.checked = true; toggleCursor.onchange();
  toggleHeatmap.checked = false; toggleHeatmap.onchange();
  smoothingSlider.value = '0.35'; smoothingSlider.oninput();
  dotSizeSlider.value = '12'; dotSizeSlider.oninput();
  heatDecaySlider.value = '0.965'; heatDecaySlider.oninput();
  clearCalibration();
  htx.clearRect(0,0,heatmap.width,heatmap.height);
};

toggleTheme.onchange = () => {
  if (toggleTheme.checked) document.body.classList.add('light');
  else document.body.classList.remove('light');
};

// Keyboard shortcuts
window.addEventListener('keydown', (e)=>{
  if (e.repeat) return;
  if (e.key.toLowerCase() === 'c'){ startCalibration(DEFAULT_POINTS); }
  if (e.key.toLowerCase() === 'r'){ btnRecord.click(); }
  if (e.key.toLowerCase() === 'h'){ toggleHeatmap.checked = !toggleHeatmap.checked; toggleHeatmap.onchange(); }
  if (e.key.toLowerCase() === 'l'){ toggleLandmarks.checked = !toggleLandmarks.checked; toggleLandmarks.onchange(); }
  if (e.key.toLowerCase() === 'd'){ toggleTheme.checked = !toggleTheme.checked; toggleTheme.onchange(); }
  if (calibUI && !calibUI.classList.contains('hidden') && e.code === 'Space'){
    e.preventDefault(); captureCalibSample();
  }
});

// Start loops after DOM ready & permissions
window.addEventListener('load', async ()=>{
  loadCalibration();
  // If permissions are already granted, auto-start
  try{
    const st = await navigator.permissions.query({ name:'camera' });
    if (st.state === 'granted') startCamera();
    st.onchange = () => {
      if (st.state === 'granted') startCamera();
      if (st.state === 'denied'){ permitCard.style.display = 'block'; permHelp.hidden = false; }
    };
  }catch{ /* some browsers don't expose permissions API */ }

  // Init webgazer once page is ready
  await initWebGazer();

  // Sizing
  resizeCanvases();
  drawLoop();
  startHeatFade();
});

/* -------------------- Notes --------------------
 * - Head pose (yaw/pitch) and eyelid-based blink detection would require a face landmark model (e.g., MediaPipe Face Landmarker).
 *   To keep things lightweight and 100% local with no build step, this version logs placeholders for yaw/pitch.
 * - If you want actual head pose & blink metrics, I can provide an optional MediaPipe integration that runs alongside WebGazer.
 */
