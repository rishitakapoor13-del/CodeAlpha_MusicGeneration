// Page for the AI music generator (CodeAlpha AI Internship, Task 3)
const $ = (id) => document.getElementById(id);
const canvas = $("roll");
const ctx = canvas.getContext("2d");
const player = $("player");
const generateBtn = $("generateBtn");

let events = [];
let total = 0;
let animating = false;

/* ---------- sliders ---------- */
function bindSlider(id, outId, format) {
  const input = $(id);
  const update = () => { $(outId).textContent = format(input.value); };
  input.addEventListener("input", update);
  update();
}
bindSlider("length", "lengthOut", (v) => v + " notes");
bindSlider("temperature", "tempOut", (v) => Number(v).toFixed(1));
bindSlider("tempo", "tempoOut", (v) => v + " BPM");

/* ---------- piano roll ---------- */
function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawRoll();
}

function drawRoll() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!events.length) return;

  const pitches = events.flatMap((e) => e.pitches);
  if (!pitches.length) return;
  const lo = Math.min(...pitches) - 1;
  const hi = Math.max(...pitches) + 1;
  const rows = hi - lo + 1;
  const rowH = h / rows;

  // faint guide lines for every C
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let p = lo; p <= hi; p++) {
    if (p % 12 === 0) {
      const y = h - (p - lo + 0.5) * rowH;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }

  const now = player.currentTime || 0;
  events.forEach((e) => {
    const x = (e.start / total) * w;
    const width = Math.max((e.dur / total) * w - 1.5, 2);
    const active = !player.paused && now >= e.start && now < e.start + e.dur;
    const top = Math.max(...e.pitches);
    e.pitches.forEach((p) => {
      const y = h - (p - lo + 1) * rowH;
      const isMelody = p === top;
      ctx.fillStyle = active ? "#ffffff" : isMelody ? "#f4b942" : "#6fb7c4";
      ctx.fillRect(x, y + 0.5, width, Math.max(rowH - 1, 2));
    });
  });

  if (now > 0) {   // playhead
    const x = (now / total) * w;
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillRect(x, 0, 2, h);
  }
}

function animate() {
  drawRoll();
  if (!player.paused && !player.ended) requestAnimationFrame(animate);
  else animating = false;
}
player.addEventListener("play", () => { if (!animating) { animating = true; animate(); } });
player.addEventListener("pause", drawRoll);
player.addEventListener("seeked", drawRoll);
player.addEventListener("ended", drawRoll);
window.addEventListener("resize", resizeCanvas);

/* ---------- generate ---------- */
async function generate() {
  generateBtn.disabled = true;
  generateBtn.textContent = "Composing...";
  $("status").textContent = "The model is writing a new piece...";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        length: Number($("length").value),
        temperature: Number($("temperature").value),
        tempo: Number($("tempo").value)
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");

    events = data.events;
    total = data.seconds;
    $("placeholder").classList.add("hidden");
    player.src = data.wav;
    $("midiLink").href = data.midi;
    $("wavLink").href = data.wav;
    $("midiLink").classList.remove("hidden");
    $("wavLink").classList.remove("hidden");
    $("status").textContent = events.length + " notes, " + Math.round(total) + " seconds";
    resizeCanvas();
    player.play().catch(() => { /* the browser may block autoplay; the play button still works */ });
  } catch (err) {
    $("status").textContent = err instanceof TypeError ? "Cannot reach the server. Is app.py running?" : err.message;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Compose music";
  }
}
generateBtn.addEventListener("click", generate);

/* ---------- model facts ---------- */
async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    const facts = [
      [info.pieces, "pieces of music studied"],
      [Number(info.events).toLocaleString(), "musical events"],
      [Number(info.parameters).toLocaleString(), "model parameters"],
      [info.melody_accuracy + "%", "melody prediction accuracy"]
    ];
    facts.forEach(([value, label]) => {
      const box = document.createElement("div");
      box.className = "fact";
      const b = document.createElement("b");
      b.textContent = value;
      const s = document.createElement("span");
      s.textContent = label;
      box.append(b, s);
      $("facts").appendChild(box);
    });
  } catch { /* info is optional */ }
}

resizeCanvas();
loadInfo();
