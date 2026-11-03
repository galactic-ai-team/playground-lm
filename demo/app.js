// Demo wiring (Lumi). Reads manifest.json; model inference hooks in once checkpoints exist.
const $ = (id) => document.getElementById(id);

async function loadManifest() {
  const res = await fetch("manifest.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest.json ${res.status}`);
  return res.json();
}

function renderCheckpoints(ckpts) {
  const slider = $("slider");
  if (!ckpts.length) return;
  slider.max = String(ckpts.length - 1);
  slider.value = slider.max;
  slider.disabled = false;
  const show = () => {
    const c = ckpts[Number(slider.value)];
    $("ckpt-label").textContent = `step ${c.step} (${c.tokens_seen.toLocaleString()} tokens)`;
  };
  slider.addEventListener("input", show);
  show();
}

function renderBenchmarks(b) {
  const thead = $("bench-table").tHead;
  const tbody = $("bench-table").tBodies[0];
  if (!b.rows.length) return;
  $("bench-empty").hidden = true;
  thead.innerHTML = "<tr><th>model</th>" + b.columns.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  tbody.innerHTML = b.rows
    .map((r) => "<tr><td>" + r.model + "</td>" + b.columns.map((c) => `<td>${r[c] ?? "pending"}</td>`).join("") + "</tr>")
    .join("");
}

function wireRiddles(prompts) {
  if (!prompts.length) return;
  const btn = $("riddle");
  btn.disabled = false;
  btn.addEventListener("click", () => {
    $("prompt-input").value = prompts[Math.floor(Math.random() * prompts.length)];
  });
}

loadManifest()
  .then((m) => {
    renderCheckpoints(m.checkpoints);
    renderBenchmarks(m.benchmarks);
    wireRiddles(m.riddle_prompts);
  })
  .catch((e) => { $("output").textContent = `Could not load the manifest: ${e.message}`; });
