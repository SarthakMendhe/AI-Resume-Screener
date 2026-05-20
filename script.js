/* AI Resume Screening — 100% client-side
 * Extracts text from PDF/DOCX/TXT, then ranks with TF-IDF + cosine similarity.
 */

const STOPWORDS = new Set("a about above after again against all am an and any are aren't as at be because been before being below between both but by can cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves".split(" "));

const $ = (id) => document.getElementById(id);
const files = []; // {name, text}

const jobDesc = $("jobDesc");
const fileInput = $("fileInput");
const dropzone = $("dropzone");
const fileList = $("fileList");
const analyzeBtn = $("analyzeBtn");
const resetBtn = $("resetBtn");
const resultsSection = $("results");
const resultsList = $("resultsList");
const toast = $("toast");

/* ---------- UI helpers ---------- */
function showToast(msg) {
  toast.textContent = msg;
  toast.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => (toast.hidden = true), 3500);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderFileList() {
  fileList.innerHTML = files.map((f, i) =>
    `<li><span>📎 ${escapeHtml(f.name)} <small style="color:var(--muted)">(${(f.text.length/1000).toFixed(1)}k chars)</small></span>
     <button class="remove" data-i="${i}" title="Remove">×</button></li>`
  ).join("");
  fileList.querySelectorAll(".remove").forEach((b) =>
    b.addEventListener("click", () => { files.splice(+b.dataset.i, 1); renderFileList(); })
  );
}

/* ---------- File parsing ---------- */
async function extractText(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".txt")) return await file.text();
  if (name.endsWith(".pdf")) return await extractPdf(file);
  if (name.endsWith(".docx")) return await extractDocx(file);
  throw new Error(`Unsupported file: ${file.name}`);
}

async function extractPdf(file) {
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  let text = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map((it) => it.str).join(" ") + "\n";
  }
  return text;
}

async function extractDocx(file) {
  const buf = await file.arrayBuffer();
  const { value } = await mammoth.extractRawText({ arrayBuffer: buf });
  return value;
}

async function handleFiles(fileListIn) {
  for (const file of fileListIn) {
    try {
      const text = await extractText(file);
      if (!text.trim()) { showToast(`Empty file: ${file.name}`); continue; }
      files.push({ name: file.name, text });
    } catch (e) {
      console.error(e);
      showToast(`Could not read ${file.name}`);
    }
  }
  renderFileList();
}

/* ---------- NLP ---------- */
function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s+#./-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

function withBigrams(tokens) {
  const out = tokens.slice();
  for (let i = 0; i < tokens.length - 1; i++) out.push(tokens[i] + "_" + tokens[i+1]);
  return out;
}

function termFreq(tokens) {
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
  return tf;
}

function tfidfVectors(docsTokens) {
  const N = docsTokens.length;
  const df = new Map();
  const tfs = docsTokens.map(termFreq);
  for (const tf of tfs) for (const term of tf.keys()) df.set(term, (df.get(term) || 0) + 1);
  const idf = new Map();
  for (const [t, d] of df) idf.set(t, Math.log((N + 1) / (d + 1)) + 1);
  return tfs.map((tf) => {
    const v = new Map();
    for (const [t, c] of tf) v.set(t, c * (idf.get(t) || 0));
    return v;
  });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (const [, v] of a) na += v * v;
  for (const [, v] of b) nb += v * v;
  const [small, big] = a.size < b.size ? [a, b] : [b, a];
  for (const [t, v] of small) if (big.has(t)) dot += v * big.get(t);
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function extractContacts(text) {
  const email = (text.match(/[\w.+-]+@[\w-]+\.[\w.-]+/) || [])[0] || "";
  const phone = (text.match(/(\+?\d[\d\s().-]{8,}\d)/) || [])[0] || "";
  return { email, phone };
}

function topKeywords(tokens, n = 20) {
  const tf = termFreq(tokens);
  return [...tf.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([k]) => k);
}

/* ---------- Scoring ---------- */
function score() {
  const jd = jobDesc.value.trim();
  if (!jd) { showToast("Please enter a job description."); return null; }
  if (files.length === 0) { showToast("Please upload at least one resume."); return null; }

  const jdTokens = withBigrams(tokenize(jd));
  const docs = [jdTokens, ...files.map((f) => withBigrams(tokenize(f.text)))];
  const vecs = tfidfVectors(docs);
  const jdVec = vecs[0];

  const jdKeywords = new Set(topKeywords(tokenize(jd), 25));

  return files.map((f, i) => {
    const tokens = tokenize(f.text);
    const resumeWords = new Set(tokens);
    const matched = [...jdKeywords].filter((k) => resumeWords.has(k));
    const missing = [...jdKeywords].filter((k) => !resumeWords.has(k));
    const sim = cosine(jdVec, vecs[i + 1]);
    const { email, phone } = extractContacts(f.text);
    return {
      name: f.name,
      score: Math.round(sim * 10000) / 100,
      matched, missing, email, phone,
    };
  }).sort((a, b) => b.score - a.score);
}

function rankClass(i) { return i === 0 ? "gold" : i === 1 ? "silver" : i === 2 ? "bronze" : ""; }

function renderResults(results) {
  resultsList.innerHTML = results.map((r, i) => `
    <article class="result-card">
      <div class="result-header">
        <div class="rank-name">
          <div class="rank ${rankClass(i)}">${i + 1}</div>
          <div>
            <div class="fname">${escapeHtml(r.name)}</div>
            <div class="contact">
              ${r.email ? `<span>✉ ${escapeHtml(r.email)}</span>` : ""}
              ${r.phone ? `<span>☎ ${escapeHtml(r.phone)}</span>` : ""}
            </div>
          </div>
        </div>
        <div class="score-big">${r.score.toFixed(1)}%</div>
      </div>
      <div class="progress"><div style="width:${r.score}%"></div></div>
      <div class="kw-section">
        <h4>✅ Matched keywords (${r.matched.length})</h4>
        <div class="chips">${r.matched.slice(0, 30).map((k) => `<span class="chip matched">${escapeHtml(k)}</span>`).join("") || '<span class="chip">none</span>'}</div>
      </div>
      <div class="kw-section">
        <h4>❌ Missing keywords (${r.missing.length})</h4>
        <div class="chips">${r.missing.slice(0, 20).map((k) => `<span class="chip missing">${escapeHtml(k)}</span>`).join("") || '<span class="chip">none</span>'}</div>
      </div>
    </article>
  `).join("");
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- Events ---------- */
fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

analyzeBtn.addEventListener("click", async () => {
  const label = analyzeBtn.querySelector(".btn-label");
  const spinner = analyzeBtn.querySelector(".spinner");
  analyzeBtn.disabled = true; spinner.hidden = false; label.textContent = "Analyzing...";
  await new Promise((r) => setTimeout(r, 50));
  try {
    const results = score();
    if (results) renderResults(results);
  } catch (e) {
    console.error(e); showToast("Something went wrong while analyzing.");
  } finally {
    analyzeBtn.disabled = false; spinner.hidden = true; label.textContent = "🚀 Analyze & Rank";
  }
});

resetBtn.addEventListener("click", () => {
  files.length = 0;
  jobDesc.value = "";
  fileInput.value = "";
  renderFileList();
  resultsSection.hidden = true;
  resultsList.innerHTML = "";
});
