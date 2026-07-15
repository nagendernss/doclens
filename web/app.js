// doclens frontend — vanilla JS, no build step, no CDN.
//
// Security invariant (hard gate): every dynamic value that ever reaches `.innerHTML`
// is passed through esc() (text) or escAttr() (HTML attribute values). There is
// exactly one call site that uses innerHTML at all — inside renderCitedAnswer(),
// which emits clickable `<button class="pill-cite" data-page="N">` citations —
// everything else (turn bubbles, retrieval/source cards, doc switcher, banners) is
// built with textContent / createElement / setAttribute / style properties, which
// never parse their input as HTML and so need no escaping at all.
(function () {
  "use strict";

  const LS_KEY = "doclens.docs.v1";
  const MAX_DOCS = 20;
  const CONVOS_LS_KEY = "doclens.convos.v1";
  const MAX_TURNS_PER_DOC = 40;
  const FLASH_MS = 1200;
  const STAGES = ["fetch", "parse", "chunk", "embed"];
  const STAGE_LABELS = { fetch: "fetching", parse: "parsing", chunk: "chunking", embed: "embedding" };
  const CITATION_RE = /\[p\.(\d+)\]/g;

  // ---------------------------------------------------------------------
  // escaping helpers (repolens-style) — required even though only one
  // renderer needs them; every other renderer below uses textContent/DOM
  // properties instead, which are inherently injection-safe.
  // ---------------------------------------------------------------------

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escAttr(value) {
    return esc(value)
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ---------------------------------------------------------------------
  // dom cache
  // ---------------------------------------------------------------------

  const dom = {};

  function cacheDom() {
    dom.ingestForm = document.getElementById("ingest-form");
    dom.dropzone = document.getElementById("dropzone");
    dom.fileInput = document.getElementById("file-input");
    dom.dropzoneFilename = document.getElementById("dropzone-filename");
    dom.clearFileBtn = document.getElementById("clear-file-btn");
    dom.urlInput = document.getElementById("url-input");
    dom.modelSelect = document.getElementById("model-select");
    dom.modelHint = document.getElementById("model-hint");
    dom.byoKeyInput = document.getElementById("byo-key-input");
    dom.ingestBtn = document.getElementById("ingest-btn");
    dom.ingestProgress = document.getElementById("ingest-progress");
    dom.progressSteps = document.getElementById("progress-steps");
    dom.progressBarFill = document.getElementById("progress-bar-fill");
    dom.progressStatus = document.getElementById("progress-status");
    dom.ingestError = document.getElementById("ingest-error");

    dom.switcherSection = document.getElementById("doc-switcher-section");
    dom.docSwitcher = document.getElementById("doc-switcher");
    dom.clearDocsBtn = document.getElementById("clear-docs-btn");

    dom.askPanel = document.getElementById("ask-panel");
    dom.askTitle = document.getElementById("ask-title");
    dom.askExpiredBanner = document.getElementById("ask-expired-banner");
    dom.reingestBtn = document.getElementById("reingest-btn");
    dom.convoLog = document.getElementById("convo-log");
    dom.askForm = document.getElementById("ask-form");
    dom.questionInput = document.getElementById("question-input");
    dom.charCounter = document.getElementById("char-counter");
    dom.askBtn = document.getElementById("ask-btn");
    dom.askStatus = document.getElementById("ask-status");
    dom.askError = document.getElementById("ask-error");

    dom.progressStepEls = {};
    for (const li of dom.progressSteps.querySelectorAll("li[data-stage]")) {
      dom.progressStepEls[li.dataset.stage] = li;
    }
  }

  // ---------------------------------------------------------------------
  // state
  // ---------------------------------------------------------------------

  const state = {
    docs: [],
    selectedDocId: null,
    models: [],
    defaultModel: null,
    // doc_id -> [{q, answer, citations, refused, retrieval, ts}, ...] — the
    // per-doc follow-up conversation log; mirrored to localStorage (see
    // loadConvos/saveConvos below) so it survives reloads, capped per doc.
    convos: {},
    // doc_id -> {type:"url", value} | {type:"file"} — session-only, powers the
    // "re-ingest" convenience button; never persisted (a File can't be, and a
    // URL is just a convenience, not part of the localStorage.v1 contract).
    sourceByDocId: new Map(),
  };

  let selectedFile = null;
  let ingestInFlight = false;
  let askInFlight = false;

  // ---------------------------------------------------------------------
  // localStorage doc list — schema is exactly [{doc_id,title,pages,chunks}]
  // ---------------------------------------------------------------------

  function loadDocs() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((d) => d && typeof d.doc_id === "string");
    } catch {
      return [];
    }
  }

  function saveDocs(docs) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(docs.slice(0, MAX_DOCS)));
    } catch {
      // storage disabled/full — degrade to in-memory only for this page load
    }
  }

  function addDoc(doc) {
    state.docs = state.docs.filter((d) => d.doc_id !== doc.doc_id);
    state.docs.unshift(doc);
    if (state.docs.length > MAX_DOCS) state.docs.length = MAX_DOCS;
    saveDocs(state.docs);
  }

  // ---------------------------------------------------------------------
  // localStorage per-doc conversation log — doclens.convos.v1 =
  // {doc_id: [{q, answer, citations, refused, retrieval, trace, ts}, ...]},
  // capped at MAX_TURNS_PER_DOC turns/doc (oldest evicted first). `refused`
  // rides alongside the brief's {q,answer,citations,retrieval,ts} shape so a
  // reload/doc-switch still renders the "Not in the document" badge on old
  // refusal turns, not only on turns freshly answered this session. `trace`
  // ({trace_id, spans}) is additive — turns persisted before this feature
  // shipped simply carry `trace: null`, which renders no waterfall.
  // ---------------------------------------------------------------------

  function loadConvos() {
    try {
      const raw = localStorage.getItem(CONVOS_LS_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};

      const out = {};
      for (const [docId, turns] of Object.entries(parsed)) {
        if (typeof docId !== "string" || !Array.isArray(turns)) continue;
        out[docId] = turns
          .filter((t) => t && typeof t === "object" &&
                        typeof t.q === "string" && typeof t.answer === "string")
          .slice(-MAX_TURNS_PER_DOC)
          .map((t) => ({
            q: t.q,
            answer: t.answer,
            citations: Array.isArray(t.citations) ? t.citations : [],
            refused: t.refused === true,
            retrieval: Array.isArray(t.retrieval) ? t.retrieval : [],
            trace: (t.trace && typeof t.trace === "object") ? t.trace : null,
            ts: Number.isFinite(t.ts) ? t.ts : Date.now(),
          }));
      }
      return out;
    } catch {
      return {};
    }
  }

  function saveConvos() {
    try {
      localStorage.setItem(CONVOS_LS_KEY, JSON.stringify(state.convos));
    } catch {
      // storage disabled/full — degrade to in-memory only for this page load
    }
  }

  function clearConvos() {
    state.convos = {};
    saveConvos();
  }

  function appendTurn(docId, turn) {
    const turns = (state.convos[docId] || []).slice();
    turns.push(turn);
    while (turns.length > MAX_TURNS_PER_DOC) turns.shift(); // evict oldest
    state.convos[docId] = turns;
    saveConvos();
  }

  function historyForDoc(docId) {
    const turns = state.convos[docId] || [];
    return turns.map((t) => ({ question: t.q, answer: t.answer }));
  }

  // ---------------------------------------------------------------------
  // SSE consumption — mirrors server._sse(): "event: X\ndata: Y\n\n" blocks
  // ---------------------------------------------------------------------

  async function consumeSSE(resp, handlers) {
    if (!resp.ok) {
      (handlers.error || function () {})({ message: `server error (${resp.status})` });
      return;
    }
    if (!resp.body || !resp.body.getReader) {
      (handlers.error || function () {})({ message: "streaming is not supported in this browser" });
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        if (!block) continue;
        let event = null;
        let dataLine = null;
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) dataLine = line.slice(6);
        }
        if (!event || dataLine === null) continue;
        let data;
        try {
          data = JSON.parse(dataLine);
        } catch {
          continue;
        }
        const handler = handlers[event];
        if (handler) handler(data);
      }
    }
  }

  // ---------------------------------------------------------------------
  // models
  // ---------------------------------------------------------------------

  async function loadModels() {
    try {
      const resp = await fetch("/api/models", { credentials: "same-origin" });
      const data = await resp.json();
      state.models = Array.isArray(data.models) ? data.models : [];
      state.defaultModel = typeof data.default === "string" ? data.default : null;
    } catch {
      state.models = [];
      state.defaultModel = null;
    }

    dom.modelSelect.replaceChildren();
    if (!state.models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "no server model configured";
      dom.modelSelect.appendChild(opt);
      dom.modelHint.hidden = false;
    } else {
      for (const m of state.models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === state.defaultModel) opt.selected = true;
        dom.modelSelect.appendChild(opt);
      }
      dom.modelHint.hidden = true;
    }
  }

  // ---------------------------------------------------------------------
  // ingest: file / dropzone
  // ---------------------------------------------------------------------

  function isPdfFile(f) {
    return f.type === "application/pdf" || /\.pdf$/i.test(f.name || "");
  }

  function setSelectedFile(file) {
    selectedFile = file;
    if (file) {
      dom.dropzoneFilename.textContent = file.name;
      dom.dropzone.classList.add("has-file");
      dom.clearFileBtn.hidden = false;
      dom.urlInput.value = "";
      dom.urlInput.disabled = true;
    } else {
      dom.dropzoneFilename.textContent = "";
      dom.dropzone.classList.remove("has-file");
      dom.clearFileBtn.hidden = true;
      dom.urlInput.disabled = ingestInFlight;
    }
    refreshIngestButton();
  }

  function clearFileSelection() {
    setSelectedFile(null);
    dom.fileInput.value = "";
  }

  function refreshIngestButton() {
    dom.ingestBtn.disabled = ingestInFlight || (!selectedFile && !dom.urlInput.value.trim());
  }

  // ---------------------------------------------------------------------
  // ingest: progress rendering
  // ---------------------------------------------------------------------

  function resetProgress() {
    for (const stage of STAGES) {
      dom.progressStepEls[stage].classList.remove("active", "done");
    }
    dom.progressBarFill.style.width = "0%";
    dom.progressBarFill.classList.remove("indeterminate");
    dom.progressStatus.textContent = "starting…";
  }

  function updateProgress(data) {
    const stage = typeof data.stage === "string" ? data.stage : "";
    const done = Number.isFinite(data.done) ? data.done : 0;
    const total = Number.isFinite(data.total) ? data.total : 0;
    const idx = STAGES.indexOf(stage);

    STAGES.forEach((s, i) => {
      const el = dom.progressStepEls[s];
      el.classList.toggle("done", idx >= 0 && i < idx);
      el.classList.toggle("active", i === idx);
    });

    const label = STAGE_LABELS[stage] || stage || "working";
    if (total > 0) {
      const pct = Math.max(0, Math.min(100, Math.round((done / total) * 100)));
      dom.progressBarFill.classList.remove("indeterminate");
      dom.progressBarFill.style.width = pct + "%";
      dom.progressStatus.textContent = `${label}… ${done}/${total}`;
    } else {
      dom.progressBarFill.classList.add("indeterminate");
      dom.progressStatus.textContent = `${label}…`;
    }
  }

  // ---------------------------------------------------------------------
  // ingest: errors / busy state
  // ---------------------------------------------------------------------

  function showIngestError(msg) {
    dom.ingestError.textContent = msg;
    dom.ingestError.hidden = false;
  }

  function clearIngestError() {
    dom.ingestError.hidden = true;
    dom.ingestError.textContent = "";
  }

  function friendlyMessage(msg) {
    if (/daily limit|global/i.test(msg)) {
      return `${msg} — add your own API key above to bypass the daily cap, or try again tomorrow.`;
    }
    return msg;
  }

  function setIngestBusy(busy) {
    ingestInFlight = busy;
    dom.fileInput.disabled = busy;
    dom.byoKeyInput.disabled = busy;
    dom.modelSelect.disabled = busy;
    dom.urlInput.disabled = busy || !!selectedFile;
    dom.ingestBtn.textContent = busy ? "ingesting…" : "Ingest";
    refreshIngestButton();
  }

  // ---------------------------------------------------------------------
  // ingest: submit
  // ---------------------------------------------------------------------

  async function handleIngestSubmit(e) {
    e.preventDefault();
    if (ingestInFlight) return;
    clearIngestError();

    const file = selectedFile;
    const url = dom.urlInput.value.trim();
    if (!file && !url) {
      showIngestError("choose a PDF file or paste a URL first");
      return;
    }

    setIngestBusy(true);
    dom.ingestProgress.hidden = false;
    resetProgress();

    const byoKey = dom.byoKeyInput.value.trim();

    try {
      let resp;
      if (file) {
        const fd = new FormData();
        fd.append("file", file, file.name);
        if (byoKey) fd.append("byo_key", byoKey);
        resp = await fetch("/api/ingest", { method: "POST", body: fd, credentials: "same-origin" });
      } else {
        const payload = { url };
        if (byoKey) payload.byo_key = byoKey;
        resp = await fetch("/api/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify(payload),
        });
      }

      await consumeSSE(resp, {
        progress: updateProgress,
        ready: (data) => onIngestReady(data, file ? null : url),
        error: (data) => {
          const msg = typeof data.message === "string" ? data.message : "ingest failed — try again";
          showIngestError(friendlyMessage(msg));
        },
      });
    } catch {
      showIngestError("network error — check your connection and try again");
    } finally {
      setIngestBusy(false);
    }
  }

  function onIngestReady(data, sourceUrl) {
    const doc = {
      doc_id: String(data.doc_id),
      title: typeof data.title === "string" && data.title ? data.title : "untitled document",
      pages: Number.isFinite(data.pages) ? data.pages : 0,
      chunks: Number.isFinite(data.chunks) ? data.chunks : 0,
    };
    addDoc(doc);
    state.sourceByDocId.set(doc.doc_id, sourceUrl ? { type: "url", value: sourceUrl } : { type: "file" });
    dom.progressStatus.textContent = "done — ready to ask";

    clearFileSelection();
    dom.urlInput.value = "";

    renderDocSwitcher();
    selectDoc(doc.doc_id);
  }

  // ---------------------------------------------------------------------
  // document switcher
  // ---------------------------------------------------------------------

  function renderDocSwitcher() {
    dom.docSwitcher.replaceChildren();
    if (!state.docs.length) {
      dom.switcherSection.hidden = true;
      return;
    }
    dom.switcherSection.hidden = false;

    for (const doc of state.docs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "doc-chip" + (doc.doc_id === state.selectedDocId ? " active" : "");

      const title = document.createElement("span");
      title.className = "doc-chip-title";
      title.textContent = doc.title;

      const meta = document.createElement("span");
      meta.className = "doc-chip-meta";
      meta.textContent = `${doc.pages} pages · ${doc.chunks} chunks`;

      btn.append(title, meta);
      btn.addEventListener("click", () => {
        if (askInFlight) return;
        selectDoc(doc.doc_id);
      });
      dom.docSwitcher.appendChild(btn);
    }
  }

  function handleClearDocs() {
    state.docs = [];
    saveDocs(state.docs);
    clearConvos();
    state.selectedDocId = null;
    renderDocSwitcher();
    dom.askPanel.hidden = true;
    dom.convoLog.replaceChildren();
  }

  function selectDoc(docId) {
    const doc = state.docs.find((d) => d.doc_id === docId);
    if (!doc) return;

    state.selectedDocId = docId;
    renderDocSwitcher();

    dom.askPanel.hidden = false;
    dom.askTitle.textContent = `${doc.title} · ${doc.pages} pages · ${doc.chunks} chunks`;
    hideAskExpiredBanner();
    clearAskError();
    renderConvoLog(docId);
    hideAskStatus();
    dom.questionInput.value = "";
    dom.charCounter.textContent = "0/500";
    refreshAskButton();
    dom.askPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    dom.questionInput.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------------
  // ask: errors / status / busy state
  // ---------------------------------------------------------------------

  function showAskError(msg) {
    dom.askError.textContent = msg;
    dom.askError.hidden = false;
  }

  function clearAskError() {
    dom.askError.hidden = true;
    dom.askError.textContent = "";
  }

  function showAskStatus(msg) {
    dom.askStatus.textContent = msg;
    dom.askStatus.hidden = false;
  }

  function hideAskStatus() {
    dom.askStatus.hidden = true;
    dom.askStatus.textContent = "";
  }

  function showAskExpiredBanner() {
    dom.askExpiredBanner.hidden = false;
  }

  function hideAskExpiredBanner() {
    dom.askExpiredBanner.hidden = true;
  }

  function refreshAskButton() {
    dom.askBtn.disabled = askInFlight || dom.questionInput.value.trim().length === 0;
  }

  function setAskBusy(busy) {
    askInFlight = busy;
    dom.questionInput.disabled = busy;
    dom.askBtn.textContent = busy ? "asking…" : "Ask";
    refreshAskButton();
  }

  // ---------------------------------------------------------------------
  // ask: conversation log rendering
  //
  // buildChunkCard/buildTraceDetails/buildTurnEl are built with textContent/
  // createElement/setAttribute/style — none of these parse their input as
  // HTML, so nothing there needs esc()/escAttr(), even though
  // `preview`/`q`/`answer` are raw, untrusted document/model text (including
  // turns reloaded from localStorage, which a user could have hand-edited).
  // renderCitedAnswer is the one deliberate innerHTML site — see its own
  // comment below.
  // ---------------------------------------------------------------------

  function buildChunkCard(c) {
    const page = c && Number.isFinite(c.page) ? c.page : 0;
    const score = c && typeof c.score === "number" && Number.isFinite(c.score) ? c.score : 0;
    const preview = c && typeof c.preview === "string" ? c.preview : "";
    const pct = Math.max(0, Math.min(100, Math.round(score * 100)));

    const li = document.createElement("li");
    li.className = "chunk-card";
    li.dataset.page = String(page);

    const head = document.createElement("div");
    head.className = "chunk-card-head";

    const pagePill = document.createElement("span");
    pagePill.className = "pill pill-page";
    pagePill.textContent = `p.${page}`;

    const scoreBar = document.createElement("span");
    scoreBar.className = "score-bar";
    scoreBar.setAttribute("title", `similarity ${score.toFixed(2)}`);
    const scoreFill = document.createElement("span");
    scoreFill.className = "score-fill";
    scoreFill.style.width = pct + "%";
    scoreBar.appendChild(scoreFill);

    const scoreNum = document.createElement("span");
    scoreNum.className = "score-num";
    scoreNum.textContent = score.toFixed(2);

    head.append(pagePill, scoreBar, scoreNum);

    const previewEl = document.createElement("p");
    previewEl.className = "chunk-preview";
    previewEl.textContent = preview;

    li.append(head, previewEl);

    // Fusion provenance: dense/bm25/rerank ranks are each either a 1-based
    // int or null (candidate wasn't surfaced by that retriever/stage) — only
    // present ranks get a badge, dense/bm25/rerank order, no row at all if
    // none are present.
    const rankBadges = [
      ["dense", c && c.dense_rank],
      ["bm25", c && c.bm25_rank],
      ["rerank", c && c.rerank_rank],
    ].filter(([, rank]) => Number.isFinite(rank));

    if (rankBadges.length) {
      const badgeRow = document.createElement("div");
      badgeRow.className = "rank-badges";
      for (const [label, rank] of rankBadges) {
        const badge = document.createElement("span");
        badge.className = "rank-badge";
        badge.textContent = `${label} #${rank}`;
        badgeRow.appendChild(badge);
      }
      li.appendChild(badgeRow);
    }

    return li;
  }

  // The one deliberate innerHTML site: splices inline citation *buttons* into
  // model-generated prose. Every interpolated value is escaped, with no
  // exceptions, so the invariant is grep-able and trivially auditable. The
  // page group is `\d+` by construction (CITATION_RE), so data-page is
  // already digits-only; escAttr() is applied anyway so the "everything
  // dynamic is escaped" rule has zero silent exceptions.
  function renderCitedAnswer(container, text) {
    CITATION_RE.lastIndex = 0;
    let last = 0;
    let match;
    let html = "";
    while ((match = CITATION_RE.exec(text)) !== null) {
      html += esc(text.slice(last, match.index));
      const page = match[1];
      html += `<button type="button" class="pill pill-cite" data-page="${escAttr(page)}" ` +
              `title="${escAttr("cited page " + page)}">p.${esc(page)}</button>`;
      last = CITATION_RE.lastIndex;
    }
    html += esc(text.slice(last));
    container.innerHTML = html;
  }

  // Per-turn pipeline waterfall. Spans arrive in the server's recording order
  // (embed, retrieve, [rerank], generate); offsets/widths are normalized
  // against the earliest start / latest end across *all* spans in this trace
  // so the row of bars reads as a true waterfall rather than a stacked bar
  // chart. Every field is defensively coerced — a trace can round-trip
  // through localStorage (a user could hand-edit it) same as any other turn
  // field, so this must never throw on odd input.
  function buildTraceDetails(trace) {
    const rawSpans = Array.isArray(trace.spans) ? trace.spans : [];
    const spans = rawSpans.map((s) => ({
      name: s && typeof s.name === "string" ? s.name : "",
      start_ms: s && Number.isFinite(s.start_ms) ? s.start_ms : 0,
      end_ms: s && Number.isFinite(s.end_ms) ? s.end_ms : 0,
      duration_ms: s && Number.isFinite(s.duration_ms) ? s.duration_ms : 0,
      meta: s && s.meta && typeof s.meta === "object" ? s.meta : {},
    }));

    const minStart = Math.min(...spans.map((s) => s.start_ms));
    const maxEnd = Math.max(...spans.map((s) => s.end_ms));
    const totalMs = maxEnd - minStart;

    const details = document.createElement("details");
    details.className = "trace-details";

    const summary = document.createElement("summary");
    summary.textContent = `trace · ${Math.round(totalMs)} ms`;
    details.appendChild(summary);

    for (const span of spans) {
      const row = document.createElement("div");
      row.className = "span-row";

      const label = document.createElement("span");
      label.className = "span-label";
      label.textContent = span.name;

      const bar = document.createElement("span");
      bar.className = "span-bar";

      const fill = document.createElement("span");
      fill.className = "span-bar-fill" + (span.name === "generate" ? " span-bar-fill--accent" : "");
      const left = totalMs > 0 ? ((span.start_ms - minStart) / totalMs) * 100 : 0;
      const width = totalMs > 0 ? (span.duration_ms / totalMs) * 100 : 100;
      fill.style.left = left + "%";
      fill.style.width = width + "%";
      bar.appendChild(fill);

      const meta = document.createElement("span");
      meta.className = "span-meta";
      let metaText = `${Math.round(span.duration_ms)} ms`;
      if (Number.isFinite(span.meta.input_tokens) && Number.isFinite(span.meta.output_tokens)) {
        metaText += ` · ${span.meta.input_tokens + span.meta.output_tokens} tok`;
      }
      meta.textContent = metaText;

      row.append(label, bar, meta);
      details.appendChild(row);
    }

    return details;
  }

  function buildTurnEl(turn, index) {
    const wrap = document.createElement("div");
    wrap.className = "turn";
    wrap.dataset.turnIndex = String(index);

    const userBubble = document.createElement("div");
    userBubble.className = "turn-user";
    userBubble.textContent = typeof turn.q === "string" ? turn.q : "";
    wrap.appendChild(userBubble);

    const refused = turn.refused === true;
    const agentBubble = document.createElement("div");
    agentBubble.className = "turn-agent answer-card" + (refused ? " refused" : "");

    if (refused) {
      const badge = document.createElement("div");
      badge.className = "answer-badge";
      badge.textContent = "Not in the document";
      agentBubble.appendChild(badge);
    }

    const textEl = document.createElement("div");
    textEl.className = "answer-text";
    renderCitedAnswer(textEl, typeof turn.answer === "string" ? turn.answer : "");
    agentBubble.appendChild(textEl);

    // Retrieval previews collapsed behind a <details> — never expanded by
    // default; a citation-pill click force-opens the one it targets (below).
    const retrieval = Array.isArray(turn.retrieval) ? turn.retrieval : [];
    if (retrieval.length) {
      const details = document.createElement("details");
      details.className = "sources-details";

      const summary = document.createElement("summary");
      summary.textContent = `${retrieval.length} source${retrieval.length === 1 ? "" : "s"}`;
      details.appendChild(summary);

      const list = document.createElement("ul");
      list.className = "retrieval-list";
      for (const c of retrieval) list.appendChild(buildChunkCard(c));
      details.appendChild(list);

      agentBubble.appendChild(details);
    }

    // Pipeline waterfall — collapsed like sources; simply absent on turns
    // persisted before this feature shipped (trace is null there).
    if (turn.trace && Array.isArray(turn.trace.spans) && turn.trace.spans.length) {
      agentBubble.appendChild(buildTraceDetails(turn.trace));
    }

    wrap.appendChild(agentBubble);
    return wrap;
  }

  // Rebuilds the whole per-doc conversation log from state.convos. Cheap
  // enough to always do in full (capped at MAX_TURNS_PER_DOC turns) rather
  // than diffing — called on every append, doc switch, and clear.
  function renderConvoLog(docId) {
    const turns = state.convos[docId] || [];
    dom.convoLog.replaceChildren();
    dom.convoLog.hidden = turns.length === 0;
    turns.forEach((turn, i) => dom.convoLog.appendChild(buildTurnEl(turn, i)));
    dom.convoLog.scrollTop = dom.convoLog.scrollHeight;
  }

  // Clicking a [p.N] citation button scrolls + briefly highlights the
  // matching retrieval-preview card(s) *within that same turn* (a citation
  // in an earlier turn must not jump to a later turn's sources, or vice
  // versa). No matching preview in that turn → no-op, never an error.
  function handleConvoLogClick(e) {
    const btn = e.target.closest("button.pill-cite");
    if (!btn) return;

    const page = btn.dataset.page || "";
    if (!/^\d+$/.test(page)) return; // pills are digits-only; anything else is inert

    const turnEl = btn.closest(".turn");
    if (!turnEl) return;

    const matches = Array.from(turnEl.querySelectorAll(".chunk-card"))
      .filter((li) => li.dataset.page === page);
    if (!matches.length) return;

    const details = turnEl.querySelector("details.sources-details");
    if (details) details.open = true;

    matches[0].scrollIntoView({ behavior: "smooth", block: "center" });
    for (const card of matches) {
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), FLASH_MS);
    }
  }

  // ---------------------------------------------------------------------
  // ask: submit
  // ---------------------------------------------------------------------

  async function handleAskSubmit(e) {
    e.preventDefault();
    if (askInFlight) return;
    if (!state.selectedDocId) return;

    clearAskError();
    hideAskExpiredBanner();

    const question = dom.questionInput.value.trim();
    if (!question) {
      showAskError("type a question first");
      return;
    }

    const docId = state.selectedDocId;
    setAskBusy(true);
    showAskStatus("retrieving relevant passages…");

    const modelValue = dom.modelSelect.value || undefined;
    const byoKey = dom.byoKeyInput.value.trim();
    // Send this doc's whole stored conversation as history; the server keeps
    // only the last 6 well-formed turns (and truncates long answers) so no
    // client-side slicing is needed here.
    const payload = { doc_id: docId, question, history: historyForDoc(docId) };
    if (modelValue) payload.model = modelValue;
    if (byoKey) payload.byo_key = byoKey;

    let pendingRetrieval = [];
    let pendingTrace = null;

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });

      await consumeSSE(resp, {
        retrieval: (data) => {
          pendingRetrieval = Array.isArray(data.chunks) ? data.chunks : [];
          showAskStatus("writing an answer…");
        },
        trace: (data) => {
          pendingTrace = data;
        },
        answer: (data) => {
          hideAskStatus();
          appendTurn(docId, {
            q: question,
            answer: typeof data.answer === "string" ? data.answer : "",
            citations: Array.isArray(data.citations) ? data.citations : [],
            refused: data.refused === true,
            retrieval: pendingRetrieval,
            trace: pendingTrace,
            ts: Date.now(),
          });
          renderConvoLog(docId);
          // Asking appends a turn; the input stays (cleared) for the next follow-up.
          dom.questionInput.value = "";
          dom.charCounter.textContent = "0/500";
          refreshAskButton();
        },
        error: (data) => {
          hideAskStatus();
          const msg = typeof data.message === "string" ? data.message : "something went wrong — try again";
          if (msg.includes("document not found")) {
            // Not saved as a turn — the doc session is gone, there's nothing to log.
            showAskExpiredBanner();
          } else {
            showAskError(friendlyMessage(msg));
          }
        },
      });
    } catch {
      hideAskStatus();
      showAskError("network error — check your connection and try again");
    } finally {
      setAskBusy(false);
      dom.questionInput.focus({ preventScroll: true });
    }
  }

  function handleReingestClick() {
    const src = state.sourceByDocId.get(state.selectedDocId);
    hideAskExpiredBanner();
    if (src && src.type === "url") {
      clearFileSelection();
      dom.urlInput.value = src.value;
      refreshIngestButton();
    } else {
      showIngestError("select your PDF file again to re-ingest it");
    }
    dom.ingestPanelEl.scrollIntoView({ behavior: "smooth", block: "start" });
    dom.fileInput.focus();
  }

  // ---------------------------------------------------------------------
  // wiring
  // ---------------------------------------------------------------------

  function wireEvents() {
    dom.ingestForm.addEventListener("submit", handleIngestSubmit);
    dom.askForm.addEventListener("submit", handleAskSubmit);

    dom.fileInput.addEventListener("change", () => {
      const f = dom.fileInput.files && dom.fileInput.files[0];
      if (f) setSelectedFile(f);
    });
    dom.clearFileBtn.addEventListener("click", clearFileSelection);

    ["dragenter", "dragover"].forEach((evt) =>
      dom.dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dom.dropzone.classList.add("dragover");
      })
    );
    ["dragleave"].forEach((evt) =>
      dom.dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dom.dropzone.classList.remove("dragover");
      })
    );
    dom.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dom.dropzone.classList.remove("dragover");
      const dt = e.dataTransfer;
      const f = dt && dt.files && dt.files[0];
      if (!f) return;
      if (isPdfFile(f)) {
        setSelectedFile(f);
      } else {
        showIngestError("please drop a PDF file");
      }
    });

    dom.urlInput.addEventListener("input", refreshIngestButton);

    dom.clearDocsBtn.addEventListener("click", handleClearDocs);
    dom.reingestBtn.addEventListener("click", handleReingestClick);
    dom.convoLog.addEventListener("click", handleConvoLogClick);

    dom.questionInput.addEventListener("input", () => {
      dom.charCounter.textContent = `${dom.questionInput.value.length}/500`;
      refreshAskButton();
    });
    dom.questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!dom.askBtn.disabled) dom.askForm.requestSubmit();
      }
    });

    // Prevent an accidental drop outside the dropzone from navigating the tab.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());
  }

  // ---------------------------------------------------------------------
  // init
  // ---------------------------------------------------------------------

  function init() {
    cacheDom();
    dom.ingestPanelEl = document.getElementById("ingest-panel");

    state.docs = loadDocs();
    state.convos = loadConvos();
    renderDocSwitcher();

    refreshIngestButton();
    refreshAskButton();

    loadModels();
    wireEvents();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
