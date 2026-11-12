// 創造力 3D — 單頁應用
// 網址帶 ?mock=1 時全部改讀 /static/mock/*.json，不打 API。

const MOCK = new URLSearchParams(location.search).get("mock") === "1";
const MAX_TEXT = 20;
const MAX_NOTE = 60;
const MAX_ITEMS = 60;
const STIM_RE = /^[一-鿿]{1,6}$/;
const POLL_MS = 2000;
const POLL_LIMIT_MS = 180000;

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// 匿名 ID（localStorage；失敗時用記憶體變數）
// ---------------------------------------------------------------------------
let memorySessionId = null;
function newId() {
  try {
    if (crypto?.randomUUID) return crypto.randomUUID().replace(/-/g, "");
  } catch { /* 忽略 */ }
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
}
function getSessionId() {
  try {
    let id = localStorage.getItem("c3d.session");
    if (!id || id.length < 8 || id.length > 64) {
      id = newId();
      localStorage.setItem("c3d.session", id);
    }
    return id;
  } catch {
    if (!memorySessionId) memorySessionId = newId();
    return memorySessionId;
  }
}
let SESSION = getSessionId();
function setSessionId(id) {
  SESSION = id;
  memorySessionId = id;
  try { localStorage.setItem("c3d.session", id); } catch { /* 忽略 */ }
}

// ---------------------------------------------------------------------------
// 玩家暱稱（localStorage；失敗時用記憶體變數）
// ---------------------------------------------------------------------------
const NICK_KEY = "c3d_nickname";
const NICK_RE = /^[\p{Script=Han}A-Za-z0-9_]{1,12}$/u;
const PIN_RE = /^\d{4}$/;
let memoryNickname = null;
function getNickname() {
  try { return localStorage.getItem(NICK_KEY) || null; } catch { return memoryNickname; }
}
function setNickname(name) {
  memoryNickname = name || null;
  try {
    if (name) localStorage.setItem(NICK_KEY, name); else localStorage.removeItem(NICK_KEY);
  } catch { /* 忽略 */ }
}

// ---------------------------------------------------------------------------
// API（含 mock）
// ---------------------------------------------------------------------------
class ApiError extends Error {
  constructor(message, code = "internal", status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function http(path, { method = "GET", body, query } = {}) {
  const url = new URL(path, location.origin);
  if (query) Object.entries(query).forEach(([k, v]) => url.searchParams.set(k, v));
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("連不上伺服器，請確認網路或稍後再試。", "network");
  }
  let json = null;
  try { json = await res.json(); } catch { /* 非 JSON */ }
  if (!res.ok) {
    // ErrorResponse {code, message}；FastAPI 驗證錯誤是 {detail: [...]}
    const msg = json?.message || (Array.isArray(json?.detail) ? "輸入格式不正確。" : json?.detail) || `伺服器錯誤（${res.status}）`;
    throw new ApiError(msg, json?.code || "internal", res.status);
  }
  return json;
}

const mockState = { stimuli: null, history: null, pending: new Map() };
async function mockJson(name) {
  const r = await fetch(`/static/mock/${name}`);
  if (!r.ok) throw new ApiError(`找不到 mock/${name}`);
  return r.json();
}

const api = MOCK ? {
  async config() { return { answer_seconds: 180, percentile_min_users: 20 }; },
  async stimuli() {
    if (!mockState.stimuli) mockState.stimuli = await mockJson("stimuli.json");
    // 模擬：新增的詞第 2 次輪詢後 ready
    for (const [id, n] of mockState.pending) {
      if (n >= 2) {
        const s = mockState.stimuli.find((x) => x.id === id);
        if (s) s.ready = true;
        mockState.pending.delete(id);
      } else mockState.pending.set(id, n + 1);
    }
    return structuredClone(mockState.stimuli);
  },
  async createStimulus(word) {
    await sleep(400);
    if (!STIM_RE.test(word)) throw new ApiError("請輸入 1–6 個中文字。", "invalid_input");
    if (/忽略|指示/.test(word)) throw new ApiError("這個詞無法使用，請換一個。", "blocked_input");
    await this.stimuli();
    const exists = mockState.stimuli.find((s) => s.word === word);
    if (exists) return structuredClone(exists);
    const s = { id: `m${Date.now()}`, word, source: "mine", abstraction_level: null, source_note: "", response_count: 0, ready: false };
    mockState.stimuli.push(s);
    mockState.pending.set(s.id, 0);
    return structuredClone(s);
  },
  async publish(id) {
    await sleep(300);
    const s = mockState.stimuli?.find((x) => x.id === id);
    if (!s) throw new ApiError("找不到這個詞。", "unknown_stimulus");
    s.source = "community";
    return structuredClone(s);
  },
  async submit() {
    await sleep(2600);
    return mockJson("analysis.json");
  },
  async consent(submissionId) {
    await sleep(400);
    const h = mockState.history?.find((x) => x.submission_id === submissionId);
    if (h) h.shared = true;
    return { submission_id: submissionId, shared: true, message: "已匿名加入星圖，謝謝你！" };
  },
  async login({ nickname, pin }) {
    await sleep(400);
    if (!NICK_RE.test(nickname) || !PIN_RE.test(pin)) throw new ApiError("暱稱或 PIN 格式不正確。", "invalid_input", 400);
    if (pin === "0000") throw new ApiError("暱稱或 PIN 不正確。", "auth_failed", 401); // 方便測錯誤狀態
    const p = await mockJson("player.json");
    return { ...p, nickname };
  },
  async player() {
    return { nickname: getNickname() };
  },
  async history() {
    if (!mockState.history) mockState.history = await mockJson("history.json");
    return structuredClone(mockState.history);
  },
  async submission(id) {
    await sleep(500);
    const data = await mockJson("analysis.json");
    const h = mockState.history?.find((x) => x.submission_id === id);
    if (h) {
      data.submission_id = h.submission_id;
      data.stimulus = { ...data.stimulus, id: h.stimulus_id, word: h.stimulus_word };
      data.created_at = h.created_at;
    }
    return data;
  },
} : {
  config: () => http("/api/config"),
  stimuli: () => http("/api/stimuli", { query: { client_session_id: SESSION } }),
  createStimulus: (word) => http("/api/stimuli", { method: "POST", body: { word, client_session_id: SESSION } }),
  publish: (id) => http(`/api/stimuli/${encodeURIComponent(id)}/publish`, { method: "POST", body: { client_session_id: SESSION, confirm_publish: true } }),
  submit: (payload) => http("/api/submit", { method: "POST", body: payload }),
  consent: (submissionId) => http("/api/consent", { method: "POST", query: { client_session_id: SESSION }, body: { submission_id: submissionId, agree_to_share: true } }),
  login: ({ nickname, pin }) => http("/api/player/login", { method: "POST", body: { nickname, pin, client_session_id: SESSION } }),
  player: () => http("/api/player", { query: { client_session_id: SESSION } }),
  history: () => http("/api/history", { query: { client_session_id: SESSION } }),
  submission: (id) => http(`/api/submissions/${encodeURIComponent(id)}`, { query: { client_session_id: SESSION } }),
  deleteSubmission: (id) => http(`/api/submissions/${encodeURIComponent(id)}`, { method: "DELETE", query: { client_session_id: SESSION } }),
};

// ---------------------------------------------------------------------------
// 狀態
// ---------------------------------------------------------------------------
const state = {
  config: { answer_seconds: 180, percentile_min_users: 20 },
  stimuli: [],
  current: null,      // 正在作答的 Stimulus
  items: [],          // [{text, note, open}]
  startedAt: 0,
  deadline: 0,
  timer: null,
  timeUp: false,
  submitting: false,
  result: null,
  map: null,
  polling: null,
  nickname: null,
  history: null,     // null = 尚未載入
  historyToken: 0,
  resultShared: false, // 目前結果頁這筆是否已分享（重看舊紀錄時用）
};

// ---------------------------------------------------------------------------
// 視圖切換
// ---------------------------------------------------------------------------
function show(view) {
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== view; });
  document.body.dataset.view = view;
  window.scrollTo({ top: 0 });
}

let toastTimer = null;
function toast(msg, ms = 2600) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function setMsg(el, text, kind = "") {
  el.className = "form-msg" + (kind ? ` ${kind}` : "") + (el.classList.contains("center") ? " center" : "");
  el.innerHTML = text;
}

// ---------------------------------------------------------------------------
// 首頁
// ---------------------------------------------------------------------------
async function loadStimuli() {
  try {
    state.stimuli = await api.stimuli();
    renderHome();
  } catch (e) {
    $("#group-preset").innerHTML = `<span class="muted">刺激詞載入失敗：${esc(e.message)}</span>`;
  }
}

function chip(s) {
  const disabled = !s.ready;
  const meta = disabled
    ? `<span class="spinner" aria-hidden="true"></span><small>準備中</small>`
    : s.source === "preset"
      ? `<small>L${s.abstraction_level ?? "–"}</small>`
      : `<small>${s.response_count} 人</small>`;
  return `<button type="button" class="chip" data-stim="${esc(s.id)}" ${disabled ? "disabled" : ""} title="${esc(s.source_note || "")}">${esc(s.word)}${meta}</button>`;
}

function renderHome(highlightId) {
  const list = state.stimuli || [];
  const preset = list.filter((s) => s.source === "preset");
  const community = list.filter((s) => s.source === "community").sort((a, b) => b.response_count - a.response_count);
  const mine = list.filter((s) => s.source === "mine");

  $("#group-preset").innerHTML = preset.length ? preset.map(chip).join("") : `<span class="muted">目前沒有預設詞。</span>`;
  $("#group-community").innerHTML = community.length
    ? community.map(chip).join("")
    : `<span class="muted">還沒有人公開詞。新增一個你的詞，公開後就會出現在這裡。</span>`;
  $("#group-mine").innerHTML = mine.length
    ? mine.map((s) => `<span class="chip-group">${chip(s)}${s.ready ? `<button type="button" class="chip-pub" data-publish="${esc(s.id)}">公開到社群</button>` : ""}</span>`).join("")
    : `<span class="muted">你還沒有新增詞。</span>`;

  if (highlightId) {
    const el = document.querySelector(`[data-stim="${CSS.escape(highlightId)}"]`);
    if (el) {
      el.classList.add("highlight");
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      setTimeout(() => el.classList.remove("highlight"), 2400);
    }
  }
}

function goHome() {
  if (!state.nickname) { showLogin(); return; }
  stopTimer();
  disposeMap();
  state.current = null;
  state.items = [];
  state.result = null;
  show("home");
  loadStimuli();
  loadHistory();
}

async function onAddStimulus(ev) {
  ev.preventDefault();
  const input = $("#addInput");
  const msg = $("#addMsg");
  const word = input.value.trim();
  if (!STIM_RE.test(word)) {
    setMsg(msg, "請輸入 1–6 個中文字，不能有英文、數字或符號。", "err");
    return;
  }
  const btn = $("#addBtn");
  btn.disabled = true;
  setMsg(msg, `<span class="spinner"></span>檢查中…`);
  try {
    const s = await api.createStimulus(word);
    input.value = "";
    if (s.ready) {
      // 已存在（或已準備好）的詞：直接進作答
      setMsg(msg, s.source === "mine" ? "" : `「${esc(s.word)}」已經在詞庫裡，直接開始。`, "ok");
      await loadStimuli();
      startAnswer(s);
    } else {
      upsertStimulus(s);
      renderHome(s.id);
      setMsg(msg, `<span class="spinner"></span>正在準備 AI 參考點…（「${esc(s.word)}」準備好就能作答）`);
      pollUntilReady(s.id);
    }
  } catch (e) {
    setMsg(msg, esc(e.message), "err");
  } finally {
    btn.disabled = false;
  }
}

function upsertStimulus(s) {
  const i = state.stimuli.findIndex((x) => x.id === s.id);
  if (i >= 0) state.stimuli[i] = s; else state.stimuli.push(s);
}

async function pollUntilReady(id) {
  const token = {};
  state.polling = token;
  const started = Date.now();
  const msg = $("#addMsg");
  while (state.polling === token && Date.now() - started < POLL_LIMIT_MS) {
    await sleep(POLL_MS);
    if (state.polling !== token) return;
    try {
      state.stimuli = await api.stimuli();
    } catch {
      continue; // 暫時失敗就繼續等
    }
    const s = state.stimuli.find((x) => x.id === id);
    if (document.body.dataset.view === "home") renderHome();
    if (!s) {
      setMsg(msg, "這個詞準備失敗，請換一個詞再試。", "err");
      state.polling = null;
      return;
    }
    if (s.ready) {
      setMsg(msg, `「${esc(s.word)}」準備好了，點一下就能開始。`, "ok");
      if (document.body.dataset.view === "home") renderHome(s.id);
      state.polling = null;
      return;
    }
  }
  if (state.polling === token) {
    setMsg(msg, "準備時間比預期久，稍後重新整理看看。", "err");
    state.polling = null;
  }
}

// 公開到社群
let publishTarget = null;
function openPublish(id) {
  const s = state.stimuli.find((x) => x.id === id);
  if (!s) return;
  publishTarget = s;
  $("#publishWord").textContent = s.word;
  $("#publishCheck").checked = false;
  $("#publishOk").disabled = true;
  setMsg($("#publishMsg"), "");
  $("#publishModal").hidden = false;
  $("#publishCheck").focus();
}
function closePublish() {
  $("#publishModal").hidden = true;
  publishTarget = null;
}
async function confirmPublish() {
  if (!publishTarget || !$("#publishCheck").checked) return;
  const btn = $("#publishOk");
  btn.disabled = true;
  setMsg($("#publishMsg"), `<span class="spinner"></span>公開中…`);
  try {
    const s = await api.publish(publishTarget.id);
    upsertStimulus(s);
    closePublish();
    renderHome(s.id);
    toast(`「${s.word}」已公開到社群詞庫`);
  } catch (e) {
    setMsg($("#publishMsg"), esc(e.message), "err");
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 玩家身分（暱稱 + PIN）
// ---------------------------------------------------------------------------
function renderWho() {
  const who = $("#who");
  if (state.nickname) {
    $("#whoName").textContent = state.nickname;
    who.hidden = false;
  } else {
    who.hidden = true;
  }
}

function showLogin() {
  stopTimer();
  disposeMap();
  state.polling = null;
  state.current = null;
  state.items = [];
  state.result = null;
  state.history = null;
  state.historyToken += 1;
  renderWho();
  $("#loginPin").value = "";
  setMsg($("#loginMsg"), "");
  $("#loginBtn").disabled = false;
  show("login");
  setTimeout(() => ($("#loginNick").value ? $("#loginPin") : $("#loginNick")).focus(), 50);
}

function enterHome() {
  renderWho();
  show("home");
  loadStimuli();
  loadHistory();
}

async function onLogin(ev) {
  ev.preventDefault();
  const nickname = $("#loginNick").value.trim();
  const pin = $("#loginPin").value.trim();
  const msg = $("#loginMsg");
  if (!nickname) { setMsg(msg, "請輸入暱稱。", "err"); $("#loginNick").focus(); return; }
  if (!NICK_RE.test(nickname)) {
    setMsg(msg, "暱稱需為 1–12 個字，只能用中文、英文、數字或底線。", "err");
    $("#loginNick").focus();
    return;
  }
  if (!PIN_RE.test(pin)) { setMsg(msg, "PIN 必須剛好是 4 位數字。", "err"); $("#loginPin").focus(); return; }

  const btn = $("#loginBtn");
  btn.disabled = true;
  setMsg(msg, `<span class="spinner"></span>確認中…`);
  try {
    const r = await api.login({ nickname, pin });
    if (r?.session_id) setSessionId(r.session_id);
    const name = r?.nickname || nickname;
    setNickname(name);
    state.nickname = name;
    state.stimuli = [];
    $("#loginPin").value = "";
    setMsg(msg, "");
    enterHome();
    if (r?.is_new) toast(`歡迎，${name}！已建立你的身分，記住你的 PIN`, 5000);
    else toast(`歡迎回來，${name}`);
  } catch (e) {
    setMsg(msg, esc(e.message || "登入失敗，請稍後再試。"), "err");
    if (e.code === "auth_failed") { $("#loginPin").value = ""; $("#loginPin").focus(); }
  } finally {
    btn.disabled = false;
  }
}

function switchIdentity() {
  setNickname(null);
  state.nickname = null;
  setSessionId(newId());
  state.stimuli = [];
  $("#loginNick").value = "";
  showLogin();
}

// ---------------------------------------------------------------------------
// 我的紀錄
// ---------------------------------------------------------------------------
const DATE_FMT = new Intl.DateTimeFormat("zh-TW", {
  year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
});
function fmtDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : DATE_FMT.format(d);
}

async function loadHistory() {
  if (!state.nickname) return;
  const token = ++state.historyToken;
  const box = $("#history");
  if (state.history === null) box.innerHTML = `<span class="muted"><span class="spinner inline"></span>載入中…</span>`;
  try {
    const list = await api.history();
    if (token !== state.historyToken) return;
    state.history = Array.isArray(list) ? list : [];
    renderHistory();
  } catch (e) {
    if (token !== state.historyToken) return;
    box.innerHTML = `<span class="muted">紀錄載入失敗：${esc(e.message)}</span>`;
  }
}

function renderHistory() {
  const list = state.history || [];
  $("#historyCount").textContent = list.length ? `共 ${list.length} 筆` : "";
  if (!list.length) {
    $("#history").innerHTML = `<span class="muted">還沒有紀錄，選一個詞開始吧</span>`;
    return;
  }
  $("#history").innerHTML = list.map((h) => {
    const dist = typeof h.distance === "number" ? h.distance.toFixed(2) : "—";
    const n = typeof h.fluency === "number" ? Math.round(h.fluency) : "—";
    return `
    <div class="hist-item">
    <button type="button" class="hist-del" data-del="${esc(h.submission_id)}" aria-label="刪除「${esc(h.stimulus_word)}」這筆紀錄">刪除</button>
    <button type="button" class="hist-card" data-sub="${esc(h.submission_id)}">
      <span class="hist-top">
        <span class="hist-word">${esc(h.stimulus_word)}</span>
        ${h.shared ? `<span class="badge-shared">已分享</span>` : ""}
      </span>
      <span class="hist-date">${esc(fmtDate(h.created_at))}</span>
      <span class="hist-meta">想了 ${n} 則<i>·</i>距離 ${dist}</span>
      ${h.headline ? `<span class="hist-headline">${esc(h.headline)}</span>` : ""}
    </button>
    </div>`;
  }).join("");
}

// ---- 刪除紀錄（需要確認，刪了無法復原） ----
let deletingId = null;
function openDelete(id) {
  const h = (state.history || []).find((x) => x.submission_id === id);
  if (!h) return;
  deletingId = id;
  $("#deleteWord").textContent = h.stimulus_word;
  $("#deleteWhen").textContent = fmtDate(h.created_at);
  $("#deleteShared").hidden = !h.shared;
  $("#deleteMsg").textContent = "";
  $("#deleteModal").hidden = false;
  $("#deleteCancel").focus();
}
function closeDelete() { $("#deleteModal").hidden = true; deletingId = null; }
async function confirmDelete() {
  if (!deletingId) return;
  const btn = $("#deleteConfirm");
  btn.disabled = true;
  try {
    if (MOCK) { state.history = state.history.filter((x) => x.submission_id !== deletingId); }
    else { await api.deleteSubmission(deletingId); state.history = state.history.filter((x) => x.submission_id !== deletingId); }
    closeDelete();
    renderHistory();
    toast("已刪除這筆紀錄。");
  } catch (e) {
    $("#deleteMsg").textContent = e.message || "刪除失敗，請稍後再試。";
  } finally {
    btn.disabled = false;
  }
}

let openingSub = null;
async function openSubmission(id, el) {
  if (openingSub) return;
  const h = (state.history || []).find((x) => x.submission_id === id);
  openingSub = id;
  el?.classList.add("loading");
  el?.setAttribute("aria-busy", "true");
  try {
    const data = await api.submission(id);
    if (document.body.dataset.view !== "home") return; // 使用者已離開
    showResult(data, { shared: typeof data?.shared === "boolean" ? data.shared : !!h?.shared, fromHistory: true });
  } catch (e) {
    toast(e.message || "讀取紀錄失敗，請稍後再試。");
  } finally {
    openingSub = null;
    el?.classList.remove("loading");
    el?.removeAttribute("aria-busy");
  }
}

// ---------------------------------------------------------------------------
// 作答頁
// ---------------------------------------------------------------------------
function startAnswer(stimulus) {
  state.polling = null;
  state.current = stimulus;
  state.items = [];
  state.timeUp = false;
  state.submitting = false;
  $("#stimWord").textContent = stimulus.word;
  $("#answerInput").value = "";
  $("#answerInput").disabled = false;
  updateCounter();
  setMsg($("#answerMsg"), "");
  renderItems();
  show("answer");
  startTimer(state.config.answer_seconds || 180);
  setTimeout(() => $("#answerInput").focus(), 50);
}

function startTimer(seconds) {
  stopTimer();
  state.total = seconds;
  state.startedAt = Date.now();
  state.deadline = state.startedAt + seconds * 1000;
  tick();
  state.timer = setInterval(tick, 250);
}
function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}
function tick() {
  const left = Math.max(0, Math.ceil((state.deadline - Date.now()) / 1000));
  const m = Math.floor(left / 60);
  const s = String(left % 60).padStart(2, "0");
  $("#timerText").textContent = `${m}:${s}`;
  const frac = state.total ? left / state.total : 0;
  $("#timerBar").style.strokeDashoffset = String(119.38 * (1 - frac));
  $("#timer").classList.toggle("low", left <= 15);
  if (left <= 0 && !state.timeUp) {
    state.timeUp = true;
    stopTimer();
    onTimeUp();
  }
}
function onTimeUp() {
  // 輸入框裡還沒按 Enter 的字也一起收進來
  const pending = $("#answerInput").value.trim();
  if (pending) addItem(pending, { silent: true });
  if (!state.items.length) {
    $("#answerInput").disabled = false;
    setMsg($("#answerMsg"), "時間到了，但還沒有任何聯想。寫下至少一則再按送出，或換一個詞。", "err");
    return;
  }
  submit();
}

function updateCounter() {
  const n = [...$("#answerInput").value].length;
  const el = $("#answerCount");
  el.textContent = `${n}/${MAX_TEXT}`;
  el.classList.toggle("full", n >= MAX_TEXT);
}

function addItem(raw, { silent = false } = {}) {
  const text = raw.trim();
  const msg = $("#answerMsg");
  if (!text) return false;
  if ([...text].length > MAX_TEXT) {
    if (!silent) setMsg(msg, `每則最多 ${MAX_TEXT} 個字。`, "err");
    return false;
  }
  if (state.items.length >= MAX_ITEMS) {
    if (!silent) setMsg(msg, `最多 ${MAX_ITEMS} 則。`, "err");
    return false;
  }
  if (state.items.some((it) => it.text === text)) {
    if (!silent) setMsg(msg, `「${esc(text)}」已經寫過了。`, "err");
    return false;
  }
  state.items.unshift({ text, note: "", open: false });
  setMsg(msg, "");
  renderItems();
  return true;
}

function renderItems() {
  const ol = $("#items");
  const total = state.items.length;
  ol.innerHTML = state.items.map((it, i) => {
    const noteLen = [...(it.note || "")].length;
    return `
    <li class="item" data-i="${i}">
      <div class="item-row">
        <span class="item-no">${String(total - i).padStart(2, "0")}</span>
        <span class="item-text">${esc(it.text)}</span>
        <button type="button" class="item-btn${it.note ? " has-note" : ""}" data-toggle="${i}" aria-expanded="${it.open}">${it.open ? "收起" : it.note ? "已補充說明" : "我怎麼想到的"}</button>
        <button type="button" class="item-btn del" data-del="${i}" aria-label="刪除「${esc(it.text)}」">刪除</button>
      </div>
      ${it.open ? `
      <div class="item-note">
        <input type="text" maxlength="${MAX_NOTE}" data-note="${i}" value="${esc(it.note)}" placeholder="選填：例如「我爸修車時都在吃蘋果」" aria-label="我怎麼想到的" />
        <span class="counter" data-note-count="${i}">${noteLen}/${MAX_NOTE}</span>
      </div>` : ""}
    </li>`;
  }).join("");
  $("#itemCount").textContent = `已寫 ${total} 則`;
  $("#submitBtn").disabled = total === 0 || state.submitting;
  $("#submitBtn").textContent = state.timeUp ? "送出" : "提早送出";
}

function onItemsClick(ev) {
  const t = ev.target.closest("button");
  if (!t) return;
  if (t.dataset.toggle !== undefined) {
    const it = state.items[+t.dataset.toggle];
    it.open = !it.open;
    renderItems();
    if (it.open) document.querySelector(`[data-note="${t.dataset.toggle}"]`)?.focus();
  } else if (t.dataset.del !== undefined) {
    state.items.splice(+t.dataset.del, 1);
    renderItems();
  }
}
function onItemsInput(ev) {
  const i = ev.target.dataset.note;
  if (i === undefined) return;
  state.items[+i].note = ev.target.value;
  const c = document.querySelector(`[data-note-count="${i}"]`);
  if (c) c.textContent = `${[...ev.target.value].length}/${MAX_NOTE}`;
}

// ---------------------------------------------------------------------------
// 送出 → 分析中
// ---------------------------------------------------------------------------
async function submit() {
  if (state.submitting || !state.items.length || !state.current) return;
  state.submitting = true;
  stopTimer();
  const elapsed = Math.min(600, Math.max(0, Math.round((Date.now() - state.startedAt) / 1000)));
  const payload = {
    stimulus_id: state.current.id,
    // 依作答順序送出（畫面上是最新的在上面）
    associations: [...state.items].reverse().map((it) => {
      const note = it.note.trim();
      return note ? { text: it.text, note } : { text: it.text };
    }),
    elapsed_seconds: elapsed,
    client_session_id: SESSION,
  };

  show("analyzing");
  $("#analyzeError").hidden = true;
  $("#steps").hidden = false;
  const steps = [...document.querySelectorAll("#steps li")];
  steps.forEach((li) => li.classList.remove("active", "done"));
  let stepIdx = 0;
  steps[0].classList.add("active");
  const stepTimer = setInterval(() => {
    if (stepIdx < steps.length - 1) {
      steps[stepIdx].classList.replace("active", "done");
      stepIdx += 1;
      steps[stepIdx].classList.add("active");
    }
  }, 1100);

  const minWait = sleep(1800);
  try {
    const result = await api.submit(payload);
    await minWait;
    clearInterval(stepTimer);
    for (let i = stepIdx; i < steps.length; i++) {
      steps[i].classList.remove("active");
      steps[i].classList.add("done");
      await sleep(180);
    }
    await sleep(350);
    state.submitting = false;
    showResult(result);
  } catch (e) {
    clearInterval(stepTimer);
    state.submitting = false;
    $("#steps").hidden = true;
    $("#analyzeErrorMsg").textContent = e.message || "分析失敗，請稍後再試。";
    $("#analyzeError").hidden = false;
  }
}

function reanalyze() {
  const data = state.result;
  if (!data || state.submitting) return;
  state.current = data.stimulus;
  // state.items 是「最新的在前面」，submit() 會再反轉回作答順序
  state.items = (data.associations || [])
    .map((a) => ({ text: a.text, note: a.note || "", open: false }))
    .reverse();
  state.startedAt = Date.now();
  state.timeUp = true;
  submit();
}

function backToAnswer() {
  // 保留已輸入的答案，計時停止，讓使用者直接重送
  state.timeUp = true;
  stopTimer();
  $("#timerText").textContent = "0:00";
  $("#stimWord").textContent = state.current?.word ?? "";
  setMsg($("#answerMsg"), "你的答案都還在，可以修改後再按送出。");
  renderItems();
  show("answer");
}

// ---------------------------------------------------------------------------
// 結果頁
// ---------------------------------------------------------------------------
const DIMS = [
  { key: "distance", name: "距離", en: "Distance", desc: "想得多遠", fmt: (v) => v.toFixed(2), max: 1 },
  { key: "breadth", name: "廣度", en: "Breadth", desc: "想得多散", fmt: (v) => v.toFixed(2), max: 1 },
  { key: "rarity", name: "稀有度", en: "Rarity", desc: "多少人也想到", fmt: (v) => v.toFixed(2), max: 1 },
  { key: "appropriateness", name: "合理性", en: "Fit", desc: "合不合理", fmt: (v) => v.toFixed(2), max: 1 },
  { key: "fluency", name: "數量", en: "Fluency", desc: "想了多少", fmt: (v) => `${Math.round(v)} 則`, max: 30 },
];
const DIM_HINT = {
  distance: "平均每個聯想離刺激詞多遠，越大跳得越遠",
  breadth: "合理的聯想彼此差多少，越大越分散",
  rarity: "越接近 1，越少人想到一樣的答案",
  appropriateness: "和刺激詞說得通的比例，1 代表全部合理",
  fluency: "被判定有效的聯想則數",
};
const NULL_REASON = {
  breadth: "合理的聯想少於 3 則",
  rarity: "作答人數還太少",
};

function renderDims(fp, pool) {
  return DIMS.map((d) => {
    const s = fp?.[d.key] || {};
    const raw = typeof s.raw === "number" ? s.raw : null;
    const pct = typeof s.percentile === "number" ? s.percentile : null;
    const valText = raw === null ? "—" : d.fmt(raw);
    const width = pct !== null ? pct : raw !== null ? Math.min(100, (raw / d.max) * 100) : 0;
    const pctText = pct !== null
      ? `<span class="dim-pct">高於 ${Math.round(pct)}% 的人</span>`
      : raw === null
        ? `<span class="dim-pct na">${NULL_REASON[d.key] || "資料不足"}</span>`
        : `<span class="dim-pct na">樣本不足${pool && !pool.enough_for_percentile ? `（${pool.human_users}/${state.config.percentile_min_users || 20} 人）` : ""}</span>`;
    return `
      <div class="dim">
        <div class="dim-top">
          <span class="dim-name">${d.desc}<small>${d.name} · ${d.en}</small></span>
          <span class="dim-val">${valText}</span>
        </div>
        <div class="dim-bar${pct === null ? " raw" : ""}"><i data-w="${width.toFixed(1)}"></i></div>
        <div class="dim-sub"><span>${DIM_HINT[d.key]}</span>${pctText}</div>
      </div>`;
  }).join("");
}

const DEFAULT_INTERP = {
  headline: "你的聯想星圖",
  strengths: ["每顆星代表你的一個聯想，離中心越遠，代表和刺激詞的語義距離越大。"],
  suggestions: ["點星星看看聯想橋，想想看還有哪些方向沒有走過。"],
};

async function showResult(data, { shared = false, fromHistory = false } = {}) {
  state.result = data;
  state.resultShared = shared;
  show("result");
  // 重看舊紀錄時，可以用最新規則重跑同一批答案（不用重打）
  $("#reanalyzeBtn").hidden = !fromHistory;

  const interp = data.interpretation || DEFAULT_INTERP;
  $("#headline").textContent = interp.headline;
  $("#mapWord").textContent = data.stimulus?.word ?? "";
  const pool = data.pool;
  $("#mapPool").textContent = pool ? `${pool.human_users} 位使用者 · ${pool.seed_points} 個 AI 參考點` : "";
  $("#dims").innerHTML = renderDims(data.fingerprint, pool);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".dim-bar i").forEach((i) => { i.style.width = `${i.dataset.w}%`; });
  }));
  $("#strengths").innerHTML = (interp.strengths || []).map((t) => `<li>${esc(t)}</li>`).join("");
  $("#suggestions").innerHTML = (interp.suggestions || []).map((t) => `<li>${esc(t)}</li>`).join("");
  const warnings = [...(data.warnings || [])];
  if (!data.interpretation) warnings.unshift("AI 解讀暫時無法產生，以下為預設說明；指紋數字不受影響。");
  $("#warnings").innerHTML = warnings.map((t) => `<li>${esc(t)}</li>`).join("");
  $("#warningsBlock").hidden = warnings.length === 0;

  const assocs = data.associations || [];
  $("#myItems").innerHTML = assocs.map((a, i) => {
    const cls = a.weight === 0 ? " is-bad" : (a.verdict === "borderline" || a.verdict === "unreasonable") ? " is-border" : "";
    return `<button type="button" class="chip${cls}" data-star="${i}">${esc(a.text)}<small>${Number(a.effective_distance).toFixed(2)}</small></button>`;
  }).join("");

  // 同意分享
  if (shared) {
    // 重看已分享的舊紀錄：不可再送
    $("#consentCheck").checked = true;
    $("#consentCheck").disabled = true;
    $("#consentBtn").disabled = true;
    $("#consentBtn").hidden = true;
    $("#consentBox").classList.add("done");
    setMsg($("#consentMsg"), "已加入星圖", "ok");
  } else {
    $("#consentCheck").checked = false;
    $("#consentCheck").disabled = false;
    $("#consentBtn").disabled = true;
    $("#consentBtn").hidden = false;
    $("#consentBox").classList.remove("done");
    setMsg($("#consentMsg"), "");
  }

  // 3D 星圖
  disposeMap();
  const mapEl = $("#map");
  mapEl.querySelector(".map-fallback")?.remove();
  try {
    const { createStarMap, webglAvailable } = await import("/static/scene.js");
    if (!webglAvailable()) throw new Error("no webgl");
    if (state.result !== data) return; // 使用者已離開
    state.map = createStarMap(mapEl, data, {
      onSelect(i) {
        document.querySelectorAll("#myItems .chip").forEach((c) => c.classList.toggle("active", +c.dataset.star === i));
      },
    });
  } catch (e) {
    console.error(e);
    const fb = document.createElement("div");
    fb.className = "map-fallback";
    fb.textContent = "這個瀏覽器無法顯示 3D 星圖（需要 WebGL）。右側的指紋與解讀仍然有效。";
    mapEl.appendChild(fb);
  }
}

function disposeMap() {
  if (state.map) {
    try { state.map.dispose(); } catch (e) { console.error(e); }
    state.map = null;
  }
}

async function onConsent() {
  if (!$("#consentCheck").checked || !state.result || state.resultShared) return;
  const btn = $("#consentBtn");
  btn.disabled = true;
  setMsg($("#consentMsg"), `<span class="spinner"></span>寫入中…`);
  try {
    const r = await api.consent(state.result.submission_id);
    if (r.shared) {
      state.resultShared = true;
      const h = (state.history || []).find((x) => x.submission_id === state.result.submission_id);
      if (h) h.shared = true;
      $("#consentBox").classList.add("done");
      $("#consentCheck").disabled = true;
      btn.hidden = true;
      setMsg($("#consentMsg"), esc(r.message || "已匿名加入星圖，謝謝你！"), "ok");
    } else {
      setMsg($("#consentMsg"), esc(r.message || "沒有寫入。"), "err");
      btn.disabled = false;
    }
  } catch (e) {
    setMsg($("#consentMsg"), esc(e.message), "err");
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 事件綁定
// ---------------------------------------------------------------------------
function bind() {
  document.addEventListener("click", (ev) => {
    const home = ev.target.closest("[data-action='home']");
    if (home) {
      ev.preventDefault();
      if (document.body.dataset.view === "answer" && state.items.length && !state.timeUp &&
          !confirm("離開後這次寫的聯想會消失，確定要換一個詞嗎？")) return;
      goHome();
      return;
    }
    const stim = ev.target.closest("[data-stim]");
    if (stim && !stim.disabled) {
      const s = state.stimuli.find((x) => x.id === stim.dataset.stim);
      if (s?.ready) startAnswer(s);
      return;
    }
    const pub = ev.target.closest("[data-publish]");
    if (pub) { openPublish(pub.dataset.publish); return; }
    const del = ev.target.closest("[data-del]");
    if (del) { openDelete(del.dataset.del); return; }
    const sub = ev.target.closest("[data-sub]");
    if (sub) { openSubmission(sub.dataset.sub, sub); return; }
    const star = ev.target.closest("[data-star]");
    if (star && state.map) state.map.select(+star.dataset.star);
  });

  $("#addForm").addEventListener("submit", onAddStimulus);
  $("#loginForm").addEventListener("submit", onLogin);
  $("#loginPin").addEventListener("input", (ev) => {
    const v = ev.target.value.replace(/\D/g, "").slice(0, 4);
    if (v !== ev.target.value) ev.target.value = v;
  });
  $("#switchBtn").addEventListener("click", switchIdentity);

  const input = $("#answerInput");
  input.addEventListener("input", updateCounter);
  input.addEventListener("keydown", (ev) => {
    // 中文輸入法選字時的 Enter 不算送出
    if (ev.key !== "Enter" || ev.isComposing || ev.keyCode === 229) return;
    ev.preventDefault();
    if (addItem(input.value)) {
      input.value = "";
      updateCounter();
    }
  });
  $("#items").addEventListener("click", onItemsClick);
  $("#items").addEventListener("input", onItemsInput);
  $("#items").addEventListener("keydown", (ev) => {
    if (ev.target.dataset.note !== undefined && ev.key === "Enter" && !ev.isComposing && ev.keyCode !== 229) {
      ev.preventDefault();
      state.items[+ev.target.dataset.note].open = false;
      renderItems();
      input.focus();
    }
  });
  $("#submitBtn").addEventListener("click", () => {
    const pending = input.value.trim();
    if (pending && addItem(pending)) { input.value = ""; updateCounter(); }
    submit();
  });
  $("#backToAnswerBtn").addEventListener("click", backToAnswer);

  $("#consentCheck").addEventListener("change", (ev) => { $("#consentBtn").disabled = !ev.target.checked; });
  $("#consentBtn").addEventListener("click", onConsent);
  $("#againBtn").addEventListener("click", goHome);
  $("#reanalyzeBtn").addEventListener("click", reanalyze);
  $("#deleteCancel").addEventListener("click", closeDelete);
  $("#deleteConfirm").addEventListener("click", confirmDelete);
  $("#deleteModal").addEventListener("click", (ev) => { if (ev.target.id === "deleteModal") closeDelete(); });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("#deleteModal").hidden) closeDelete(); });

  $("#publishCheck").addEventListener("change", (ev) => { $("#publishOk").disabled = !ev.target.checked; });
  $("#publishOk").addEventListener("click", confirmPublish);
  $("#publishCancel").addEventListener("click", closePublish);
  $("#publishModal").addEventListener("click", (ev) => { if (ev.target.id === "publishModal") closePublish(); });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("#publishModal").hidden) closePublish(); });
}

async function init() {
  $("#mockBadge").hidden = !MOCK;
  bind();
  const configP = api.config().then((cfg) => {
    if (cfg && typeof cfg.answer_seconds === "number" && cfg.answer_seconds > 0) state.config = { ...state.config, ...cfg };
  }).catch(() => { /* 用預設 180 秒 */ });

  const saved = getNickname();
  if (!saved) {
    showLogin();
    await configP;
    return;
  }
  // 有暱稱：先照本機狀態進首頁，再跟伺服器確認身分
  state.nickname = saved;
  $("#loginNick").value = saved;
  enterHome();
  try {
    const p = await api.player();
    if (!p?.nickname) {
      setNickname(null);
      state.nickname = null;
      showLogin();
      setMsg($("#loginMsg"), "身分已失效，請重新輸入暱稱和 PIN。", "err");
    } else if (p.nickname !== saved) {
      setNickname(p.nickname);
      state.nickname = p.nickname;
      renderWho();
    }
  } catch { /* 連不上時沿用本機暱稱 */ }
  await configP;
}

init();
