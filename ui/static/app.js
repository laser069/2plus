'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  messages: [],           // {role, content, citations, meta, ts}
  streaming: false,
  mode: localStorage.getItem('2plus_mode') || 'local',
  cloudModel: localStorage.getItem('2plus_cloud_model') || '',
  think: localStorage.getItem('2plus_think') === '1',
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const msgList      = $('messages');
const sessionList  = $('session-list');
const chatInput    = $('chat-input');
const btnSend      = $('btn-send');
const btnNewChat   = $('btn-new-chat');
const btnExport    = $('btn-export');
const selectModel  = $('select-model');
const inputCloud   = $('input-cloud-model');
const toggleThink  = $('toggle-think');
const modeBadge    = $('mode-badge');
const modeBtns     = document.querySelectorAll('.mode-btn');
const localWrap    = $('local-model-wrap');
const cloudWrap    = $('cloud-model-wrap');
const fileUpload   = $('file-upload');
const docsList     = $('docs-list');
const factsList    = $('facts-list');
const btnMenu      = $('btn-menu');
const sidebar      = $('sidebar');
const sidebarBackdrop = $('sidebar-backdrop');

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  // Restore UI prefs
  toggleThink.checked = state.think;
  inputCloud.value = state.cloudModel;
  setMode(state.mode, false);

  await loadModels();
  await loadSessions();
  await loadDocs();
  await loadFacts();

  // Start with most recent session or new
  const sessions = await fetchJSON('/sessions');
  if (sessions.length > 0) {
    await switchSession(sessions[0].session_id);
  } else {
    await newChat();
  }
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function fetchJSON(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Sessions ──────────────────────────────────────────────────────────────────
async function loadSessions() {
  const sessions = await fetchJSON('/sessions');
  renderSessionList(sessions);
}

function renderSessionList(sessions) {
  sessionList.innerHTML = '';
  for (const s of sessions) {
    const li = document.createElement('li');
    li.dataset.sid = s.session_id;
    if (s.session_id === state.sessionId) li.classList.add('active');

    const titleEl = document.createElement('span');
    titleEl.className = 'session-title';
    titleEl.textContent = s.title || 'New Chat';
    titleEl.title = s.title || 'New Chat';

    const tsEl = document.createElement('span');
    tsEl.className = 'session-ts';
    tsEl.textContent = fmtDate(s.ts);

    const delBtn = document.createElement('button');
    delBtn.className = 'btn-del-session';
    delBtn.title = 'Delete chat';
    delBtn.textContent = '✕';
    delBtn.onclick = async e => {
      e.stopPropagation();
      await deleteSession(s.session_id);
    };

    li.append(titleEl, tsEl, delBtn);
    li.onclick = () => switchSession(s.session_id);
    sessionList.appendChild(li);
  }
}

async function switchSession(sid) {
  state.sessionId = sid;
  const msgs = await fetchJSON(`/sessions/${sid}/messages`);
  state.messages = msgs;
  renderMessages();
  highlightActiveSession();
  scrollBottom();
}

async function newChat() {
  const data = await fetchJSON('/sessions', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
  state.sessionId = data.session_id;
  state.messages = [];
  msgList.innerHTML = '';
  await loadSessions();
  highlightActiveSession();
}

async function deleteSession(sid) {
  await fetch(`/sessions/${sid}`, { method: 'DELETE' });
  if (sid === state.sessionId) {
    await newChat();
  } else {
    await loadSessions();
  }
}

function highlightActiveSession() {
  sessionList.querySelectorAll('li').forEach(li => {
    li.classList.toggle('active', li.dataset.sid === state.sessionId);
  });
}

// ── Message rendering ─────────────────────────────────────────────────────────
function renderMessages() {
  msgList.innerHTML = '';
  for (const msg of state.messages) {
    if (msg.role === 'user' || msg.role === 'assistant') {
      appendMessage(msg);
    }
  }
}

function appendMessage(msg) {
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap ${msg.role}`;

  const bubble = document.createElement('div');
  bubble.className = 'msg';
  bubble.innerHTML = renderMarkdown(msg.content || '');
  wrap.appendChild(bubble);

  // Meta row: timestamp + actions
  const meta = document.createElement('div');
  meta.className = 'msg-meta';

  const ts = document.createElement('time');
  ts.textContent = msg.ts ? fmtTime(msg.ts) : '';
  meta.appendChild(ts);

  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  const copyBtn = document.createElement('button');
  copyBtn.className = 'msg-action-btn';
  copyBtn.textContent = 'Copy';
  copyBtn.onclick = () => copyText(msg.content || '');
  actions.appendChild(copyBtn);

  if (msg.role === 'assistant') {
    const regenBtn = document.createElement('button');
    regenBtn.className = 'msg-action-btn';
    regenBtn.textContent = 'Regen';
    regenBtn.onclick = () => regenerate(msg);
    actions.appendChild(regenBtn);
  }

  meta.appendChild(actions);
  wrap.appendChild(meta);

  // Sources
  if (msg.citations && msg.citations.length > 0) {
    const src = document.createElement('div');
    src.className = 'msg-sources';
    src.textContent = 'Sources: ' + msg.citations.join(' · ');
    wrap.appendChild(src);
  }

  msgList.appendChild(wrap);
  return wrap;
}

// Streaming message: returns live-update handle
function appendStreamingMessage() {
  const msg = { role: 'assistant', content: '', ts: new Date().toISOString() };

  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap assistant';
  wrap.id = 'streaming-msg';

  const bubble = document.createElement('div');
  bubble.className = 'msg';
  wrap.appendChild(bubble);

  const status = document.createElement('div');
  status.className = 'msg-status';
  status.id = 'stream-status';
  wrap.appendChild(status);

  msgList.appendChild(wrap);
  scrollBottom();

  return {
    appendToken(text) {
      msg.content += text;
      bubble.innerHTML = renderMarkdown(msg.content) + '<span class="cursor">▌</span>';
      scrollBottom();
    },
    setStatus(text) {
      status.textContent = text;
    },
    finalize(citations) {
      msg.citations = citations;
      wrap.id = '';
      bubble.innerHTML = renderMarkdown(msg.content);

      // Add meta row
      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      const ts = document.createElement('time');
      ts.textContent = fmtTime(msg.ts);
      meta.appendChild(ts);
      const actions = document.createElement('div');
      actions.className = 'msg-actions';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'msg-action-btn';
      copyBtn.textContent = 'Copy';
      copyBtn.onclick = () => copyText(msg.content);
      actions.appendChild(copyBtn);
      const regenBtn = document.createElement('button');
      regenBtn.className = 'msg-action-btn';
      regenBtn.textContent = 'Regen';
      regenBtn.onclick = () => regenLastUser();
      actions.appendChild(regenBtn);
      meta.appendChild(actions);
      wrap.insertBefore(meta, status);
      status.remove();

      if (citations && citations.length > 0) {
        const src = document.createElement('div');
        src.className = 'msg-sources';
        src.textContent = 'Sources: ' + citations.join(' · ');
        wrap.appendChild(src);
      }

      state.messages.push(msg);
    },
  };
}

// ── Chat send ─────────────────────────────────────────────────────────────────
async function sendMessage(query) {
  if (!query.trim() || !state.sessionId) return;

  const ts = new Date().toISOString();
  const userMsg = { role: 'user', content: query, ts };
  state.messages.push(userMsg);
  appendMessage(userMsg);
  scrollBottom();

  // Update session list title if first message
  await loadSessions();
  highlightActiveSession();

  const handle = appendStreamingMessage();
  state.streaming = true;
  updateSendBtn();

  const body = JSON.stringify({
    session_id: state.sessionId,
    query,
    model: resolveModel(),
    think: state.think,
    fallback_model: resolveFallback(),
  });

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if (!resp.ok) throw new Error(`${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const ev = JSON.parse(raw);
            handleSSEEvent(ev, handle);
          } catch { /* ignore parse error */ }
        }
      }
    }
  } catch (err) {
    handle.setStatus(`Error: ${err.message}`);
    handle.finalize([]);
  } finally {
    state.streaming = false;
    updateSendBtn();
  }
}

function handleSSEEvent(ev, handle) {
  if (ev.type === 'token') {
    handle.appendToken(ev.text);
  } else if (ev.type === 'routing') {
    const routes = ev.routes && ev.routes.length ? ev.routes.join(', ') : 'direct';
    handle.setStatus(`routing · ${routes}`);
  } else if (ev.type === 'tool') {
    handle.setStatus(`tool · ${ev.name} (step ${ev.step})`);
  } else if (ev.type === 'done') {
    handle.setStatus('');
    handle.finalize(ev.citations || []);
  } else if (ev.type === 'error') {
    handle.setStatus(`Error: ${ev.message}`);
    handle.finalize([]);
  }
}

// ── Regenerate ────────────────────────────────────────────────────────────────
function regenerate() {
  regenLastUser();
}

function regenLastUser() {
  const lastUser = [...state.messages].reverse().find(m => m.role === 'user');
  if (!lastUser) return;
  // Remove last assistant message from DOM if present
  const lastWrap = msgList.lastElementChild;
  if (lastWrap && lastWrap.classList.contains('assistant')) {
    lastWrap.remove();
    state.messages = state.messages.filter((_, i) => i !== state.messages.length - 1);
  }
  sendMessage(lastUser.content);
}

// ── Model resolution ──────────────────────────────────────────────────────────
function resolveModel() {
  const m = state.mode;
  if (m === 'local' || m === 'auto') return selectModel.value || '';
  if (m === 'cloud') return state.cloudModel || selectModel.value || '';
  return '';
}

function resolveFallback() {
  if (state.mode === 'auto') return state.cloudModel || null;
  return null;
}

// ── Mode switcher ─────────────────────────────────────────────────────────────
function setMode(mode, save = true) {
  state.mode = mode;
  if (save) localStorage.setItem('2plus_mode', mode);

  modeBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));

  const showLocal = mode === 'local' || mode === 'auto';
  const showCloud = mode === 'cloud' || mode === 'auto';
  localWrap.classList.toggle('hidden', !showLocal);
  cloudWrap.classList.toggle('hidden', !showCloud);

  updateModeBadge();
}

function updateModeBadge() {
  const icons = { local: '⬡', auto: '⟳', cloud: '☁' };
  const colors = { local: '#2dd4bf', auto: '#f59e0b', cloud: '#9d7af7' };
  const m = state.mode;
  const model = resolveModel().split('/').pop() || '';
  modeBadge.innerHTML =
    `<span style="color:${colors[m]}">${icons[m]} ${m.toUpperCase()}</span>` +
    (model ? `&nbsp;<span style="color:#888899;font-weight:500">${model}</span>` : '');
}

// ── Models ────────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await fetchJSON('/models');
    selectModel.innerHTML = '';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === data.default) opt.selected = true;
      selectModel.appendChild(opt);
    }
  } catch { /* offline */ }
  updateModeBadge();
}

// ── Docs ──────────────────────────────────────────────────────────────────────
async function loadDocs() {
  try {
    const data = await fetchJSON('/docs');
    docsList.textContent = data.docs.length ? data.docs.join('\n') : '';
  } catch { /* ignore */ }
}

// ── Facts ─────────────────────────────────────────────────────────────────────
async function loadFacts() {
  try {
    const facts = await fetchJSON('/facts');
    const entries = Object.entries(facts);
    if (entries.length === 0) {
      factsList.textContent = 'No facts yet.';
      factsList.className = 'dim';
    } else {
      factsList.innerHTML = entries.map(([k, v]) => `<b>${k}</b>: ${v}`).join('<br>');
      factsList.className = '';
    }
  } catch { /* ignore */ }
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportChat() {
  if (!state.messages.length) return;
  const lines = state.messages
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => `**${m.role === 'user' ? 'You' : '2Plus'}** _(${fmtTime(m.ts)})_\n\n${m.content}`)
    .join('\n\n---\n\n');
  const blob = new Blob([lines], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `chat-${state.sessionId}-${Date.now()}.md`;
  a.click();
}

// ── Copy ──────────────────────────────────────────────────────────────────────
function copyText(text) {
  navigator.clipboard.writeText(text).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function scrollBottom() {
  msgList.scrollTop = msgList.scrollHeight;
}

function updateSendBtn() {
  // Input stays enabled — just update button label
  btnSend.textContent = state.streaming ? '…' : 'Send';
}

function fmtDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 86400000) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch { return ''; }
}

function fmtTime(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function renderMarkdown(text) {
  // Minimal markdown: code blocks, inline code, bold, line breaks
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\*(.+?)\*/g, '<i>$1</i>')
    .replace(/\n/g, '<br>');
}

// ── Event listeners ───────────────────────────────────────────────────────────
btnNewChat.onclick = () => newChat();
btnExport.onclick = () => exportChat();
btnSend.onclick = () => {
  const q = chatInput.value.trim();
  if (!q) return;
  chatInput.value = '';
  autoResize();
  sendMessage(q);
};

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    btnSend.click();
  }
});

function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
}
chatInput.addEventListener('input', autoResize);

modeBtns.forEach(b => b.onclick = () => setMode(b.dataset.mode));

selectModel.onchange = () => updateModeBadge();

inputCloud.oninput = () => {
  state.cloudModel = inputCloud.value;
  localStorage.setItem('2plus_cloud_model', state.cloudModel);
  updateModeBadge();
};

toggleThink.onchange = () => {
  state.think = toggleThink.checked;
  localStorage.setItem('2plus_think', state.think ? '1' : '0');
};

fileUpload.onchange = async () => {
  for (const file of fileUpload.files) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/upload', { method: 'POST', body: fd });
      const d = await r.json();
      console.log('Ingested', d.filename, d.chunks, 'chunks');
    } catch (e) {
      console.error('Upload failed', e);
    }
  }
  fileUpload.value = '';
  await loadDocs();
};

// ── Mobile sidebar drawer ─────────────────────────────────────────────────────
function openSidebar() {
  sidebar.classList.add('open');
  sidebarBackdrop.classList.add('open');
}
function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarBackdrop.classList.remove('open');
}
btnMenu.onclick = () => sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
sidebarBackdrop.onclick = () => closeSidebar();

const _origSwitchSession = switchSession;
switchSession = async sid => { await _origSwitchSession(sid); closeSidebar(); };

// ── Bootstrap ─────────────────────────────────────────────────────────────────
init().catch(console.error);
