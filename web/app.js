/* AJP (Agent Jose Pereira) - web UI logic (vanilla JS, no framework).
 *
 * Talks to server.py:
 *   POST /api/chat          -> Server-Sent Events: session, status, sources, token, done, error
 *   GET  /api/sessions[/id] -> sidebar history
 *   GET  /api/health        -> engine status (every 10 s)
 *   GET  /api/stats         -> System tab (every 2 s while open)
 *   GET  /api/documents     -> Documents tab
 *   POST /api/upload        -> add a file to the knowledge base
 *   GET/POST/DELETE /api/agent/folders -> folders Agent mode may read
 *   (Agent mode adds two SSE events: step, step_result)
 */
(() => {
    'use strict';

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const el = (tag, className, html) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (html !== undefined) node.innerHTML = html;
        return node;
    };
    const escapeHtml = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const icon = (name, extra = '') => `<i class="ph ph-${name} ${extra}"></i>`;

    async function api(path, options = {}) {
        const response = await fetch(path, options);
        let data = null;
        try { data = await response.json(); } catch { /* empty body */ }
        if (!response.ok) {
            const detail = data && (typeof data.detail === 'string' ? data.detail
                : Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join('; ') : null);
            throw new Error(detail || `Request failed (${response.status})`);
        }
        return data;
    }

    function toast(message, kind = 'info', ms = 4000) {
        const icons = { ok: 'check-circle', error: 'warning-circle', info: 'info' };
        const node = el('div', `toast ${kind}`, `${icon(icons[kind] || 'info')}<span>${escapeHtml(message)}</span>`);
        $('toast-stack').appendChild(node);
        setTimeout(() => node.remove(), ms);
    }

    const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
    const hostOf = (url) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url || ''; } };

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    const state = {
        sessionId: null,
        mode: ['cowork', 'agent', 'email'].includes(localStorage.getItem('ajp.mode')) ? localStorage.getItem('ajp.mode') : 'chat',
        controller: null,          // AbortController of the running answer
        statsTimer: null,
        health: null,
    };

    const main = $('main');
    const conversation = $('conversation');
    const messagesEl = $('messages');
    const input = $('chat-input');
    const sendBtn = $('send-btn');

    // ------------------------------------------------------------------
    // Greeting
    // ------------------------------------------------------------------
    function setGreeting() {
        const hour = new Date().getHours();
        const part = hour < 5 ? 'Working late' : hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
        $('greeting').textContent = `${part}.`;
    }

    // ------------------------------------------------------------------
    // Sidebar
    // ------------------------------------------------------------------
    const sidebar = $('sidebar');
    function setSidebar(collapsed) {
        sidebar.classList.toggle('collapsed', collapsed);
        $('open-sidebar-btn').classList.toggle('hidden', !collapsed);
        localStorage.setItem('ajp.sidebar', collapsed ? 'collapsed' : 'open');
    }
    $('toggle-sidebar').addEventListener('click', () => setSidebar(true));
    $('open-sidebar-btn').addEventListener('click', () => setSidebar(false));
    setSidebar(localStorage.getItem('ajp.sidebar') === 'collapsed' || window.innerWidth < 760);

    // ------------------------------------------------------------------
    // Mode toggle (Chat = documents only, Cowork = documents + web, Agent = your folders)
    // ------------------------------------------------------------------
    const MODE_TEXT = {
        chat: {
            badge: 'Chat',
            hint: 'Chat mode: answers come only from your documents. Nothing leaves this computer.',
            sub: 'Ask anything about your documents. Answers cite their sources.',
        },
        cowork: {
            badge: 'Cowork',
            hint: 'Cowork mode: searches your documents and the live web. Only the search query leaves this computer.',
            sub: 'Research with your documents and the web. Every claim is cited.',
        },
        agent: {
            badge: 'Agent',
            hint: 'Agent mode: explores the folders you allow (read-only). Nothing leaves this computer.',
            sub: 'Point me at a folder or codebase and ask anything about it.',
        },
        email: {
            badge: 'Email',
            hint: 'Email mode: read, search, and send emails from your connected Gmail account.',
            sub: 'Check your inbox, search emails, send messages, and manage your Gmail.',
        },
    };
    function setMode(mode) {
        state.mode = mode;
        localStorage.setItem('ajp.mode', mode);
        ['chat', 'cowork', 'agent', 'email'].forEach((m) => {
            $(`mode-${m}`).classList.toggle('active', m === mode);
            $(`mode-${m}`).setAttribute('aria-selected', String(m === mode));
        });
        $('composer').classList.toggle('cowork', mode === 'cowork');
        $('composer').classList.toggle('agent', mode === 'agent');
        $('composer').classList.toggle('email', mode === 'email');
        $('top-badge').classList.toggle('cowork', mode === 'cowork');
        $('top-badge').classList.toggle('agent', mode === 'agent');
        $('top-badge').classList.toggle('email', mode === 'email');
        $('agent-folders').classList.toggle('hidden', mode !== 'agent');
        $('badge-mode').textContent = MODE_TEXT[mode].badge;
        $('composer-hint').textContent = MODE_TEXT[mode].hint;
        $('welcome-sub').textContent = MODE_TEXT[mode].sub;
        input.placeholder = mode === 'agent' ? 'e.g. Explain the structure of this project'
                          : mode === 'email' ? 'e.g. Check my inbox / Send an email to ...'
                          : 'How can I help you today?';
        if (mode === 'agent') loadFolders();
    }
    $('mode-chat').addEventListener('click', () => setMode('chat'));
    $('mode-cowork').addEventListener('click', () => setMode('cowork'));
    $('mode-agent').addEventListener('click', () => setMode('agent'));
    $('mode-email').addEventListener('click', () => setMode('email'));

    // ------------------------------------------------------------------
    // Agent folders (the sandbox: the agent can only read inside these)
    // ------------------------------------------------------------------
    const folderForm = $('folder-add-form');
    const folderInput = $('folder-input');

    function renderFolders(folders) {
        const chips = $('folder-chips');
        chips.innerHTML = '';
        if (!folders.length) {
            chips.appendChild(el('span', 'folder-empty', 'None yet. Add a folder the agent may read.'));
            return;
        }
        folders.forEach((path) => {
            const name = path.split(/[\\/]/).filter(Boolean).pop() || path;
            const chip = el('span', 'folder-chip', `${icon('folder')}<span>${escapeHtml(name)}</span>`);
            chip.title = path;
            const del = el('button', 'folder-chip-x', icon('x'));
            del.type = 'button';
            del.title = `Remove ${path}`;
            del.setAttribute('aria-label', `Remove folder ${path}`);
            del.addEventListener('click', () => removeFolder(path));
            chip.appendChild(del);
            chips.appendChild(chip);
        });
    }
    async function loadFolders() {
        try { renderFolders((await api('/api/agent/folders')).folders); } catch { /* offline */ }
    }
    async function removeFolder(path) {
        try {
            const data = await api('/api/agent/folders', {
                method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
            });
            renderFolders(data.folders);
        } catch (err) { toast(err.message, 'error'); }
    }
    function showFolderForm(show) {
        folderForm.classList.toggle('hidden', !show);
        $('folder-add-btn').classList.toggle('hidden', show);
        if (show) { folderInput.value = ''; folderInput.focus(); }
    }
    $('folder-add-btn').addEventListener('click', () => showFolderForm(true));
    folderInput.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); showFolderForm(false); } });
    folderInput.addEventListener('blur', () => setTimeout(() => { if (!folderInput.value.trim()) showFolderForm(false); }, 150));
    folderForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const path = folderInput.value.trim();
        if (!path) return;
        try {
            const data = await api('/api/agent/folders', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
            });
            renderFolders(data.folders);
            showFolderForm(false);
            toast(`Agent can now read ${data.folder}`, 'ok');
            input.focus();
        } catch (err) { toast(err.message, 'error', 6000); }
    });

    // ------------------------------------------------------------------
    // Composer
    // ------------------------------------------------------------------
    function autoResize() {
        input.style.height = 'auto';
        input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
    }
    function updateSendButton() {
        const streaming = Boolean(state.controller);
        sendBtn.classList.toggle('stop', streaming);
        sendBtn.innerHTML = streaming ? icon('stop') : icon('arrow-up');
        sendBtn.title = streaming ? 'Stop generating' : 'Send (Enter)';
        sendBtn.setAttribute('aria-label', streaming ? 'Stop generating' : 'Send message');
        sendBtn.disabled = !streaming && !input.value.trim();
    }
    input.addEventListener('input', () => { autoResize(); updateSendButton(); });
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            if (!state.controller) sendMessage();
        }
    });
    $('composer').addEventListener('submit', (e) => {
        e.preventDefault();
        if (state.controller) stopGenerating(); else sendMessage();
    });

    // ------------------------------------------------------------------
    // Scrolling: follow the answer only if the user is already at the bottom
    // ------------------------------------------------------------------
    let stickToBottom = true;
    conversation.addEventListener('scroll', () => {
        stickToBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 80;
    });
    const scrollToBottom = (force = false) => {
        if (force || stickToBottom) conversation.scrollTop = conversation.scrollHeight;
    };

    // ------------------------------------------------------------------
    // Markdown + citations
    // ------------------------------------------------------------------
    if (window.marked) window.marked.setOptions({ gfm: true, breaks: true });

    function markdownToHtml(text) {
        if (window.marked && window.DOMPurify) {
            return window.DOMPurify.sanitize(window.marked.parse(text), { ADD_ATTR: ['target'] });
        }
        return `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;   // offline fallback
    }

    const CITE_RE = /\[(\d{1,2}(?:\s*[,\u2013-]\s*\d{1,2})*)\]/g;

    function expandCitation(group) {
        const numbers = [];
        group.split(',').forEach((part) => {
            const [a, b] = part.split(/[\u2013-]/).map((s) => parseInt(s.trim(), 10));
            if (b && b >= a && b - a < 10) { for (let n = a; n <= b; n++) numbers.push(n); }
            else if (a) numbers.push(a);
        });
        return numbers;
    }

    /** Replace "[1]" / "[2, 3]" in text nodes with clickable chips (skipping code blocks). */
    function linkCitations(container, sources) {
        if (!sources || !sources.length) return;
        const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
            acceptNode: (node) => (node.parentElement.closest('code, pre, a') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
        });
        const targets = [];
        while (walker.nextNode()) if (CITE_RE.test(walker.currentNode.nodeValue)) targets.push(walker.currentNode);
        targets.forEach((node) => {
            const fragment = document.createDocumentFragment();
            let last = 0;
            const text = node.nodeValue;
            text.replace(CITE_RE, (match, group, offset) => {
                const numbers = expandCitation(group).filter((n) => n >= 1 && n <= sources.length);
                if (!numbers.length) return match;
                fragment.appendChild(document.createTextNode(text.slice(last, offset)));
                numbers.forEach((n) => {
                    const src = sources[n - 1];
                    const chip = el('button', `cite ${src.type === 'web' ? 'web' : ''}`, String(n));
                    chip.type = 'button';
                    chip.dataset.n = String(n);
                    chip.title = src.title + (src.page ? `, page ${src.page}` : '');
                    fragment.appendChild(chip);
                });
                last = offset + match.length;
                return match;
            });
            fragment.appendChild(document.createTextNode(text.slice(last)));
            node.replaceWith(fragment);
        });
    }

    // ------------------------------------------------------------------
    // Message rendering
    // ------------------------------------------------------------------
    function setEmpty(empty) {
        main.classList.toggle('is-empty', empty);
    }

    function addUserMessage(text) {
        const wrap = el('div', 'message user');
        const bubble = el('div', 'user-bubble');
        bubble.textContent = text;
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        return wrap;
    }

    function sourceFromSaved(n, r) {
        const meta = r.metadata || {};
        const web = meta.type === 'web';
        const section = meta.section ? String(meta.section).split(' > ').pop().slice(0, 90) : null;
        const score = r.rerank_score ?? r.similarity ?? r.keyword_score ?? null;
        return {
            n, type: web ? 'web' : 'document', title: meta.source || 'Untitled', url: web ? meta.url : null,
            page: meta.page || null, section, score, preview: (r.text || '').slice(0, 700),
        };
    }

    /** One assistant answer: avatar + status/answer + sources + footer. Returns a small controller. */
    const MODE_ICON = { chat: 'sparkle', cowork: 'globe-hemisphere-west', agent: 'robot', email: 'envelope' };
    const TOOL_ICON = {
        repo_overview: 'tree-structure', list_dir: 'folder-open', find_files: 'file-magnifying-glass',
        grep: 'magnifying-glass', read_file: 'file-text', disk_usage: 'hard-drives', list_allowed_folders: 'folders',
        find_temp_junk: 'trash', find_large_files: 'file-arrow-up', find_duplicates: 'copy',
        get_inbox_summary: 'envelope', list_emails: 'list-bullets', read_email: 'envelope-open',
        search_emails: 'magnifying-glass', send_email: 'paper-plane-tilt', reply_email: 'arrow-bend-up-left',
        mark_read: 'check',
    };
    const TOOL_LABEL = {
        repo_overview: 'Project overview', list_dir: 'List folder', find_files: 'Find files', grep: 'Search code',
        read_file: 'Read file', disk_usage: 'Disk usage', list_allowed_folders: 'Allowed folders',
        find_temp_junk: 'Scan junk/temp', find_large_files: 'Large files', find_duplicates: 'Duplicates',
        get_inbox_summary: 'Inbox summary', list_emails: 'List emails', read_email: 'Read email',
        search_emails: 'Search emails', send_email: 'Send email', reply_email: 'Reply', mark_read: 'Mark read',
    };

    function createAssistantMessage(mode) {
        const wrap = el('div', `message assistant ${mode === 'chat' ? '' : mode}`);
        wrap.appendChild(el('div', 'assistant-avatar', icon(MODE_ICON[mode] || 'sparkle')));
        const body = el('div', 'assistant-body');
        const noticeSlot = el('div');
        const stepsEl = el('div', 'agent-steps hidden');
        const status = el('div', 'status-line', `${icon('circle-notch', 'spin')}<span>Thinking...</span>`);
        const answer = el('div', 'answer');
        const sourcesSlot = el('div');
        const footer = el('div', 'answer-footer');
        body.append(noticeSlot, stepsEl, status, answer, sourcesSlot, footer);
        wrap.appendChild(body);
        messagesEl.appendChild(wrap);

        let raw = '';
        let sources = [];
        let frame = 0;

        const render = (streaming) => {
            answer.classList.toggle('streaming', streaming);
            answer.innerHTML = markdownToHtml(raw);
            linkCitations(answer, sources);
            scrollToBottom();
        };

        const msg = {
            el: wrap,
            setStatus(text) {
                status.classList.remove('hidden');
                status.querySelector('span').textContent = text;
            },
            setNotice(text) {
                noticeSlot.innerHTML = text ? `<div class="notice">${icon('info')}${escapeHtml(text)}</div>` : '';
            },
            setSources(list) {
                sources = list || [];
                renderSources(sourcesSlot, sources, wrap);
            },
            append(piece) {
                if (!raw) status.classList.add('hidden');
                raw += piece;
                if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(true); });
            },
            finish(stats, opts = {}) {
                if (frame) { cancelAnimationFrame(frame); frame = 0; }
                status.classList.add('hidden');
                if (!raw && opts.stopped) raw = '_Stopped._';
                render(false);
                renderFooter(footer, raw, stats, mode);
            },
            fail(message) {
                if (frame) { cancelAnimationFrame(frame); frame = 0; }
                status.classList.add('hidden');
                answer.classList.remove('streaming');
                if (raw) { render(false); answer.insertAdjacentHTML('beforeend', `<p class="answer error">${icon('warning-circle')}<span>${escapeHtml(message)}</span></p>`); }
                else { answer.className = 'answer error'; answer.innerHTML = `${icon('warning-circle')}<span>${escapeHtml(message)}</span>`; }
            },
            setRaw(text) { raw = text; },
            /** Agent mode: one tool call appears in the timeline (spinner until its result arrives). */
            addStep(step) {
                stepsEl.classList.remove('hidden');
                const item = el('div', 'agent-step running');
                item.dataset.n = String(step.n);
                const head = el('button', 'agent-step-head',
                    `${icon(TOOL_ICON[step.tool] || 'wrench')}<span class="agent-step-name">${escapeHtml(TOOL_LABEL[step.tool] || step.tool)}</span>`
                    + `<span class="agent-step-args">${escapeHtml(step.label || '')}</span>`
                    + `<span class="agent-step-state">${icon('circle-notch', 'spin')}</span>`);
                head.type = 'button';
                head.addEventListener('click', () => item.classList.toggle('open'));
                item.append(head, el('pre', 'agent-step-output'));
                stepsEl.appendChild(item);
                scrollToBottom();
            },
            stepResult(result) {
                const item = stepsEl.querySelector(`.agent-step[data-n="${result.n}"]`);
                if (!item) return;
                item.classList.remove('running');
                item.classList.add(result.ok ? 'ok' : 'fail');
                item.querySelector('.agent-step-state').innerHTML = result.ok
                    ? `<span class="agent-step-ms">${result.ms != null ? formatMs(result.ms) : ''}</span>${icon('check')}`
                    : icon('warning-circle');
                item.querySelector('.agent-step-output').textContent = result.preview || '';
            },
        };

        // Clicking a citation chip opens the sources and highlights the card
        answer.addEventListener('click', (e) => {
            const chip = e.target.closest('.cite');
            if (chip) focusSource(wrap, Number(chip.dataset.n));
        });
        return msg;
    }

    function renderSources(slot, sources, messageEl) {
        slot.innerHTML = '';
        if (!sources.length) return;
        const docs = sources.filter((s) => s.type !== 'web').length;
        const webs = sources.length - docs;
        const parts = [];
        if (docs) parts.push(`${docs} document${docs > 1 ? 's' : ''}`);
        if (webs) parts.push(`${webs} web`);

        const block = el('div', 'sources');
        const toggle = el('button', 'sources-toggle', `${icon('caret-right', 'chev')}${icon('books')}<span>Sources &middot; ${parts.join(' + ')}</span>`);
        toggle.type = 'button';
        toggle.addEventListener('click', () => block.classList.toggle('open'));
        const list = el('div', 'source-list');

        sources.forEach((s) => {
            const web = s.type === 'web';
            const meta = web ? hostOf(s.url) : [s.page ? `p. ${s.page}` : null, s.section].filter(Boolean).join(' \u00b7 ');
            const card = el('button', `source-card ${web ? 'web' : ''}`,
                `<span class="source-num">${s.n}</span>${icon(web ? 'globe-simple' : 'file-text')}`
                + `<span class="source-title">${escapeHtml(s.title)}</span>`
                + (meta ? `<span class="source-meta">${escapeHtml(meta)}</span>` : ''));
            card.type = 'button';
            card.dataset.n = String(s.n);
            card.addEventListener('click', () => togglePreview(list, s, card));
            list.appendChild(card);
        });
        block.append(toggle, list);
        slot.appendChild(block);
        messageEl._sources = sources;
    }

    function togglePreview(list, source, card) {
        const existing = list.querySelector('.source-preview');
        const same = existing && existing.dataset.n === String(source.n);
        list.querySelectorAll('.source-card').forEach((c) => c.classList.remove('highlight'));
        if (existing) existing.remove();
        if (same) return;
        card.classList.add('highlight');
        const web = source.type === 'web';
        const preview = el('div', 'source-preview',
            `<header>${icon(web ? 'globe-simple' : 'file-text')}<span>[${source.n}] ${escapeHtml(source.title)}`
            + `${source.page ? `, page ${source.page}` : ''}</span>`
            + (web && source.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">Open ${icon('arrow-square-out')}</a>` : '')
            + `</header>${source.section ? `<div class="source-meta">${escapeHtml(source.section)}</div>` : ''}`
            + `<div class="preview-text"></div>`);
        preview.querySelector('.preview-text').textContent = source.preview || '';
        preview.dataset.n = String(source.n);
        card.after(preview);
    }

    function focusSource(messageEl, n) {
        const block = messageEl.querySelector('.sources');
        if (!block) return;
        block.classList.add('open');
        const card = block.querySelector(`.source-card[data-n="${n}"]`);
        const list = block.querySelector('.source-list');
        const open = list.querySelector('.source-preview');
        if (card && !(open && open.dataset.n === String(n))) togglePreview(list, messageEl._sources[n - 1], card);
        card?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function renderFooter(footer, raw, stats, mode) {
        footer.innerHTML = '';
        const copy = el('button', 'footer-btn', icon('copy'));
        copy.type = 'button';
        copy.title = 'Copy answer';
        copy.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(raw);
                copy.innerHTML = icon('check');
                setTimeout(() => { copy.innerHTML = icon('copy'); }, 1500);
            } catch { toast('Could not copy to clipboard', 'error'); }
        });
        footer.appendChild(copy);
        const bits = [MODE_TEXT[mode] ? MODE_TEXT[mode].badge : 'Chat'];
        if (stats) {
            if (stats.steps) bits.push(`${stats.steps} tool step${stats.steps > 1 ? 's' : ''}`);
            if (stats.tokens_per_second) bits.push(`${stats.tokens_per_second} tok/s`);
            if (stats.retrieval_ms != null) bits.push(`search ${formatMs(stats.retrieval_ms)}`);
            if (stats.total_ms != null) bits.push(`total ${formatMs(stats.total_ms)}`);
            if (stats.stopped) bits.push('stopped');
        }
        footer.appendChild(el('span', 'answer-stats', escapeHtml(bits.join(' \u00b7 '))));
    }

    // ------------------------------------------------------------------
    // Sending + streaming (Server-Sent Events over fetch)
    // ------------------------------------------------------------------
    async function sendMessage() {
        const text = input.value.trim();
        if (!text || state.controller) return;
        input.value = '';
        autoResize();

        setEmpty(false);
        addUserMessage(text);
        const mode = state.mode;
        const msg = createAssistantMessage(mode);
        scrollToBottom(true);
        stickToBottom = true;

        const controller = new AbortController();
        state.controller = controller;
        updateSendButton();
        let finished = false;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: state.sessionId, mode }),
                signal: controller.signal,
            });
            if (!response.ok) {
                let detail = `Request failed (${response.status})`;
                try { const data = await response.json(); if (typeof data.detail === 'string') detail = data.detail; } catch { /* ignore */ }
                throw new Error(detail);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let boundary;
                while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                    const block = buffer.slice(0, boundary);
                    buffer = buffer.slice(boundary + 2);
                    let event = 'message';
                    let data = '';
                    block.split('\n').forEach((line) => {
                        if (line.startsWith('event:')) event = line.slice(6).trim();
                        else if (line.startsWith('data:')) data += line.slice(5).trim();
                    });
                    if (!data) continue;
                    const payload = JSON.parse(data);
                    if (handleEvent(event, payload, msg)) finished = true;
                }
            }
            if (!finished) msg.finish(null);
        } catch (err) {
            if (err.name === 'AbortError') {
                msg.finish({ stopped: true }, { stopped: true });
            } else {
                msg.fail(err.message || 'Connection lost.');
            }
        } finally {
            state.controller = null;
            updateSendButton();
            input.focus();
        }
    }

    /** Returns true when the answer is complete (done or error). */
    function handleEvent(event, data, msg) {
        switch (event) {
            case 'session':
                if (state.sessionId !== data.id) {
                    state.sessionId = data.id;
                    $('topbar-title').textContent = data.title || '';
                }
                return false;
            case 'status':
                msg.setStatus(data.text);
                return false;
            case 'sources':
                msg.setSources(data.sources);
                msg.setNotice(data.notice);
                return false;
            case 'token':
                msg.append(data.text);
                return false;
            case 'step':
                msg.addStep(data);
                return false;
            case 'step_result':
                msg.stepResult(data);
                return false;
            case 'done':
                msg.finish(data);
                if (data.session) $('topbar-title').textContent = data.session.title;
                loadSessions();
                return true;
            case 'error':
                msg.fail(data.message);
                return true;
            default:
                return false;
        }
    }

    function stopGenerating() {
        if (state.controller) state.controller.abort();
    }

    // ------------------------------------------------------------------
    // Sessions (sidebar history)
    // ------------------------------------------------------------------
    async function loadSessions() {
        try {
            const { sessions } = await api('/api/sessions');
            const list = $('session-list');
            list.innerHTML = '';
            $('session-empty').classList.toggle('hidden', sessions.length > 0);
            sessions.forEach((s) => {
                const item = el('li', `session-item ${s.id === state.sessionId ? 'active' : ''}`);
                item.tabIndex = 0;
                item.title = s.title;
                item.innerHTML = `<span class="session-title">${escapeHtml(s.title)}</span>`;
                const del = el('button', 'session-delete', icon('trash'));
                del.type = 'button';
                del.title = 'Delete chat';
                del.setAttribute('aria-label', `Delete chat ${s.title}`);
                del.addEventListener('click', (e) => { e.stopPropagation(); deleteSession(s.id, s.title); });
                item.appendChild(del);
                item.addEventListener('click', () => openSession(s.id));
                item.addEventListener('keydown', (e) => { if (e.key === 'Enter') openSession(s.id); });
                list.appendChild(item);
            });
        } catch { /* server offline - the health indicator shows it */ }
    }

    async function openSession(id) {
        if (state.controller) stopGenerating();
        try {
            const session = await api(`/api/sessions/${id}`);
            state.sessionId = session.id;
            $('topbar-title').textContent = session.title;
            messagesEl.innerHTML = '';
            setEmpty(session.messages.length === 0);
            session.messages.forEach((m) => {
                if (m.role === 'user') { addUserMessage(m.content); return; }
                const msg = createAssistantMessage(m.mode || 'chat');
                (m.steps || []).forEach((s) => {
                    msg.addStep({ n: s.n, tool: s.tool, label: Object.entries(s.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ') });
                    msg.stepResult(s);
                });
                msg.setSources((m.sources || []).map((r, i) => sourceFromSaved(i + 1, r)));
                msg.setNotice(m.notice);
                msg.setRaw(m.content);
                msg.finish(m.stats);
            });
            scrollToBottom(true);
            loadSessions();
            if (window.innerWidth < 760) setSidebar(true);
        } catch (err) {
            toast(err.message, 'error');
        }
    }

    async function deleteSession(id, title) {
        if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
        try {
            await api(`/api/sessions/${id}`, { method: 'DELETE' });
            if (id === state.sessionId) newChat();
            loadSessions();
        } catch (err) { toast(err.message, 'error'); }
    }

    function newChat() {
        if (state.controller) stopGenerating();
        state.sessionId = null;
        messagesEl.innerHTML = '';
        $('topbar-title').textContent = '';
        setEmpty(true);
        setGreeting();
        loadSessions();
        input.focus();
    }
    $('new-chat-btn').addEventListener('click', newChat);

    // ------------------------------------------------------------------
    // Engine health (sidebar footer + composer model name)
    // ------------------------------------------------------------------
    async function refreshHealth() {
        const dot = $('status-dot');
        try {
            const h = await api('/api/health');
            state.health = h;
            $('model-name').textContent = h.chat_model;
            $('doc-count').textContent = h.chunks ? String(h.chunks) : '';
            if (!h.ollama_running) {
                dot.className = 'status-dot down';
                $('engine-name').textContent = 'Ollama is offline';
                $('engine-sub').textContent = 'Start the Ollama app';
            } else if (!h.chat_model_ready) {
                dot.className = 'status-dot warn';
                $('engine-name').textContent = `${h.chat_model} not installed`;
                $('engine-sub').textContent = `Run: ollama pull ${h.chat_model}`;
            } else if (h.warming_up) {
                dot.className = 'status-dot warn';
                $('engine-name').textContent = h.chat_model;
                $('engine-sub').textContent = 'Warming up search...';
            } else {
                dot.className = 'status-dot ok';
                $('engine-name').textContent = h.chat_model;
                $('engine-sub').textContent = h.chunks ? `Ready \u00b7 ${h.chunks} chunks indexed` : 'Ready \u00b7 no documents yet';
            }
        } catch {
            dot.className = 'status-dot down';
            $('engine-name').textContent = 'Server offline';
            $('engine-sub').textContent = 'Run: python server.py';
        }
    }

    // ------------------------------------------------------------------
    // Settings modal: Documents + System
    // ------------------------------------------------------------------
    const modal = $('settings-modal');

    function openModal(tab = 'documents') {
        modal.classList.remove('hidden');
        switchTab(tab);
    }
    function closeModal() {
        modal.classList.add('hidden');
        clearInterval(state.statsTimer);
        state.statsTimer = null;
    }
    function switchTab(tab) {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
        $('panel-documents').classList.toggle('hidden', tab !== 'documents');
        $('panel-system').classList.toggle('hidden', tab !== 'system');
        clearInterval(state.statsTimer);
        state.statsTimer = null;
        if (tab === 'documents') loadDocuments();
        if (tab === 'system') { loadStats(); state.statsTimer = setInterval(loadStats, 2000); }
    }

    $('open-settings-btn').addEventListener('click', () => openModal('system'));
    $('open-documents-btn').addEventListener('click', () => openModal('documents'));
    $('close-settings-btn').addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
    document.querySelectorAll('.tab-btn').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));

    async function loadDocuments() {
        const list = $('doc-list');
        try {
            const { documents, supported } = await api('/api/documents');
            fileInput.accept = supported.join(',');
            list.innerHTML = '';
            const total = documents.reduce((sum, d) => sum + d.chunks, 0);
            $('kb-summary').textContent = documents.length ? `${documents.length} files \u00b7 ${total} chunks` : '';
            if (!documents.length) {
                list.innerHTML = `<li class="doc-empty">No documents yet. Drop a file above to start.</li>`;
                return;
            }
            const iconFor = (name) => {
                const ext = name.split('.').pop().toLowerCase();
                if (ext === 'pdf') return 'file-pdf';
                if (ext === 'docx') return 'file-doc';
                if (['png', 'jpg', 'jpeg', 'webp'].includes(ext)) return 'file-image';
                if (['md', 'txt'].includes(ext)) return 'file-text';
                return 'file-code';
            };
            documents.forEach((d) => {
                list.appendChild(el('li', 'doc-item',
                    `${icon(iconFor(d.name))}<span class="name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>`
                    + `<span class="chunks">${d.chunks} chunks</span>`));
            });
        } catch (err) {
            list.innerHTML = `<li class="doc-empty">${escapeHtml(err.message)}</li>`;
        }
    }

    async function loadStats() {
        try {
            const s = await api('/api/stats');
            const gpu = s.gpu;
            if (gpu) {
                const used = gpu.memory_used_mb / 1024;
                const total = gpu.memory_total_mb / 1024;
                const pct = Math.min(100, (gpu.memory_used_mb / gpu.memory_total_mb) * 100);
                $('stat-vram').textContent = `${used.toFixed(1)} GB / ${total.toFixed(1)} GB`;
                $('stat-vram-bar').style.width = `${pct.toFixed(1)}%`;
                $('stat-gpu-name').textContent = gpu.name;
                $('stat-gpu-util').textContent = `${gpu.utilization}%`;
                $('stat-gpu-temp').textContent = `${gpu.temperature}\u00b0C`;
            } else {
                $('stat-vram').textContent = 'Not available';
                $('stat-vram-bar').style.width = '0%';
                $('stat-gpu-name').textContent = 'No NVIDIA GPU detected (nvidia-smi)';
                $('stat-gpu-util').textContent = '-';
                $('stat-gpu-temp').textContent = '';
            }
            const g = s.last_generation;
            $('stat-tps').textContent = g && g.tokens_per_second ? `${g.tokens_per_second} tok/s` : '-';
            $('stat-tps-foot').textContent = g ? `${g.tokens} tokens \u00b7 total ${formatMs(g.total_ms)}` : 'Ask a question to measure';
            $('stat-chunks').textContent = s.chunks.toLocaleString();
            $('stat-files').textContent = `${s.files} file${s.files === 1 ? '' : 's'}${s.ingesting ? ` \u00b7 indexing ${s.ingesting}` : ''}`;
            $('stat-ollama').textContent = s.ollama_running ? 'Running' : 'Offline';
            $('stat-loaded').textContent = s.loaded_models.length
                ? s.loaded_models.map((m) => `${m.name} (${(m.vram_mb / 1024).toFixed(1)} GB)`).join(', ')
                : 'No model in memory';

            const rows = [
                ['Chat model', s.chat_model], ['Embedding model', s.embed_model],
                ['Search mode', s.retrieval_mode], ['Re-ranker', s.reranker],
                ['Chunks per answer', s.top_k], ['Context window', `${s.context_window.toLocaleString()} tokens`],
            ];
            $('config-list').innerHTML = rows.map(([k, v]) =>
                `<div class="config-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`).join('');
        } catch {
            $('stat-ollama').textContent = 'Server offline';
        }
    }

    // ------------------------------------------------------------------
    // Uploads (drag & drop, click, or the paperclip in the composer)
    // ------------------------------------------------------------------
    const dropzone = $('dropzone');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.hidden = true;
    document.body.appendChild(fileInput);

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
    ['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }));
    dropzone.addEventListener('drop', (e) => uploadFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', () => { uploadFiles(fileInput.files); fileInput.value = ''; });

    $('attach-btn').addEventListener('click', () => { openModal('documents'); fileInput.click(); });

    // Dropping a file anywhere on the page opens the Documents tab
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.target.closest && e.target.closest('#dropzone')) return;
        if (e.dataTransfer?.files?.length) { openModal('documents'); uploadFiles(e.dataTransfer.files); }
    });

    let uploadChain = Promise.resolve();
    function uploadFiles(fileList) {
        const files = Array.from(fileList || []);
        files.forEach((file) => {
            const item = el('li', 'upload-item busy',
                `${icon('circle-notch', 'spin')}<span class="name">${escapeHtml(file.name)}</span><span class="state">Waiting...</span>`);
            $('upload-queue').prepend(item);
            // one at a time: ingestion is GPU-heavy, and the server indexes sequentially anyway
            uploadChain = uploadChain.then(() => uploadOne(file, item));
        });
    }

    async function uploadOne(file, item) {
        const stateEl = item.querySelector('.state');
        stateEl.textContent = 'Indexing...';
        const form = new FormData();
        form.append('file', file);
        try {
            const result = await api('/api/upload', { method: 'POST', body: form });
            item.className = 'upload-item ok';
            item.querySelector('i').className = 'ph ph-check-circle';
            stateEl.textContent = `${result.chunks} chunks`;
            if (modal.classList.contains('hidden')) toast(result.message, 'ok');
        } catch (err) {
            item.className = 'upload-item fail';
            item.querySelector('i').className = 'ph ph-warning-circle';
            stateEl.textContent = 'Failed';
            item.title = err.message;
            toast(`${file.name}: ${err.message}`, 'error', 6000);
        }
        loadDocuments();
        refreshHealth();
    }

    // ------------------------------------------------------------------
    // Keyboard shortcuts
    // ------------------------------------------------------------------
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!modal.classList.contains('hidden')) closeModal();
            else if (state.controller) stopGenerating();
        }
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); newChat(); }
    });

    // ------------------------------------------------------------------
    // Start
    // ------------------------------------------------------------------
    setGreeting();
    setMode(state.mode);
    updateSendButton();
    refreshHealth();
    loadSessions();
    setInterval(refreshHealth, 10000);
    input.focus();
})();
