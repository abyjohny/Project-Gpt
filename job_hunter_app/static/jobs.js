/* Job Hunter Standalone - Page logic (vanilla JS, no framework).
 *
 * Talks to app.py endpoints — same API shape as the original server.py
 * but all requests carry the session cookie automatically.
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
        const response = await fetch(path, { ...options, credentials: 'same-origin' });
        let data = null;
        try { data = await response.json(); } catch { /* empty body */ }
        if (response.status === 401) {
            // Not authenticated — redirect to login
            window.location.href = '/auth/login';
            return;
        }
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

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    const state = {
        jobs: [],
        selectedJobs: new Set(),
        drafts: [],
        cv: null,
        loading: false,
        presets: [],
    };

    // ------------------------------------------------------------------
    // Sidebar
    // ------------------------------------------------------------------
    const sidebar = $('sidebar');
    function setSidebar(collapsed) {
        sidebar.classList.toggle('collapsed', collapsed);
        $('open-sidebar-btn').classList.toggle('hidden', !collapsed);
        localStorage.setItem('jh.sidebar', collapsed ? 'collapsed' : 'open');
    }
    $('toggle-sidebar').addEventListener('click', () => setSidebar(true));
    $('open-sidebar-btn').addEventListener('click', () => setSidebar(false));
    setSidebar(localStorage.getItem('jh.sidebar') === 'collapsed' || window.innerWidth < 760);

    // ------------------------------------------------------------------
    // Navigation (Search / Presets / Runner / History)
    // ------------------------------------------------------------------
    const views = ['search', 'presets', 'runner', 'history'];
    function showView(view) {
        views.forEach((v) => {
            const viewEl = $(`view-${v}`);
            const navEl = $(`nav-${v}`);
            if (viewEl) viewEl.classList.toggle('hidden', v !== view);
            if (navEl) navEl.classList.toggle('active', v === view);
        });
        if (view === 'history') loadHistory();
        if (view === 'presets') loadPresets();
        if (view === 'runner') { loadRunnerConfig(); loadDailyResults(); loadGeminiStatus(); }
    }
    views.forEach((v) => {
        const navEl = $(`nav-${v}`);
        if (navEl) navEl.addEventListener('click', () => showView(v));
    });

    // ------------------------------------------------------------------
    // Config (load / save search preferences)
    // ------------------------------------------------------------------
    async function loadConfig() {
        try {
            const { config } = await api('/api/jobs/config');
            if (config) {
                if (config.role) $('job-role').value = config.role;
                if (config.country) $('job-country').value = config.country;
                if (config.state) $('job-state').value = config.state;
                if (config.experience) $('job-experience').value = config.experience;
                if (config.date_from) $('job-date-from').value = config.date_from;
                if (config.date_to) $('job-date-to').value = config.date_to;
            }
        } catch { /* first time, no config */ }
    }

    function getFormConfig() {
        return {
            role: $('job-role').value.trim(),
            country: $('job-country').value.trim(),
            state: $('job-state').value.trim(),
            experience: $('job-experience').value,
            date_from: $('job-date-from').value,
            date_to: $('job-date-to').value,
        };
    }

    // ------------------------------------------------------------------
    // Presets
    // ------------------------------------------------------------------
    async function loadPresets() {
        try {
            const { presets } = await api('/api/jobs/presets');
            state.presets = presets || [];
            renderPresets();
        } catch { /* no presets */ }
    }

    function renderPresets() {
        const grid = $('presets-grid');
        if (!state.presets.length) {
            grid.innerHTML = '<p class="empty-state">No presets saved yet. Use "Save as Preset" from the Search tab.</p>';
            return;
        }
        grid.innerHTML = '';
        state.presets.forEach((p) => {
            const card = el('div', `preset-card${p.active_for_runner ? ' runner-active' : ''}`);
            const exp = p.experience ? `<span class="preset-exp">${escapeHtml(p.experience)}</span>` : '';
            card.innerHTML = `
                <div class="preset-info">
                    <div class="preset-name">${icon('bookmark-simple')} ${escapeHtml(p.name)}</div>
                    <div class="preset-details">
                        ${icon('briefcase')} ${escapeHtml(p.role || '—')} · ${icon('map-pin')} ${escapeHtml(p.state ? `${p.state}, ${p.country}` : p.country || '—')}
                        ${exp}
                    </div>
                </div>
                <div class="preset-actions">
                    <button class="action-btn preset-run-btn" title="Load & Search" data-id="${p.id}">
                        ${icon('play')} Run
                    </button>
                    <label class="toggle-switch mini" title="Active for daily runner">
                        <input type="checkbox" class="preset-runner-toggle" data-id="${p.id}" ${p.active_for_runner ? 'checked' : ''}>
                        <span class="toggle-slider"></span>
                    </label>
                    <button class="action-btn danger preset-delete-btn" title="Delete" data-id="${p.id}">
                        ${icon('trash')}
                    </button>
                </div>
            `;
            grid.appendChild(card);
        });

        grid.querySelectorAll('.preset-run-btn').forEach((btn) => {
            btn.addEventListener('click', () => runPreset(btn.dataset.id));
        });
        grid.querySelectorAll('.preset-delete-btn').forEach((btn) => {
            btn.addEventListener('click', () => deletePreset(btn.dataset.id));
        });
        grid.querySelectorAll('.preset-runner-toggle').forEach((cb) => {
            cb.addEventListener('change', () => togglePresetRunner(cb.dataset.id, cb.checked));
        });
    }

    function runPreset(id) {
        const p = state.presets.find((pr) => pr.id === id);
        if (!p) return;
        $('job-role').value = p.role || '';
        $('job-country').value = p.country || '';
        $('job-state').value = p.state || '';
        $('job-experience').value = p.experience || '';
        $('job-date-from').value = p.date_from || '';
        $('job-date-to').value = p.date_to || '';
        showView('search');
        searchJobs();
    }

    async function deletePreset(id) {
        if (!confirm('Delete this preset?')) return;
        try {
            await api(`/api/jobs/presets/${id}`, { method: 'DELETE' });
            toast('Preset deleted', 'ok');
            loadPresets();
        } catch (err) { toast(err.message, 'error'); }
    }

    async function togglePresetRunner(id, active) {
        try {
            await api(`/api/jobs/presets/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active_for_runner: active }),
            });
            const p = state.presets.find((pr) => pr.id === id);
            if (p) p.active_for_runner = active;
            toast(active ? 'Preset activated for daily runner' : 'Preset deactivated', 'ok');
        } catch (err) { toast(err.message, 'error'); }
    }

    // Save preset modal
    $('save-preset-btn').addEventListener('click', () => {
        const cfg = getFormConfig();
        if (!cfg.role) { toast('Enter a role first', 'error'); return; }
        $('preset-name-input').value = `${cfg.role} - ${cfg.country || 'Global'}`;
        $('preset-name-modal').classList.remove('hidden');
        setTimeout(() => $('preset-name-input').focus(), 100);
    });

    $('close-preset-modal').addEventListener('click', () => $('preset-name-modal').classList.add('hidden'));
    $('cancel-preset-btn').addEventListener('click', () => $('preset-name-modal').classList.add('hidden'));
    $('preset-name-modal').addEventListener('click', (e) => { if (e.target === $('preset-name-modal')) $('preset-name-modal').classList.add('hidden'); });

    $('confirm-preset-btn').addEventListener('click', async () => {
        const name = $('preset-name-input').value.trim();
        if (!name) { toast('Enter a name', 'error'); return; }
        const cfg = getFormConfig();
        try {
            await api('/api/jobs/presets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, ...cfg }),
            });
            $('preset-name-modal').classList.add('hidden');
            toast('Preset saved!', 'ok');
            await api('/api/jobs/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: { ...cfg } }),
            });
            loadPresets();
        } catch (err) { toast(err.message, 'error'); }
    });

    // ------------------------------------------------------------------
    // Daily Runner
    // ------------------------------------------------------------------
    async function loadRunnerConfig() {
        try {
            const { runner } = await api('/api/jobs/runner');
            if (runner) {
                $('runner-enabled').checked = !!runner.enabled;
                $('runner-time').value = runner.run_time || '09:00';
                $('runner-min-score').value = runner.min_match_score ?? 50;
                $('runner-min-score-val').textContent = `${runner.min_match_score ?? 50}%`;
                $('runner-email').value = runner.to_email || '';
                $('runner-last-run').textContent = runner.last_run
                    ? new Date(runner.last_run).toLocaleString()
                    : 'Never';
                if (runner.enabled && runner.run_time) {
                    $('runner-next-run').textContent = `Today/Tomorrow at ${runner.run_time}`;
                } else {
                    $('runner-next-run').textContent = 'Disabled';
                }
            }
        } catch { /* no runner config */ }
    }

    $('runner-min-score').addEventListener('input', (e) => {
        $('runner-min-score-val').textContent = `${e.target.value}%`;
    });

    $('save-runner-btn').addEventListener('click', async () => {
        try {
            await api('/api/jobs/runner', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    enabled: $('runner-enabled').checked,
                    run_time: $('runner-time').value,
                    min_match_score: parseInt($('runner-min-score').value, 10),
                    to_email: $('runner-email').value.trim(),
                }),
            });
            toast('Runner configuration saved', 'ok');
            loadRunnerConfig();
        } catch (err) { toast(err.message, 'error'); }
    });

    $('trigger-runner-btn').addEventListener('click', async () => {
        if (!confirm('Run a job scan now for all active presets?')) return;
        $('trigger-runner-btn').disabled = true;
        $('trigger-runner-btn').innerHTML = `${icon('circle-notch', 'spin')} Running...`;
        try {
            const result = await api('/api/jobs/runner/trigger', { method: 'POST' });
            const r = result.results || {};
            toast(`Scan complete: ${r.total_jobs || 0} jobs found across ${r.presets_run || 0} presets`, 'ok', 6000);
            if (result.email && result.email.status === 'sent') {
                toast('Summary email sent!', 'ok');
            }
            loadRunnerConfig();
            loadDailyResults();
        } catch (err) {
            toast(`Scan failed: ${err.message}`, 'error', 6000);
        } finally {
            $('trigger-runner-btn').disabled = false;
            $('trigger-runner-btn').innerHTML = `${icon('play')} Run Now`;
        }
    });

    // ------------------------------------------------------------------
    // Gemini API Configuration
    // ------------------------------------------------------------------
    async function loadGeminiStatus() {
        try {
            const { configured, model } = await api('/api/jobs/gemini');
            const dot = $('gemini-dot');
            const text = $('gemini-status-text');
            if (configured) {
                dot.className = 'status-dot ok';
                text.textContent = `Connected — ${model}`;
            } else {
                dot.className = 'status-dot down';
                text.textContent = 'Not configured — set a Gemini API key to enable AI features';
            }
        } catch {
            $('gemini-dot').className = 'status-dot down';
            $('gemini-status-text').textContent = 'Could not check status';
        }
    }

    $('save-gemini-btn').addEventListener('click', async () => {
        const key = $('gemini-key').value.trim();
        if (!key) { toast('Enter your Gemini API key', 'error'); return; }
        try {
            await api('/api/jobs/gemini', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: key }),
            });
            $('gemini-key').value = '';
            toast('Gemini API key saved!', 'ok');
            loadGeminiStatus();
        } catch (err) { toast(err.message, 'error'); }
    });

    async function loadDailyResults() {
        try {
            const { results } = await api('/api/jobs/runner/results');
            const container = $('daily-results-content');
            if (!results || !results.results || !results.results.length) {
                container.innerHTML = '<p class="empty-state">No scan results yet. Click "Run Now" or wait for the scheduled time.</p>';
                return;
            }
            let html = `<div class="daily-summary-meta">
                <span>${icon('clock')} ${new Date(results.run_at).toLocaleString()}</span>
                <span>${icon('briefcase')} ${results.total_jobs} jobs found</span>
                <span>${icon('bookmarks')} ${results.presets_run} presets scanned</span>
            </div>`;
            results.results.forEach((pr) => {
                html += `<div class="daily-preset-result">
                    <h4>${icon('bookmark-simple')} ${escapeHtml(pr.preset_name)} <span class="daily-job-count">(${pr.job_count} jobs)</span></h4>`;
                if (pr.error) {
                    html += `<p class="error-text">${escapeHtml(pr.error)}</p>`;
                } else if (pr.jobs && pr.jobs.length) {
                    html += '<div class="daily-job-list">';
                    pr.jobs.slice(0, 10).forEach((j) => {
                        const score = j.match_score != null ? j.match_score : null;
                        const level = score != null ? (score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low') : '';
                        html += `<div class="daily-job-row">
                            <a href="${escapeHtml(j.url || '#')}" target="_blank" rel="noopener">${escapeHtml(j.title || 'Untitled')}</a>
                            <span class="daily-job-company">${escapeHtml(j.company || '')}</span>
                            ${score != null ? `<span class="score-badge ${level}">${score}%</span>` : ''}
                        </div>`;
                    });
                    if (pr.jobs.length > 10) {
                        html += `<p class="daily-more">...and ${pr.jobs.length - 10} more</p>`;
                    }
                    html += '</div>';
                } else {
                    html += '<p class="empty-state-sm">No jobs found for this preset.</p>';
                }
                html += '</div>';
            });
            container.innerHTML = html;
        } catch { /* no results */ }
    }

    // ------------------------------------------------------------------
    // CV upload
    // ------------------------------------------------------------------
    const cvDropzone = $('cv-dropzone');
    const cvFileInput = document.createElement('input');
    cvFileInput.type = 'file';
    cvFileInput.accept = '.pdf,.docx,.doc,.txt';
    cvFileInput.hidden = true;
    document.body.appendChild(cvFileInput);

    function updateCVStatus(cv) {
        state.cv = cv;
        const statusEl = $('cv-status');
        if (cv) {
            statusEl.className = 'cv-status uploaded';
            statusEl.innerHTML = `${icon('check-circle')} ${escapeHtml(cv.filename)}`;
        } else {
            statusEl.className = 'cv-status';
            statusEl.textContent = 'No CV uploaded';
        }
        updateActionButtons();
    }

    async function loadCV() {
        try {
            const { cv } = await api('/api/jobs/cv');
            updateCVStatus(cv);
        } catch { /* no CV */ }
    }

    async function uploadCV(file) {
        const form = new FormData();
        form.append('file', file);
        try {
            const result = await api('/api/jobs/cv', { method: 'POST', body: form });
            updateCVStatus({ filename: result.filename, path: result.path });
            toast(`CV uploaded: ${result.filename}`, 'ok');
        } catch (err) {
            toast(`CV upload failed: ${err.message}`, 'error', 6000);
        }
    }

    cvDropzone.addEventListener('click', () => cvFileInput.click());
    cvDropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); cvFileInput.click(); } });
    ['dragenter', 'dragover'].forEach((ev) => cvDropzone.addEventListener(ev, (e) => { e.preventDefault(); cvDropzone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach((ev) => cvDropzone.addEventListener(ev, (e) => { e.preventDefault(); cvDropzone.classList.remove('dragover'); }));
    cvDropzone.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) uploadCV(e.dataTransfer.files[0]); });
    cvFileInput.addEventListener('change', () => { if (cvFileInput.files.length) { uploadCV(cvFileInput.files[0]); cvFileInput.value = ''; } });

    // ------------------------------------------------------------------
    // Job search
    // ------------------------------------------------------------------
    async function searchJobs() {
        const role = $('job-role').value.trim();
        const country = $('job-country').value.trim();
        if (!role) { toast('Please enter a job role', 'error'); return; }
        if (!country) { toast('Please enter a country', 'error'); return; }

        setLoading(true, 'Searching for jobs...');
        try {
            const result = await api('/api/jobs/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    role,
                    country,
                    state: $('job-state').value.trim(),
                    date_from: $('job-date-from').value,
                    date_to: $('job-date-to').value,
                    experience: $('job-experience').value,
                }),
            });
            state.jobs = result.jobs || [];
            state.selectedJobs.clear();
            renderJobs();
            $('results-panel').classList.remove('hidden');
            toast(`Found ${state.jobs.length} jobs`, 'ok');
        } catch (err) {
            toast(`Search failed: ${err.message}`, 'error', 6000);
        } finally {
            setLoading(false);
        }
    }
    $('search-btn').addEventListener('click', searchJobs);

    // ------------------------------------------------------------------
    // Job rendering
    // ------------------------------------------------------------------
    function renderJobs() {
        const list = $('job-list');
        list.innerHTML = '';
        $('result-count').textContent = `${state.jobs.length} job${state.jobs.length !== 1 ? 's' : ''} found`;

        if (!state.jobs.length) {
            list.innerHTML = '<p class="empty-state">No jobs found. Try different search terms.</p>';
            return;
        }

        state.jobs.forEach((job, i) => {
            const card = el('div', `job-card${state.selectedJobs.has(i) ? ' selected' : ''}`);
            card.dataset.index = String(i);

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = state.selectedJobs.has(i);
            cb.addEventListener('change', () => toggleJob(i, cb.checked));

            const info = el('div', 'job-info');
            const titleLink = job.url
                ? `<a href="${escapeHtml(job.url)}" target="_blank" rel="noopener" title="Open posting">${escapeHtml(job.title || 'Untitled')}</a> ${icon('arrow-square-out')}`
                : escapeHtml(job.title || 'Untitled');
            info.innerHTML = `
                <div class="job-title">${titleLink}</div>
                <div class="job-company">${escapeHtml(job.company || 'Unknown company')}</div>
                <div class="job-location">${icon('map-pin')} ${escapeHtml(job.location || 'Location not specified')}</div>
                <div class="job-snippet">${escapeHtml(job.snippet || '')}</div>
                ${job.contact_email ? `<div class="job-contact">${icon('envelope')} ${escapeHtml(job.contact_email)}</div>` : ''}
                ${job.match_reason ? `<div class="job-match-reason">${escapeHtml(job.match_reason)}</div>` : ''}
            `;

            const meta = el('div', 'job-meta');
            if (job.match_score != null) {
                const level = job.match_score >= 70 ? 'high' : job.match_score >= 40 ? 'medium' : 'low';
                meta.innerHTML += `<span class="score-badge ${level}">${job.match_score}%</span>`;
            }
            if (job.posting_type) {
                const isDirect = job.posting_type === 'direct';
                meta.innerHTML += `<span class="posting-badge ${isDirect ? 'direct' : 'aggregator'}">${isDirect ? 'Direct' : 'Aggregator'}</span>`;
            }
            if (job.has_company === false) {
                meta.innerHTML += `<span class="posting-badge no-company" title="No verified company info">${icon('warning')} No Company</span>`;
            }
            if (job.source) {
                meta.innerHTML += `<span class="job-source">${escapeHtml(job.source)}</span>`;
            }

            card.append(cb, info, meta);
            list.appendChild(card);
        });

        updateActionButtons();
    }

    function toggleJob(index, checked) {
        if (checked) state.selectedJobs.add(index);
        else state.selectedJobs.delete(index);
        const card = $('job-list').querySelector(`[data-index="${index}"]`);
        if (card) card.classList.toggle('selected', checked);
        $('select-all').checked = state.selectedJobs.size === state.jobs.length && state.jobs.length > 0;
        updateActionButtons();
    }

    $('select-all').addEventListener('change', (e) => {
        const checked = e.target.checked;
        state.jobs.forEach((_, i) => {
            if (checked) state.selectedJobs.add(i);
            else state.selectedJobs.delete(i);
        });
        $('job-list').querySelectorAll('input[type="checkbox"]').forEach((cb) => { cb.checked = checked; });
        $('job-list').querySelectorAll('.job-card').forEach((card) => { card.classList.toggle('selected', checked); });
        updateActionButtons();
    });

    function updateActionButtons() {
        const hasSelection = state.selectedJobs.size > 0;
        const hasCV = !!state.cv;
        const analyzeBtn = $('analyze-btn');
        const draftBtn = $('draft-btn');
        const sendBtn = $('send-btn');
        if (analyzeBtn) analyzeBtn.disabled = !(hasSelection && hasCV);
        if (draftBtn) draftBtn.disabled = !(hasSelection && hasCV);
        if (sendBtn) sendBtn.disabled = state.drafts.length === 0;
    }

    // ------------------------------------------------------------------
    // Loading overlay
    // ------------------------------------------------------------------
    function setLoading(loading, text = 'Working...') {
        state.loading = loading;
        document.querySelectorAll('.loading-overlay').forEach((o) => o.remove());
        if (loading) {
            const overlay = el('div', 'loading-overlay',
                `${icon('circle-notch', 'spin')}<span>${escapeHtml(text)}</span>`);
            const panel = $('results-panel').classList.contains('hidden') ? $('config-panel') : $('results-panel');
            panel.style.position = 'relative';
            panel.appendChild(overlay);
        }
    }

    // ------------------------------------------------------------------
    // Analyze CV match
    // ------------------------------------------------------------------
    async function analyzeMatch() {
        if (!state.cv) { toast('Upload your CV first', 'error'); return; }
        const selected = getSelectedJobs();
        if (!selected.length) { toast('Select at least one job', 'error'); return; }

        setLoading(true, 'Analyzing CV match... This may take a minute.');
        try {
            const result = await api('/api/jobs/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ jobs: selected }),
            });
            const matched = result.jobs || [];
            matched.forEach((mj) => {
                const idx = state.jobs.findIndex((j) => j.url === mj.url);
                if (idx >= 0) {
                    state.jobs[idx].match_score = mj.match_score;
                    state.jobs[idx].match_reason = mj.match_reason;
                }
            });
            state.jobs.sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
            state.selectedJobs.clear();
            matched.forEach((mj) => {
                const idx = state.jobs.findIndex((j) => j.url === mj.url);
                if (idx >= 0) state.selectedJobs.add(idx);
            });
            renderJobs();
            toast('CV analysis complete!', 'ok');
        } catch (err) {
            toast(`Analysis failed: ${err.message}`, 'error', 6000);
        } finally {
            setLoading(false);
        }
    }
    $('analyze-btn').addEventListener('click', analyzeMatch);

    // ------------------------------------------------------------------
    // Draft emails
    // ------------------------------------------------------------------
    async function draftEmails() {
        if (!state.cv) { toast('Upload your CV first', 'error'); return; }
        const selected = getSelectedJobs();
        if (!selected.length) { toast('Select at least one job', 'error'); return; }

        setLoading(true, 'Drafting cover emails... This may take a minute.');
        try {
            const result = await api('/api/jobs/draft', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ jobs: selected }),
            });
            state.drafts = result.drafts || [];
            showEmailModal();
            toast(`${state.drafts.length} email draft(s) ready`, 'ok');
            $('send-btn').disabled = false;
        } catch (err) {
            toast(`Drafting failed: ${err.message}`, 'error', 6000);
        } finally {
            setLoading(false);
        }
    }
    $('draft-btn').addEventListener('click', draftEmails);

    // ------------------------------------------------------------------
    // Email preview modal
    // ------------------------------------------------------------------
    function showEmailModal() {
        const container = $('email-drafts-container');
        container.innerHTML = '';

        state.drafts.forEach((item, i) => {
            const draft = item.draft || {};
            const card = el('div', 'email-draft-card');
            card.innerHTML = `
                <h3>${icon('briefcase')} ${escapeHtml(item.title || 'Job')} — ${escapeHtml(item.company || '')}</h3>
                <div class="draft-field">
                    <label>To</label>
                    <input type="email" id="draft-to-${i}" value="${escapeHtml(draft.to || item.contact_email || '')}">
                </div>
                <div class="draft-field">
                    <label>Subject</label>
                    <input type="text" id="draft-subject-${i}" value="${escapeHtml(draft.subject || '')}">
                </div>
                <textarea class="draft-body" id="draft-body-${i}" rows="8">${escapeHtml(draft.body || '')}</textarea>
            `;
            container.appendChild(card);
        });

        $('email-modal').classList.remove('hidden');
    }

    function closeEmailModal() {
        $('email-modal').classList.add('hidden');
    }
    $('close-email-modal').addEventListener('click', closeEmailModal);
    $('cancel-send-btn').addEventListener('click', closeEmailModal);
    $('email-modal').addEventListener('click', (e) => { if (e.target === $('email-modal')) closeEmailModal(); });

    // ------------------------------------------------------------------
    // Send applications
    // ------------------------------------------------------------------
    async function sendApplications() {
        const finalDrafts = state.drafts.map((item, i) => ({
            ...item,
            draft: {
                to: (document.getElementById(`draft-to-${i}`) || {}).value || item.draft?.to || '',
                subject: (document.getElementById(`draft-subject-${i}`) || {}).value || item.draft?.subject || '',
                body: (document.getElementById(`draft-body-${i}`) || {}).value || item.draft?.body || '',
            },
        }));

        const withRecipient = finalDrafts.filter((d) => d.draft.to);
        if (!withRecipient.length) {
            toast('No drafts have a recipient email address', 'error');
            return;
        }

        if (!confirm(`Send ${withRecipient.length} application email(s) with your CV attached?`)) return;

        closeEmailModal();
        setLoading(true, `Sending ${withRecipient.length} application(s)...`);
        try {
            const result = await api('/api/jobs/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ drafts: withRecipient }),
            });
            const results = result.results || [];
            const sent = results.filter((r) => r.send_status === 'sent').length;
            const failed = results.filter((r) => r.send_status === 'failed').length;
            const skipped = results.filter((r) => r.send_status === 'skipped').length;
            let msg = `${sent} sent`;
            if (failed) msg += `, ${failed} failed`;
            if (skipped) msg += `, ${skipped} skipped`;
            toast(msg, failed ? 'error' : 'ok', 6000);
            state.drafts = [];
            $('send-btn').disabled = true;
        } catch (err) {
            toast(`Send failed: ${err.message}`, 'error', 6000);
        } finally {
            setLoading(false);
        }
    }

    $('send-btn').addEventListener('click', () => {
        if (state.drafts.length) showEmailModal();
        else toast('Draft emails first', 'info');
    });
    $('confirm-send-btn').addEventListener('click', sendApplications);

    // ------------------------------------------------------------------
    // Analytics
    // ------------------------------------------------------------------
    async function loadAnalytics() {
        try {
            const { analytics } = await api('/api/jobs/analytics');
            if (!analytics) return;

            $('stat-total').textContent = analytics.total;
            $('stat-sent').textContent = analytics.sent;
            $('stat-failed').textContent = analytics.failed;
            $('stat-week').textContent = analytics.this_week;
            $('stat-month').textContent = analytics.this_month;
            $('stat-rate').textContent = analytics.total > 0
                ? `${Math.round((analytics.sent / analytics.total) * 100)}%`
                : '—';

            const companyEl = $('breakdown-company');
            if (analytics.by_company && analytics.by_company.length) {
                const maxCount = analytics.by_company[0].count;
                companyEl.innerHTML = analytics.by_company.map((c) =>
                    `<div class="breakdown-row">
                        <span class="breakdown-name">${escapeHtml(c.name)}</span>
                        <div class="breakdown-bar-wrap">
                            <div class="breakdown-bar" style="width: ${Math.max(8, (c.count / maxCount) * 100)}%"></div>
                        </div>
                        <span class="breakdown-count">${c.count}</span>
                    </div>`
                ).join('');
            } else {
                companyEl.innerHTML = '<span class="empty-state-sm">No data yet</span>';
            }

            const roleEl = $('breakdown-role');
            if (analytics.by_role && analytics.by_role.length) {
                const maxCount = analytics.by_role[0].count;
                roleEl.innerHTML = analytics.by_role.map((r) =>
                    `<div class="breakdown-row">
                        <span class="breakdown-name">${escapeHtml(r.name)}</span>
                        <div class="breakdown-bar-wrap">
                            <div class="breakdown-bar role" style="width: ${Math.max(8, (r.count / maxCount) * 100)}%"></div>
                        </div>
                        <span class="breakdown-count">${r.count}</span>
                    </div>`
                ).join('');
            } else {
                roleEl.innerHTML = '<span class="empty-state-sm">No data yet</span>';
            }

            const chartEl = $('chart-bars');
            if (analytics.timeline && analytics.timeline.length) {
                const maxDay = Math.max(...analytics.timeline.map((t) => t.count));
                chartEl.innerHTML = analytics.timeline.map((t) => {
                    const height = Math.max(4, (t.count / maxDay) * 100);
                    const label = t.date.slice(5);
                    return `<div class="chart-col" title="${t.date}: ${t.count} application(s)">
                        <div class="chart-bar-fill" style="height: ${height}%"></div>
                        <span class="chart-label">${label}</span>
                    </div>`;
                }).join('');
            } else {
                chartEl.innerHTML = '<span class="empty-state-sm">No activity yet</span>';
            }
        } catch { /* analytics not critical */ }
    }

    // ------------------------------------------------------------------
    // History
    // ------------------------------------------------------------------
    async function loadHistory() {
        loadAnalytics();
        try {
            const { applications } = await api('/api/jobs/history');
            const list = $('history-list');
            if (!applications || !applications.length) {
                list.innerHTML = '<p class="empty-state">No applications sent yet.</p>';
                return;
            }
            list.innerHTML = '';
            applications.reverse().forEach((app) => {
                const item = el('div', `history-item${app.send_status === 'failed' ? ' failed' : ''}`);
                const statusIcon = app.send_status === 'sent' ? 'check' : 'warning-circle';
                item.innerHTML = `
                    <div class="history-icon">${icon(statusIcon)}</div>
                    <div class="history-info">
                        <div class="hist-title">${escapeHtml(app.title || app.job?.title || 'Application')}</div>
                        <div class="hist-to">${icon('envelope')} ${escapeHtml(app.to || app.draft?.to || '-')}</div>
                        <div class="hist-meta">${escapeHtml(app.company || app.job?.company || '')} · ${escapeHtml(app.sent_at || '')}</div>
                    </div>
                `;
                list.appendChild(item);
            });
        } catch (err) {
            $('history-list').innerHTML = `<p class="empty-state">${escapeHtml(err.message)}</p>`;
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function getSelectedJobs() {
        return Array.from(state.selectedJobs).map((i) => state.jobs[i]).filter(Boolean);
    }

    // ------------------------------------------------------------------
    // Keyboard shortcuts
    // ------------------------------------------------------------------
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!$('email-modal').classList.contains('hidden')) closeEmailModal();
            if (!$('preset-name-modal').classList.contains('hidden')) $('preset-name-modal').classList.add('hidden');
        }
    });

    // ------------------------------------------------------------------
    // Start
    // ------------------------------------------------------------------
    loadConfig();
    loadCV();
    loadGeminiStatus();
})();
