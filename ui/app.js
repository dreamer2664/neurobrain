(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const captureMode = new URLSearchParams(location.search).has('capture');
  if (captureMode) document.documentElement.dataset.capture = 'true';

  const icon = (name, className = '') => `<svg class="icon ${className}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  const escapeHTML = (value) => String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const safeStorage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* private browsing */ } }
  };

  const themeContent = {
    pulse: {
      title: 'Neurobrain — Neural Command Center', context: 'COMMAND CENTER',
      eyebrow: 'NEURAL COMMAND CENTER', hero: 'Give your ideas a pair of hands.',
      copy: 'A private, local brain that sees your screen, takes action, and checks its work.', color: '#080a0f'
    },
    atelier: {
      title: 'Neurobrain — A calmer way to work', context: 'YOUR WORKSPACE',
      eyebrow: 'A QUIETER WAY TO WORK', hero: 'Less busywork. More headspace.',
      copy: 'Hand off the small things to a thoughtful local companion. You stay in control.', color: '#f2f0ea'
    },
    terminal: {
      title: 'Neurobrain — Operator Console', context: 'BRAINBOX // OPERATOR',
      eyebrow: 'BRAINBOX // OPERATOR CONSOLE', hero: 'Objective in. Outcome out.',
      copy: 'LOCAL INFERENCE · VISUAL GROUNDING · VERIFIED ACTIONS · UNDO TRAIL', color: '#070907'
    }
  };

  const state = {
    theme: captureMode ? 'pulse' : (safeStorage.get('neurobrain-theme') || 'pulse'),
    view: 'overview', filter: 'all', runtime: 'demo',
    apiBase: safeStorage.get('neurobrain-api') || '/api',
    created: 0, activeDialogTask: null
  };
  if (!themeContent[state.theme]) state.theme = 'pulse';

  const tasks = [
    {
      id: 't-0248', title: 'Research & draft: boiling point',
      objective: 'Research the boiling point of water and save a short note in Docs',
      status: 'working', progress: 68, icon: 'search', tone: '', step: 'Reading source 2 of 3',
      description: 'Chrome · Source comparison · Docs', elapsed: '02:14', stepsDone: 8, stepsTotal: 12,
      plan: [
        ['Open Chrome and read the objective', 'Observation verified · window focused', 'done'],
        ['Search for a reliable reference', '3 sources found · no blind action', 'done'],
        ['Compare sources and extract the answer', 'Reading source 2 of 3', 'current'],
        ['Open Docs and write the note', 'Waiting for verified research', ''],
        ['Read back and save', 'Runtime will verify the saved state', '']
      ]
    },
    {
      id: 't-0247', title: 'Organise Downloads',
      objective: 'Organise my Downloads folder and move old installers to trash',
      status: 'queued', progress: 0, icon: 'folder', tone: 'violet', step: 'Waiting behind active objective',
      description: 'Files · Recoverable cleanup · 8 steps', elapsed: 'QUEUED', stepsDone: 0, stepsTotal: 8,
      plan: [
        ['Inspect the Downloads folder', 'Starts when the active objective finishes', ''],
        ['Group files by type and age', 'Nothing is deleted at this step', ''],
        ['Move old installers to brainbox trash', 'Recoverable with undo', ''],
        ['Verify every file and write a summary', 'Byte-level undo trail retained', '']
      ]
    },
    {
      id: 't-0246', title: 'Summarise meeting notes',
      objective: 'Summarise my meeting notes into clear action items',
      status: 'done', progress: 100, icon: 'doc', tone: 'lime', step: 'Saved and verified · 5 action items',
      description: 'Notes · Summary · Verified save', elapsed: '00:48', stepsDone: 6, stepsTotal: 6,
      plan: [
        ['Read the selected notes', '1,284 words observed', 'done'],
        ['Extract decisions and owners', '5 action items identified', 'done'],
        ['Draft a concise summary', 'Kept on this machine', 'done'],
        ['Save and read back the document', 'Saved state verified', 'done']
      ]
    }
  ];

  const activities = [
    {icon:'eye', title:'Observation verified', detail:'Chrome changed to the selected reference article.', time:'11:42:18', code:'t-0248 · step 08 · progress', tone:''},
    {icon:'search', title:'Source comparison started', detail:'The objective now has three candidate references.', time:'11:41:52', code:'action → click(812, 405)', tone:''},
    {icon:'check', title:'Summary saved and read back', detail:'Five action items were verified in the saved document.', time:'11:36:09', code:'t-0246 · done · success', tone:'success'},
    {icon:'undo', title:'Recovery recorded', detail:'A wrong result was detected; the runtime returned instead of repeating a dead click.', time:'11:31:44', code:'verdict → no-op · retry 1/2', tone:'warning'},
    {icon:'shield', title:'Local runtime check passed', detail:'Both checkpoint families loaded inside the memory budget.', time:'11:28:03', code:'external_calls → 0', tone:'success'}
  ];

  const taskStatus = {
    working: ['WORKING', ''], queued: ['QUEUED', 'queued'], done: ['COMPLETE', 'done'], paused: ['PAUSED', 'paused']
  };

  function taskCard(task) {
    const [statusLabel, statusClass] = taskStatus[task.status];
    return `<article class="task-card ${task.status}" data-task-id="${task.id}">
      <button class="task-main" data-task-open="${task.id}" aria-label="Open ${escapeHTML(task.title)} details">
        <span class="task-symbol ${task.tone}">${icon(task.icon)}</span>
        <span class="task-info">
          <span class="task-title-row"><h3>${escapeHTML(task.title)}</h3><span class="task-status ${statusClass}"><i></i>${statusLabel}</span></span>
          <p>${escapeHTML(task.step)}</p>
          <span class="task-progress"><span class="track"><i style="width:${task.progress}%"></i></span><span>${String(task.progress).padStart(2,'0')}%</span></span>
        </span>
        <span class="task-meta"><span>${task.elapsed}</span><small>${task.stepsDone} / ${task.stepsTotal} STEPS</small></span>
        ${icon('chevron')}
      </button>
    </article>`;
  }

  function renderTasks() {
    const overview = $('#overview-task-list');
    const full = $('#full-task-list');
    overview.innerHTML = tasks.slice(0, 3).map(taskCard).join('');
    const filtered = state.filter === 'all' ? tasks : tasks.filter(task => task.status === state.filter);
    full.innerHTML = filtered.length ? filtered.map(taskCard).join('') : `<div class="empty-state">${icon('tasks')}<strong>No objectives in this view.</strong><p>Try another filter or create a new objective.</p></div>`;
    const active = tasks.filter(task => task.status === 'working' || task.status === 'paused').length;
    $('#active-count').textContent = active;
    $('#nav-task-count').textContent = tasks.length;
    $('#mobile-task-count').textContent = active;
    $('#filter-all-count').textContent = tasks.length;
  }

  function renderActivity() {
    $('#activity-feed').innerHTML = activities.map(event => `<article class="activity-event">
      <span class="event-icon ${event.tone}">${icon(event.icon)}</span>
      <div class="event-body"><div><strong>${escapeHTML(event.title)}</strong><time>${event.time}</time></div>
      <p>${escapeHTML(event.detail)}</p><code>${escapeHTML(event.code)}</code></div>
    </article>`).join('');
  }

  function setTheme(theme, announce = false) {
    if (!themeContent[theme]) return;
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    safeStorage.set('neurobrain-theme', theme);
    $$('[data-theme-choice]').forEach(button => {
      const selected = button.dataset.themeChoice === theme;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    const content = themeContent[theme];
    document.title = content.title;
    $('meta[name="theme-color"]').content = content.color;
    $('#context-label').textContent = content.context;
    $('#hero-eyebrow').innerHTML = `<span></span> ${content.eyebrow}`;
    $('#hero-title').textContent = content.hero;
    $('#hero-copy').textContent = content.copy;
    if (window.neuralField) window.neuralField.refresh();
    if (announce) toast(`${themeContent[theme].context.replace('BRAINBOX // ','')} concept applied`, 'check');
  }

  const viewLabels = {overview:'COMMAND CENTER', tasks:'OBJECTIVE QUEUE', activity:'VERIFIED EVENT LOG', brain:'MODEL & EVALUATIONS'};
  function setView(view, focus = false) {
    if (!$(`[data-panel="${view}"]`)) return;
    state.view = view;
    $$('[data-panel]').forEach(panel => {
      const selected = panel.dataset.panel === view;
      panel.hidden = !selected;
      panel.classList.toggle('active', selected);
    });
    $$('[data-nav]').forEach(button => button.classList.toggle('active', button.dataset.nav === view));
    $('#context-label').textContent = view === 'overview' ? themeContent[state.theme].context : viewLabels[view];
    window.scrollTo({top:0, behavior:captureMode ? 'auto' : 'smooth'});
    if (focus) $('#main-content').focus({preventScroll:true});
  }

  function toast(message, iconName = 'check', duration = 3200) {
    const element = document.createElement('div');
    element.className = 'toast';
    element.innerHTML = `${icon(iconName)}<span>${escapeHTML(message)}</span>`;
    $('#toast-region').append(element);
    setTimeout(() => {
      element.classList.add('leaving');
      setTimeout(() => element.remove(), 260);
    }, duration);
  }

  function openTask(id) {
    const task = tasks.find(item => item.id === id);
    if (!task) return;
    state.activeDialogTask = id;
    $('#task-dialog-id').textContent = `${task.id.toUpperCase()} · ${taskStatus[task.status][0]}`;
    const actionLabel = task.status === 'paused' ? 'Resume objective' : 'Pause objective';
    const actionIcon = task.status === 'paused' ? 'play' : 'pause';
    $('#task-dialog-content').innerHTML = `
      <h2 id="task-dialog-title">${escapeHTML(task.title)}</h2>
      <div class="dialog-task-meta"><span class="task-status ${taskStatus[task.status][1]}"><i></i>${taskStatus[task.status][0]}</span><span>${task.stepsDone} / ${task.stepsTotal} STEPS · ${task.elapsed}</span></div>
      <div class="dialog-description">${escapeHTML(task.objective)}</div>
      <div class="dialog-progress"><div><span>Verified progress</span><strong>${task.progress}%</strong></div><div class="track"><i style="width:${task.progress}%"></i></div></div>
      <h3 class="timeline-heading">PLAN & OBSERVATIONS</h3>
      <ol class="task-timeline">${task.plan.map(([title, detail, status]) => `<li class="${status}"><span class="timeline-dot">${status === 'done' ? icon('check') : status === 'current' ? icon('activity') : ''}</span><div><strong>${escapeHTML(title)}</strong><small>${escapeHTML(detail)}</small></div></li>`).join('')}</ol>
      <div class="dialog-actions">
        ${task.status !== 'done' ? `<button class="button button-secondary" data-task-action="toggle">${icon(actionIcon)}${actionLabel}</button>` : ''}
        <button class="button button-secondary danger-button" data-task-action="undo">${icon('undo')}Undo trail</button>
        <button class="button" data-close-dialog="task-dialog">Done</button>
      </div>`;
    $('#task-dialog').showModal();
  }

  function toggleActiveTask() {
    const task = tasks.find(item => item.id === state.activeDialogTask);
    if (!task || task.status === 'done') return;
    if (task.status === 'paused') {
      task.status = 'working'; task.step = 'Resuming from last verified observation';
      toast(`${task.id} resumed from its last checkpoint`, 'play');
    } else {
      task.status = 'paused'; task.step = 'Paused by operator · state preserved';
      toast(`${task.id} paused safely`, 'pause');
    }
    renderTasks();
    const id = task.id;
    $('#task-dialog').close();
    openTask(id);
  }

  function runObjective() {
    const input = $('#objective-input');
    const objective = input.value.trim();
    if (!objective) {
      toast('Write an objective first.', 'command');
      input.focus();
      return;
    }
    state.created += 1;
    const id = `t-${String(248 + state.created).padStart(4,'0')}`;
    const title = objective.length > 44 ? `${objective.slice(0, 44).trim()}…` : objective;
    const task = {
      id, title, objective, status:'working', progress:4, icon:'command', tone:'orange',
      step:'Planning from the first observation', description:'New objective · Demo simulation', elapsed:'00:00', stepsDone:0, stepsTotal:10,
      plan:[
        ['Read the objective and set a budget', '40 steps · 5 minutes · local only', 'done'],
        ['Observe the current screen', 'Planning the first safe action', 'current'],
        ['Act and verify each step', 'No blind actions', ''],
        ['Write a terminal record', 'Success is verified by the runtime', '']
      ]
    };
    tasks.unshift(task);
    activities.unshift({icon:'command', title:'New objective accepted', detail:objective, time:new Date().toLocaleTimeString('en-GB'), code:`${id} · planning · demo`, tone:''});
    input.value = '';
    renderTasks(); renderActivity();
    setView('overview');
    toast(`${id} added to the local demo queue`, 'check');
    simulateNewTask(task);
  }

  function simulateNewTask(task) {
    if (captureMode) return;
    const timer = setInterval(() => {
      if (!tasks.includes(task) || task.status === 'done') return clearInterval(timer);
      if (task.status !== 'working') return;
      task.progress = Math.min(100, task.progress + 7);
      task.stepsDone = Math.min(10, Math.floor(task.progress / 10));
      task.elapsed = `00:${String(Math.min(59, task.stepsDone * 4 + 2)).padStart(2,'0')}`;
      if (task.progress >= 100) {
        task.status = 'done'; task.step = 'Demo complete · terminal record written'; task.elapsed = '00:42';
        task.plan = task.plan.map(([title, detail]) => [title, detail, 'done']);
        activities.unshift({icon:'check', title:'Demo objective completed', detail:task.objective, time:new Date().toLocaleTimeString('en-GB'), code:`${task.id} · done · simulated`, tone:'success'});
        clearInterval(timer); renderActivity(); toast(`${task.id} demo run completed`, 'check');
      } else if (task.progress > 65) task.step = 'Verifying the final observation';
      else if (task.progress > 30) task.step = 'Acting, observing, and checking';
      renderTasks();
    }, 1800);
  }

  function updateConnectionUI(result) {
    const online = state.runtime === 'online';
    $('#runtime-label').textContent = online ? 'Runtime online' : 'Demo workspace';
    $$('.runtime-pill .status-dot').forEach(dot => dot.classList.toggle('online', online));
    const box = $('#connection-result');
    box.innerHTML = `<span class="status-dot ${online ? 'online' : ''}"></span><span>${escapeHTML(result || 'No runtime connected. Interactive demo data is shown.')}</span>`;
  }

  async function apiRequest(path, options = {}) {
    const base = state.apiBase.replace(/\/$/, '');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 8000);
    try {
      const response = await fetch(`${base}${path}`, {
        method: options.method || 'GET',
        headers: options.body ? {'Content-Type':'application/json'} : {},
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.reason || data.error || `HTTP ${response.status}`);
      return data;
    } finally { clearTimeout(timeout); }
  }

  async function testConnection() {
    state.apiBase = $('#api-base').value.trim() || '/api';
    safeStorage.set('neurobrain-api', state.apiBase);
    const button = $('#test-connection');
    button.disabled = true; button.textContent = 'Testing…';
    $('#connection-result').innerHTML = '<span class="status-dot"></span><span>Looking for brainbox on the local proxy…</span>';
    try {
      const health = await apiRequest('/health', {timeout:4500});
      if (!health.ok) throw new Error('Runtime responded but is not ready');
      state.runtime = 'online';
      updateConnectionUI(`Connected · checkpoint step ${health.step} · ${health.rss_mb} MB RSS. Tasks remain in demo mode.`);
      toast('Brainbox runtime is online', 'link');
    } catch (error) {
      state.runtime = 'demo';
      const message = error.name === 'AbortError' ? 'Connection timed out.' : error.message;
      updateConnectionUI(`Not connected: ${message}. Demo mode is still available.`);
    } finally {
      button.disabled = false; button.textContent = 'Test';
    }
  }

  document.addEventListener('click', event => {
    const themeButton = event.target.closest('[data-theme-choice]');
    if (themeButton) return setTheme(themeButton.dataset.themeChoice, !captureMode);
    const nav = event.target.closest('[data-nav]');
    if (nav) return setView(nav.dataset.nav, true);
    const suggestion = event.target.closest('[data-suggestion]');
    if (suggestion) {
      $('#objective-input').value = suggestion.dataset.suggestion;
      $('#objective-input').focus();
      return;
    }
    const taskButton = event.target.closest('[data-task-open]');
    if (taskButton) return openTask(taskButton.dataset.taskOpen);
    const closeButton = event.target.closest('[data-close-dialog]');
    if (closeButton) return $(`#${closeButton.dataset.closeDialog}`).close();
    const filter = event.target.closest('[data-filter]');
    if (filter) {
      state.filter = filter.dataset.filter;
      $$('[data-filter]').forEach(button => button.classList.toggle('active', button === filter));
      renderTasks(); return;
    }
    const taskAction = event.target.closest('[data-task-action]');
    if (taskAction) {
      if (taskAction.dataset.taskAction === 'toggle') toggleActiveTask();
      else toast('This demo task has no recoverable file changes. Undo history would appear here.', 'undo', 4300);
      return;
    }
    const action = event.target.closest('[data-action]');
    if (!action) return;
    switch (action.dataset.action) {
      case 'new-task':
        setView('overview');
        setTimeout(() => { $('#objective-input').focus(); $('#objective-input').scrollIntoView({behavior:captureMode ? 'auto' : 'smooth', block:'center'}); }, 60);
        break;
      case 'open-settings': $('#settings-dialog').showModal(); break;
      case 'notifications': toast('You’re all caught up. Runtime events appear in Activity.', 'bell'); break;
      case 'attach': toast('Context attachments are a design preview in this release.', 'paperclip'); break;
      case 'voice': toast('Voice input is a planned platform integration; no audio was recorded.', 'mic'); break;
      case 'workspace': toast('Single-machine local workspace · no remote account required', 'shield'); break;
    }
  });

  $('#run-objective').addEventListener('click', runObjective);
  $('#test-connection').addEventListener('click', testConnection);
  $('#use-demo').addEventListener('click', () => {
    state.runtime = 'demo'; updateConnectionUI('Demo mode selected. No runtime requests will be made.'); toast('Demo workspace active', 'check');
  });
  $('#api-base').value = state.apiBase;
  $('#objective-input').addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); runObjective(); }
  });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); setView('overview'); $('#objective-input').focus();
    }
    if (!event.ctrlKey && !event.metaKey && !event.altKey && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName) && !$('dialog[open]')) {
      const views = {'1':'overview','2':'tasks','3':'activity','4':'brain'};
      if (views[event.key]) setView(views[event.key], true);
    }
  });
  $$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
    if (event.target === dialog) {
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    }
  }));

  // Deterministic, lightweight neural topology. No images or third-party scripts.
  class NeuralField {
    constructor(canvas) {
      this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.dpr = 1; this.frame = null;
      let seed = 42;
      const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
      this.nodes = Array.from({length:38}, (_, index) => ({
        x:.05 + random() * .9, y:.08 + random() * .82, r:1 + random() * 1.6,
        phase:random() * Math.PI * 2, major:index % 7 === 0
      }));
      this.edges = [];
      this.nodes.forEach((node, index) => {
        const nearest = this.nodes.map((other, otherIndex) => ({otherIndex, distance:Math.hypot(node.x-other.x,node.y-other.y)}))
          .filter(item => item.otherIndex !== index).sort((a,b) => a.distance-b.distance).slice(0,2);
        nearest.forEach(({otherIndex,distance}) => {
          if (distance < .32 && !this.edges.some(edge => (edge[0] === otherIndex && edge[1] === index))) this.edges.push([index,otherIndex]);
        });
      });
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas.parentElement);
      this.resize(); this.refresh(); this.animate(1600);
    }
    resize() {
      const bounds = this.canvas.parentElement.getBoundingClientRect();
      this.width = bounds.width; this.height = bounds.height;
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(this.width * this.dpr));
      this.canvas.height = Math.max(1, Math.round(this.height * this.dpr));
      this.draw(captureMode ? 1600 : performance.now());
    }
    refresh() {
      const styles = getComputedStyle(document.documentElement);
      this.accent = styles.getPropertyValue('--accent').trim();
      this.secondary = styles.getPropertyValue('--secondary').trim();
      this.draw(captureMode ? 1600 : performance.now());
    }
    color(hex, alpha) {
      const value = hex.replace('#','');
      if (value.length !== 6) return hex;
      const number = parseInt(value,16);
      return `rgba(${number>>16},${(number>>8)&255},${number&255},${alpha})`;
    }
    draw(time) {
      if (!this.width || !this.height || !this.accent) return;
      const ctx = this.ctx, w = this.width, h = this.height;
      ctx.setTransform(this.dpr,0,0,this.dpr,0,0); ctx.clearRect(0,0,w,h);
      const moving = !captureMode && !matchMedia('(prefers-reduced-motion: reduce)').matches;
      const points = this.nodes.map(node => ({
        x:(node.x + (moving ? Math.sin(time/4200+node.phase)*.005 : 0))*w,
        y:(node.y + (moving ? Math.cos(time/4800+node.phase)*.006 : 0))*h,
        r:node.r, major:node.major
      }));
      this.edges.forEach(([a,b], index) => {
        const p1=points[a],p2=points[b];
        ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);
        ctx.strokeStyle=this.color(this.secondary,index%4===0?.14:.075);ctx.lineWidth=.65;ctx.stroke();
        if (index % 6 === 0) {
          const progress = captureMode ? .64 : ((time/2400 + index*.13)%1);
          const x=p1.x+(p2.x-p1.x)*progress,y=p1.y+(p2.y-p1.y)*progress;
          ctx.beginPath();ctx.arc(x,y,1.35,0,Math.PI*2);ctx.fillStyle=this.color(this.accent,.75);ctx.fill();
        }
      });
      points.forEach((point,index) => {
        if (point.major) {
          ctx.beginPath();ctx.arc(point.x,point.y,6,0,Math.PI*2);ctx.fillStyle=this.color(this.secondary,.035);ctx.fill();
          ctx.beginPath();ctx.arc(point.x,point.y,3.2,0,Math.PI*2);ctx.strokeStyle=this.color(this.secondary,.32);ctx.lineWidth=.7;ctx.stroke();
        }
        ctx.beginPath();ctx.arc(point.x,point.y,point.major?1.6:point.r,0,Math.PI*2);
        ctx.fillStyle=this.color(index%5===0?this.accent:this.secondary,point.major?.9:.55);ctx.fill();
      });
    }
    animate(time) {
      this.draw(time);
      if (!captureMode && !matchMedia('(prefers-reduced-motion: reduce)').matches) this.frame = requestAnimationFrame(next => this.animate(next));
    }
  }

  renderTasks(); renderActivity(); setTheme(state.theme); setView('overview');
  const requestedView = new URLSearchParams(location.search).get('view');
  if (['tasks','activity','brain'].includes(requestedView)) setView(requestedView);
  if (new URLSearchParams(location.search).get('action') === 'new') {
    setTimeout(() => $('#objective-input').focus(), 80);
  }
  window.neuralField = new NeuralField($('#neural-canvas'));
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.neuralField && !captureMode && !window.neuralField.frame) window.neuralField.animate(performance.now());
  });

  // Small, documented adapter for integration work. Task execution above is explicitly simulated.
  window.Neurobrain = {
    version:'1.0.0', setTheme, navigate:setView,
    api:{
      health:() => apiRequest('/health'),
      generate:(prompt, options={}) => apiRequest('/generate',{method:'POST',body:{prompt,max_tokens:options.maxTokens||64,temperature:options.temperature??.8,top_k:options.topK||40},timeout:options.timeout||120000}),
      score:(prompt, topK=5) => apiRequest('/score',{method:'POST',body:{prompt,top_k:topK}}),
      ground:(pngBase64,instruction) => apiRequest('/ground',{method:'POST',body:{png_b64:pngBase64,instruction},timeout:120000})
    }
  };

  if ('serviceWorker' in navigator && location.protocol.startsWith('http') && !captureMode) {
    window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
  }
})();
