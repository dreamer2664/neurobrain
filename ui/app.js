(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const captureMode = new URLSearchParams(location.search).has('capture');
  if (captureMode) document.documentElement.dataset.capture = 'true';

  const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  const escapeHTML = value => String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const safeStorage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage can be unavailable */ } }
  };

  const themes = {
    linen: {name:'Linen', description:'stone and forest', color:'#f4f3f1'},
    mist: {name:'Mist', description:'blue-grey and cobalt', color:'#f2f4f7'},
    rose: {name:'Rose', description:'neutral rose and raspberry', color:'#f6f3f4'},
    ink: {name:'Ink', description:'neutral dark and sage', color:'#101111'}
  };
  const savedSteps = Number(safeStorage.get('neurobrain-steps'));
  const savedMinutes = Number(safeStorage.get('neurobrain-minutes'));
  const state = {
    theme: captureMode ? 'linen' : (safeStorage.get('neurobrain-palette') || 'linen'),
    view: 'overview', filter: 'all', runtime: 'demo',
    apiBase: safeStorage.get('neurobrain-api') || '/api',
    steps: [20,40,60].includes(savedSteps) ? savedSteps : 40,
    minutes: [2,5,10].includes(savedMinutes) ? savedMinutes : 5,
    reduceMotion: safeStorage.get('neurobrain-motion') === 'true',
    created: 0, activeDialogTask: null
  };
  if (!themes[state.theme]) state.theme = 'linen';

  const tasks = [
    {
      id:'t-0248', title:'Research & draft: boiling point', objective:'Research the boiling point of water and save a short note in Docs',
      status:'working', progress:68, step:'Reading source 2 of 3', elapsed:'02:14', stepsDone:8, stepsTotal:12,
      plan:[
        ['Open Chrome and read the objective','Observation verified · window focused','done'],
        ['Search for a reliable reference','3 sources found · no blind action','done'],
        ['Compare sources and extract the answer','Reading source 2 of 3','current'],
        ['Open Docs and write the note','Waiting for verified research',''],
        ['Read back and save','Runtime will verify the saved state','']
      ]
    },
    {
      id:'t-0247', title:'Organise Downloads', objective:'Organise my Downloads folder and move old installers to trash',
      status:'queued', progress:0, step:'Waiting behind active objective', elapsed:'Queued', stepsDone:0, stepsTotal:8,
      plan:[
        ['Inspect the Downloads folder','Starts when the active objective finishes',''],
        ['Group files by type and age','Nothing is deleted at this step',''],
        ['Move old installers to brainbox trash','Recoverable with undo',''],
        ['Verify every file and write a summary','Byte-level undo trail retained','']
      ]
    },
    {
      id:'t-0246', title:'Summarise meeting notes', objective:'Summarise my meeting notes into clear action items',
      status:'done', progress:100, step:'Saved and verified · 5 action items', elapsed:'00:48', stepsDone:6, stepsTotal:6,
      plan:[
        ['Read the selected notes','1,284 words observed','done'],
        ['Extract decisions and owners','5 action items identified','done'],
        ['Draft a concise summary','Kept on this machine','done'],
        ['Save and read back the document','Saved state verified','done']
      ]
    }
  ];

  const activities = [
    {title:'Observation verified', detail:'Chrome changed to the selected reference article.', time:'11:42:18', code:'t-0248 · step 8 · progress'},
    {title:'Source comparison started', detail:'The objective now has three candidate references.', time:'11:41:52', code:'action → click(812, 405)'},
    {title:'Summary saved and read back', detail:'Five action items were verified in the saved document.', time:'11:36:09', code:'t-0246 · done · success'},
    {title:'Recovery recorded', detail:'A wrong result was detected; the runtime returned instead of repeating a dead click.', time:'11:31:44', code:'verdict → no-op · retry 1/2'},
    {title:'Local runtime check passed', detail:'Both checkpoint families loaded inside the memory budget.', time:'11:28:03', code:'external_calls → 0'}
  ];
  const statusLabels = {working:'Working', queued:'Queued', done:'Complete', paused:'Paused'};
  const viewLabels = {overview:'Overview', tasks:'Objectives', activity:'Activity', brain:'The brain', settings:'Settings'};
  let savedTimer;

  function taskCard(task) {
    return `<article class="task-card ${task.status}" data-task-id="${task.id}">
      <button class="task-main" data-task-open="${task.id}" aria-label="Open ${escapeHTML(task.title)} details">
        <span class="task-info">
          <span class="task-title-row"><h3 class="task-title">${escapeHTML(task.title)}</h3><span class="task-status ${task.status}"><i></i>${statusLabels[task.status]}</span></span>
          <p>${escapeHTML(task.step)}</p>
          <span class="task-progress"><span class="track"><i style="width:${task.progress}%"></i></span><span>${task.progress}%</span></span>
        </span>
        <span class="task-meta"><span>${task.elapsed}</span><small>${task.stepsDone} / ${task.stepsTotal} steps</small></span>
        ${icon('chevron')}
      </button>
    </article>`;
  }

  function renderObservation() {
    const task = tasks.find(item => item.status === 'working' || item.status === 'paused') || tasks[0];
    if (!task) return;
    $('#observation-task-id').textContent = `${task.id} · step ${task.stepsDone}`;
    const shot = $('#grounding-shot');
    const details = $('#observation-details');
    shot.classList.remove('empty');
    if (task.id === 't-0248') {
      shot.innerHTML = `<div class="mock-browser-bar"><span></span><div>reference.local/water</div></div><div class="mock-page"><h3>Boiling point of water</h3><p>100 °C at standard atmospheric pressure.</p><div class="grounded-row"><span>Read the summary</span>${icon('chevron')}</div></div>`;
      details.innerHTML = '<div><span>Verified</span><strong>Source 2 opened · click(812, 405)</strong></div>';
      $('#overview-finding').textContent = task.status === 'paused' ? 'Research is paused at source 2 of 3. Its last verified state is preserved.' : 'Research is comparing source 2 of 3. Everything runs on this machine.';
    } else if (task.id === 't-0246') {
      shot.innerHTML = `<div class="mock-browser-bar"><span></span><div>docs.local/meeting-summary</div></div><div class="mock-page"><h3>Meeting summary</h3><p>Five action items saved and verified.</p><div class="grounded-row"><span>Saved state verified</span>${icon('check')}</div></div>`;
      details.innerHTML = '<div><span>Verified</span><strong>Document read back · five items</strong></div>';
      $('#overview-finding').textContent = 'No active objectives. The last summary was saved and verified on this machine.';
    } else if (task.status === 'queued' || (task.id !== 't-0246' && task.progress < 30)) {
      shot.classList.add('empty');
      shot.innerHTML = '<p>No screen observation yet. The first rung reads the current desktop before any action.</p>';
      details.innerHTML = `<div><span>State</span><strong>${task.status === 'queued' ? 'Waiting to start' : 'Planning first observation'}</strong></div><div><span>Mode</span><strong>Interactive demo</strong></div>`;
      $('#overview-finding').textContent = task.status === 'queued' ? 'No objective is running. The next run starts by reading the screen.' : `${task.title} is planning its first observation. Demo execution stays on this device.`;
    } else {
      shot.innerHTML = `<div class="mock-browser-bar"><span></span><div>desktop.local/observation</div></div><div class="mock-page"><span class="mock-source">Simulated desktop</span><h3>Observation ${task.stepsDone}</h3><p>${escapeHTML(task.step)}</p><div class="grounded-row"><span>Demo state checked</span>${icon('check')}</div></div>`;
      details.innerHTML = `<div><span>Action</span><strong>Simulated objective step</strong></div><div><span>Verified</span><strong>Demo state changed</strong></div><code>${task.id} · simulated</code>`;
      $('#overview-finding').textContent = `${task.title} is at ${task.progress}% in the interactive demo.`;
    }
  }

  function renderTasks() {
    $('#overview-task-list').innerHTML = tasks.slice(0,3).map(taskCard).join('');
    const filtered = state.filter === 'all' ? tasks : state.filter === 'working' ? tasks.filter(task => ['working','paused'].includes(task.status)) : tasks.filter(task => task.status === state.filter);
    $('#full-task-list').innerHTML = filtered.length ? filtered.map(taskCard).join('') : '<div class="empty-state">No objectives in this view. Choose another filter or start a new objective.</div>';
    const active = tasks.filter(task => ['working','paused'].includes(task.status)).length;
    const queued = tasks.filter(task => task.status === 'queued').length;
    const done = tasks.filter(task => task.status === 'done').length;
    $('#active-summary').textContent = `${active} active`;
    $('#rail-task-count').textContent = active;
    $('#objectives-summary').textContent = `${active} active · ${queued} queued · ${done} complete`;
    renderObservation();
  }

  function renderActivity() {
    $('#activity-feed').innerHTML = activities.map(event => `<article class="activity-event"><div class="event-top"><strong>${escapeHTML(event.title)}</strong><time>${event.time}</time></div><p>${escapeHTML(event.detail)}</p><code>${escapeHTML(event.code)}</code></article>`).join('');
    $('#activity-count').textContent = `${activities.length} events`;
  }

  function showSaved() {
    if (captureMode) return;
    clearTimeout(savedTimer);
    $('#settings-saved').textContent = 'Saved on this device';
    savedTimer = setTimeout(() => { $('#settings-saved').textContent = ''; }, 1800);
  }

  function setTheme(theme, userInitiated = false) {
    if (!themes[theme]) return;
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    safeStorage.set('neurobrain-palette', theme);
    $$('[data-theme-choice]').forEach(button => {
      const selected = button.dataset.themeChoice === theme;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    $('#theme-summary').textContent = `${themes[theme].name} · ${themes[theme].description}`;
    $('meta[name="theme-color"]').content = themes[theme].color;
    if (userInitiated) showSaved();
  }

  function setView(view, focus = false) {
    if (!viewLabels[view]) return;
    state.view = view;
    $$('[data-panel]').forEach(panel => {
      const selected = panel.dataset.panel === view;
      panel.hidden = !selected;
      panel.classList.toggle('active', selected);
    });
    $$('[data-nav]').forEach(button => button.classList.toggle('active', button.dataset.nav === view));
    document.title = `Neurobrain Atelier — ${viewLabels[view]}`;
    window.scrollTo({top:0, behavior:captureMode || state.reduceMotion ? 'auto' : 'smooth'});
    if (focus) $('#main-content').focus({preventScroll:true});
  }

  function updateBudget() {
    const label = `${state.steps} steps · ${state.minutes} minutes`;
    $('#budget-label').textContent = label;
    $('#policy-budget').textContent = label;
  }

  function applyMotionPreference() {
    document.documentElement.dataset.reduceMotion = String(state.reduceMotion);
    $('#reduce-motion').checked = state.reduceMotion;
  }

  function toast(message, duration = 3200) {
    const element = document.createElement('div');
    element.className = 'toast';
    element.textContent = message;
    $('#toast-region').append(element);
    setTimeout(() => { element.classList.add('leaving'); setTimeout(() => element.remove(), 220); }, duration);
  }

  function openTask(id) {
    const task = tasks.find(item => item.id === id);
    if (!task) return;
    state.activeDialogTask = id;
    $('#task-dialog-id').textContent = `${task.id} · ${statusLabels[task.status]}`;
    const actionLabel = task.status === 'paused' ? 'Resume objective' : 'Pause objective';
    const actionIcon = task.status === 'paused' ? 'play' : 'pause';
    $('#task-dialog-content').innerHTML = `
      <h2 id="task-dialog-title">${escapeHTML(task.title)}</h2>
      <div class="dialog-task-meta"><span>${statusLabels[task.status]}</span><span class="num">${task.stepsDone} / ${task.stepsTotal} steps · ${task.elapsed}</span></div>
      <div class="dialog-description">${escapeHTML(task.objective)}</div>
      <div class="dialog-progress"><div><span>Verified progress</span><strong>${task.progress}%</strong></div><div class="track"><i style="width:${task.progress}%"></i></div></div>
      <h3 class="timeline-heading">Plan and observations</h3>
      <ol class="task-timeline">${task.plan.map(([title, detail, status]) => `<li class="${status}"><span class="timeline-dot">${status === 'done' ? icon('check') : ''}</span><div><strong>${escapeHTML(title)}</strong><small>${escapeHTML(detail)}</small></div></li>`).join('')}</ol>
      <div class="dialog-actions">${task.status !== 'done' ? `<button class="button button-secondary" data-task-action="toggle">${icon(actionIcon)}${actionLabel}</button>` : ''}<button class="button button-secondary" data-task-action="undo">${icon('undo')}Undo trail</button><button class="button button-primary" data-close-dialog="task-dialog">Done</button></div>`;
    $('#task-dialog').showModal();
  }

  function toggleActiveTask() {
    const task = tasks.find(item => item.id === state.activeDialogTask);
    if (!task || task.status === 'done') return;
    if (task.status === 'paused') {
      task.status = 'working'; task.step = 'Resuming from last verified observation'; toast(`${task.id} resumed from its last checkpoint`);
    } else {
      task.status = 'paused'; task.step = 'Paused by operator · state preserved'; toast(`${task.id} paused safely`);
    }
    renderTasks();
    const id = task.id;
    $('#task-dialog').close();
    openTask(id);
  }

  function runObjective() {
    const input = $('#objective-input');
    const objective = input.value.trim();
    if (!objective) { toast('Write an objective first.'); input.focus(); return; }
    state.created += 1;
    const id = `t-${String(248 + state.created).padStart(4,'0')}`;
    const title = objective.length > 44 ? `${objective.slice(0,44).trim()}…` : objective;
    const task = {
      id, title, objective, status:'working', progress:4, step:'Planning from the first observation', elapsed:'00:00', stepsDone:0, stepsTotal:state.steps,
      plan:[
        ['Read the objective and set a budget',`${state.steps} steps · ${state.minutes} minutes · local only`,'done'],
        ['Observe the current screen','Planning the first safe action','current'],
        ['Act and verify each step','No blind actions',''],
        ['Write a terminal record','Success is verified by the runtime','']
      ]
    };
    tasks.unshift(task);
    activities.unshift({title:'New objective accepted', detail:objective, time:new Date().toLocaleTimeString('en-GB'), code:`${id} · planning · demo`});
    input.value = '';
    renderTasks(); renderActivity(); setView('overview');
    toast(`${id} added to the local demo queue`);
    simulateNewTask(task);
  }

  function simulateNewTask(task) {
    if (captureMode) return;
    const timer = setInterval(() => {
      if (!tasks.includes(task) || task.status === 'done') return clearInterval(timer);
      if (task.status !== 'working') return;
      task.progress = Math.min(100, task.progress + 7);
      task.stepsDone = Math.min(task.stepsTotal, Math.floor(task.progress / 100 * task.stepsTotal));
      task.elapsed = `00:${String(Math.min(59, task.stepsDone * 2 + 2)).padStart(2,'0')}`; // taste-skill-ignore: elapsed-time clock
      if (task.progress >= 100) {
        task.status = 'done'; task.step = 'Demo complete · terminal record written'; task.elapsed = '00:42'; task.stepsDone = task.stepsTotal;
        task.plan = task.plan.map(([title, detail]) => [title, detail, 'done']);
        activities.unshift({title:'Demo objective completed', detail:task.objective, time:new Date().toLocaleTimeString('en-GB'), code:`${task.id} · done · simulated`});
        clearInterval(timer); renderActivity(); toast(`${task.id} demo run completed`);
      } else if (task.progress > 65) task.step = 'Verifying the final observation';
      else if (task.progress > 30) task.step = 'Acting, observing, and checking';
      renderTasks();
    }, 1800);
  }

  function updateConnectionUI(result) {
    const online = state.runtime === 'online';
    $('#runtime-label').textContent = online ? 'Runtime online' : 'Demo runtime';
    $('#settings-runtime-summary').textContent = online ? 'Connected on this device' : 'No runtime connected';
    $$('.runtime-link .status-dot').forEach(dot => dot.classList.toggle('online', online));
    $('#connection-result').innerHTML = `<span class="status-dot ${online ? 'online' : ''}"></span><span>${escapeHTML(result || 'Demo data is shown. No runtime requests have been made.')}</span>`;
  }

  async function apiRequest(path, options = {}) {
    const base = state.apiBase.replace(/\/$/, '');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 8000);
    try {
      const response = await fetch(`${base}${path}`, {method:options.method || 'GET', headers:options.body ? {'Content-Type':'application/json'} : {}, body:options.body ? JSON.stringify(options.body) : undefined, signal:controller.signal});
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
      updateConnectionUI(`Connected at checkpoint ${health.step} with ${health.rss_mb} MB RSS. Objectives remain in demo mode.`);
      toast('Brainbox runtime is online');
    } catch (error) {
      state.runtime = 'demo';
      updateConnectionUI(`Not connected: ${error.name === 'AbortError' ? 'connection timed out' : error.message}. Demo mode is still available.`);
    } finally {
      button.disabled = false; button.textContent = 'Test connection';
    }
  }

  document.addEventListener('click', event => {
    const themeButton = event.target.closest('[data-theme-choice]');
    if (themeButton) return setTheme(themeButton.dataset.themeChoice, true);
    const nav = event.target.closest('[data-nav]');
    if (nav) return setView(nav.dataset.nav, true);
    const suggestion = event.target.closest('[data-suggestion]');
    if (suggestion) { $('#objective-input').value = suggestion.dataset.suggestion; $('#objective-input').focus(); return; }
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
      else toast('This demo objective has no recoverable file changes. Its undo history would appear here.', 4300);
      return;
    }
    const action = event.target.closest('[data-action]');
    if (!action) return;
    if (action.dataset.action === 'new-task') {
      setView('overview');
      setTimeout(() => { $('#objective-input').focus(); $('#objective-input').scrollIntoView({behavior:captureMode || state.reduceMotion ? 'auto' : 'smooth', block:'center'}); }, 60);
    } else if (action.dataset.action === 'workspace') toast('Single-machine workspace. No remote account is required.');
  });

  $('#run-objective').addEventListener('click', runObjective);
  $('#test-connection').addEventListener('click', testConnection);
  $('#use-demo').addEventListener('click', () => { state.runtime = 'demo'; updateConnectionUI('Demo mode selected. No runtime requests will be made.'); showSaved(); });
  $('#default-steps').addEventListener('change', event => { state.steps = Number(event.target.value); safeStorage.set('neurobrain-steps', state.steps); updateBudget(); showSaved(); });
  $('#default-minutes').addEventListener('change', event => { state.minutes = Number(event.target.value); safeStorage.set('neurobrain-minutes', state.minutes); updateBudget(); showSaved(); });
  $('#reduce-motion').addEventListener('change', event => { state.reduceMotion = event.target.checked; safeStorage.set('neurobrain-motion', state.reduceMotion); applyMotionPreference(); showSaved(); });
  $('#objective-input').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); runObjective(); } });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); setView('overview'); $('#objective-input').focus(); }
    if (!event.ctrlKey && !event.metaKey && !event.altKey && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) && !$('dialog[open]')) {
      const views = {'1':'overview','2':'tasks','3':'activity','4':'brain','5':'settings'};
      if (views[event.key]) setView(views[event.key], true);
    }
  });
  $$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
    if (event.target === dialog) {
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    }
  }));

  $('#default-steps').value = String(state.steps);
  $('#default-minutes').value = String(state.minutes);
  $('#api-base').value = state.apiBase;
  renderTasks(); renderActivity(); setTheme(state.theme); updateBudget(); applyMotionPreference(); updateConnectionUI(); setView('overview');
  const requestedView = new URLSearchParams(location.search).get('view');
  if (viewLabels[requestedView]) setView(requestedView);
  if (new URLSearchParams(location.search).get('action') === 'new') setTimeout(() => $('#objective-input').focus(), 80);

  window.Neurobrain = {
    version:'2.0.0', themes:Object.keys(themes), setTheme, navigate:setView,
    api:{
      health:() => apiRequest('/health'),
      generate:(prompt, options={}) => apiRequest('/generate',{method:'POST',body:{prompt,max_tokens:options.maxTokens||64,temperature:options.temperature??.8,top_k:options.topK||40},timeout:options.timeout||120000}),
      score:(prompt, topK=5) => apiRequest('/score',{method:'POST',body:{prompt,top_k:topK}}),
      ground:(pngBase64,instruction) => apiRequest('/ground',{method:'POST',body:{png_b64:pngBase64,instruction},timeout:120000})
    }
  };

  if ('serviceWorker' in navigator && location.protocol.startsWith('http') && !captureMode) window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
})();
