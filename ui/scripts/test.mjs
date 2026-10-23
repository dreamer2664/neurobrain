import { chromium } from 'playwright';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:4173';
const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
const checks = [];
const check = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
const noOverflow = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
const visiblePanel = (page, view) => page.locator(`[data-panel="${view}"]`).isVisible();

try {
  const context = await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${baseURL}/?capture=1`,{waitUntil:'networkidle'});
  await page.evaluate(() => document.fonts.ready);

  check(await page.title() === 'Neurobrain Atelier — Overview','Atelier loads with the correct document title');
  check(await page.evaluate(() => window.Neurobrain?.version === '2.0.0'),'Public integration adapter exposes version 2.0.0');
  check(await page.evaluate(() => window.Neurobrain?.themes.join(',') === 'linen,mist,rose,ink'),'Public adapter exposes all four palettes');
  check(await page.locator('#overview-task-list .task-card').count() === 3,'Overview renders three objective cards');
  check(await noOverflow(page),'Desktop overview has no horizontal overflow');
  check(await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
    return new Set(ids).size === ids.length;
  }),'Document contains no duplicate element IDs');

  await page.locator('.rail [data-nav="settings"]').click();
  check(await visiblePanel(page,'settings'),'Desktop Settings navigation works');
  check(await page.locator('[data-theme-choice]').count() === 4,'Settings contains four palette choices');
  const themeColors = {linen:'#f4f3f1',mist:'#f2f4f7',rose:'#f6f3f4',ink:'#101111'};
  for (const theme of ['mist','rose','ink','linen']) {
    await page.locator(`[data-theme-choice="${theme}"]`).click();
    check(await page.locator('html').getAttribute('data-theme') === theme,`${theme} palette applies from Settings`);
    check(await page.locator(`[data-theme-choice="${theme}"]`).getAttribute('aria-pressed') === 'true',`${theme} palette exposes selected state`);
    check(await page.evaluate(() => localStorage.getItem('neurobrain-palette')) === theme,`${theme} palette is stored locally`);
    check(await page.locator('meta[name="theme-color"]').getAttribute('content') === themeColors[theme],`${theme} updates the browser theme colour`);
    check(await noOverflow(page),`${theme} Settings has no horizontal overflow`);
  }

  await page.locator('#default-steps').selectOption('60');
  await page.locator('#default-minutes').selectOption('10');
  await page.locator('label[for="reduce-motion"]').click();
  check(await page.locator('#reduce-motion').isChecked(),'Reduced-motion switch can be toggled');
  check(await page.locator('html').getAttribute('data-reduce-motion') === 'true','Reduced-motion preference is applied to the document');
  await page.locator('.rail-nav [data-nav="overview"]').click();
  check(await page.locator('#budget-label').textContent() === '60 steps · 10 minutes','Configured objective budget reaches the composer');

  await page.locator('#overview-task-list [data-task-open="t-0248"]').click();
  check(await page.locator('#task-dialog').isVisible(),'Objective detail dialog opens');
  check(await page.locator('#task-dialog .task-timeline li').count() === 5,'Objective detail includes its observation plan');
  await page.locator('[data-task-action="toggle"]').click();
  check((await page.locator('#task-dialog-id').textContent()).includes('Paused'),'Pause control preserves and displays objective state');
  await page.locator('[data-task-action="toggle"]').click();
  check((await page.locator('#task-dialog-id').textContent()).includes('Working'),'Resume control restores the working state');
  await page.locator('[data-task-action="undo"]').click();
  check((await page.locator('.toast').last().textContent()).includes('no recoverable file changes'),'Undo trail explains the current demo boundary');
  await page.locator('#task-dialog .dialog-actions [data-close-dialog]').click();
  check(!(await page.locator('#task-dialog').isVisible()),'Objective dialog closes cleanly');

  const objective = 'Verify the Atelier release workflow';
  await page.locator('#objective-input').fill(objective);
  await page.locator('#run-objective').click();
  check(await page.locator('#overview-task-list .task-card').first().locator('h3').textContent() === objective,'New objective enters the local demo queue');
  check(await page.locator('#overview-task-list .task-card').count() === 3,'Overview remains capped at three recent objectives');
  check(await page.locator('#objective-input').inputValue() === '','Composer clears after accepting an objective');
  await page.locator('#run-objective').click();
  check((await page.locator('.toast').last().textContent()).includes('Write an objective first'),'Empty objectives are rejected with useful feedback');
  await page.locator('[data-suggestion]').first().click();
  check((await page.locator('#objective-input').inputValue()).includes('Downloads folder'),'Composer suggestions populate the objective field');
  await page.locator('#objective-input').fill('');

  await page.locator('.rail [data-nav="tasks"]').click();
  check(await page.locator('#full-task-list .task-card').count() === 4,'Full objective view includes the newly created item');
  await page.locator('[data-filter="done"]').click();
  check(await page.locator('#full-task-list .task-card').count() === 1,'Completed objective filter works');
  await page.locator('[data-filter="queued"]').click();
  check(await page.locator('#full-task-list .task-card').count() === 1,'Queued objective filter works');
  await page.locator('[data-filter="working"]').click();
  check(await page.locator('#full-task-list .task-card').count() === 2,'Active filter includes both working objectives');
  await page.locator('[data-filter="all"]').click();
  await page.locator('#full-task-list [data-task-open="t-0249"]').click();
  check((await page.locator('#task-dialog .task-timeline li').first().textContent()).includes('60 steps · 10 minutes'),'New objective retains the configured budget in its plan');
  await page.locator('#task-dialog .dialog-actions [data-close-dialog]').click();

  await page.locator('.rail [data-nav="activity"]').click();
  check(await page.locator('#activity-feed .activity-event').count() === 6,'New objective writes a visible activity event');
  check((await page.locator('#activity-feed').textContent()).includes(objective),'Activity record preserves the objective text');
  await page.locator('.rail [data-nav="brain"]').click();
  check(await page.locator('.metric-strip article').count() === 4,'Brain view exposes four specifically labelled metrics');
  check((await page.locator('.metrics-note').textContent()).includes('not a general capability claim'),'Evaluation caveat remains visible');
  check((await page.locator('.component-table').textContent()).includes('0.44M'),'Brain view identifies the vision grounder');

  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press('5');
  check(await visiblePanel(page,'settings'),'Number shortcut opens Settings');
  await page.keyboard.press('Control+k');
  check(await page.locator('#objective-input').evaluate(element => element === document.activeElement),'Composer keyboard shortcut focuses the objective field');
  await page.locator('#objective-input').fill('Run by keyboard');
  await page.keyboard.press('Control+Enter');
  check(await page.locator('#overview-task-list .task-card').first().locator('h3').textContent() === 'Run by keyboard','Ctrl/Cmd+Enter starts the composed objective');

  await page.route('**/api/health', route => route.fulfill({contentType:'application/json',body:JSON.stringify({ok:true,step:18600,rss_mb:312})}));
  await page.route('**/api/generate', route => route.fulfill({contentType:'application/json',body:JSON.stringify({ok:true,text:'controlled generation'})}));
  await page.route('**/api/score', route => route.fulfill({contentType:'application/json',body:JSON.stringify({ok:true,scores:[1,2]})}));
  await page.route('**/api/ground', route => route.fulfill({contentType:'application/json',body:JSON.stringify({ok:true,x:12,y:24})}));
  await page.locator('.rail [data-nav="settings"]').click();
  await page.locator('#test-connection').click();
  await page.waitForFunction(() => document.querySelector('#runtime-label').textContent === 'Runtime online');
  check(await page.locator('#runtime-label').textContent() === 'Runtime online','Runtime health check updates the global status');
  check((await page.locator('#connection-result').textContent()).includes('checkpoint 18600'),'Runtime health check reports the actual checkpoint');
  check((await page.locator('#connection-result').textContent()).includes('Objectives remain in demo mode'),'Connected state keeps the objective simulation boundary explicit');
  check(await page.evaluate(async () => (await Neurobrain.api.generate('hello',{maxTokens:12})).text) === 'controlled generation','Generate adapter returns a controlled real HTTP response');
  check(await page.evaluate(async () => (await Neurobrain.api.score('hello',2)).scores.length) === 2,'Score adapter returns controlled scores');
  check(await page.evaluate(async () => (await Neurobrain.api.ground('cG5n','click Save')).x) === 12,'Ground adapter returns controlled coordinates');
  await page.locator('#use-demo').click();
  check(await page.locator('#runtime-label').textContent() === 'Demo runtime','Use demo mode disconnects the displayed runtime state');
  const unknownRoute = await context.request.get(`${baseURL}/api/not-a-route`);
  check(unknownRoute.status() === 404,'Proxy refuses unknown API routes');

  await page.setViewportSize({width:768,height:1024});
  for (const view of ['overview','tasks','activity','brain','settings']) {
    await page.evaluate(view => Neurobrain.navigate(view),view);
    check(await noOverflow(page),`Tablet ${view} has no horizontal overflow`);
  }
  await page.setViewportSize({width:1440,height:1000});
  check(errors.length === 0,`Desktop and tablet have no browser exceptions${errors.length ? ': '+errors.join('; ') : ''}`);
  await context.close();

  const mobile = await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,reducedMotion:'reduce'});
  const mobilePage = await mobile.newPage();
  const mobileErrors = [];
  mobilePage.on('pageerror', error => mobileErrors.push(error.message));
  await mobilePage.goto(`${baseURL}/?capture=1`,{waitUntil:'networkidle'});
  check(await mobilePage.locator('.mobile-nav').isVisible(),'Mobile bottom navigation is visible');
  check(!(await mobilePage.locator('.rail').isVisible()),'Desktop icon rail collapses on mobile');
  for (const view of ['settings','tasks','activity','brain','overview']) {
    await mobilePage.locator(`.mobile-nav [data-nav="${view}"]`).click();
    check(await visiblePanel(mobilePage,view),`Mobile ${view} navigation works`);
    check(await noOverflow(mobilePage),`Mobile ${view} has no horizontal overflow`);
  }
  await mobilePage.locator('.mobile-nav [data-nav="settings"]').click();
  await mobilePage.locator('[data-theme-choice="ink"]').click();
  check(await mobilePage.locator('html').getAttribute('data-theme') === 'ink','Ink palette works with touch navigation');
  await mobilePage.locator('.mobile-nav [data-nav="overview"]').click();
  await mobilePage.locator('#overview-task-list [data-task-open="t-0248"]').click();
  check(await mobilePage.locator('#task-dialog').isVisible(),'Objective detail opens on mobile');
  check(await mobilePage.locator('#task-dialog').evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight;
  }),'Mobile objective dialog stays inside the viewport');
  await mobilePage.locator('#task-dialog .dialog-top [data-close-dialog]').click();
  await mobilePage.locator('#objective-input').fill('Mobile objective');
  await mobilePage.locator('#run-objective').click();
  check(await mobilePage.locator('#overview-task-list .task-card').first().locator('h3').textContent() === 'Mobile objective','Mobile composer creates an objective');
  await mobilePage.setViewportSize({width:320,height:700});
  for (const view of ['overview','tasks','activity','brain','settings']) {
    await mobilePage.evaluate(view => Neurobrain.navigate(view),view);
    check(await noOverflow(mobilePage),`320px ${view} has no horizontal overflow`);
  }
  check(mobileErrors.length === 0,`Mobile has no browser exceptions${mobileErrors.length ? ': '+mobileErrors.join('; ') : ''}`);
  await mobile.close();

  const persisted = await browser.newContext({viewport:{width:1280,height:900},reducedMotion:'reduce'});
  const persistedPage = await persisted.newPage();
  await persistedPage.goto(`${baseURL}/?view=settings`,{waitUntil:'networkidle'});
  await persistedPage.locator('[data-theme-choice="rose"]').click();
  await persistedPage.locator('#default-steps').selectOption('20');
  await persistedPage.locator('#default-minutes').selectOption('2');
  await persistedPage.locator('label[for="reduce-motion"]').click();
  await persistedPage.locator('#api-base').fill('/api');
  await persistedPage.reload({waitUntil:'networkidle'});
  check(await persistedPage.locator('html').getAttribute('data-theme') === 'rose','Palette preference survives a full reload');
  check(await persistedPage.locator('#default-steps').inputValue() === '20','Step budget survives a full reload');
  check(await persistedPage.locator('#default-minutes').inputValue() === '2','Time limit survives a full reload');
  check(await persistedPage.locator('#reduce-motion').isChecked(),'Motion preference survives a full reload');
  check(await persistedPage.locator('#api-base').inputValue() === '/api','API endpoint survives a full reload');
  await persisted.close();

  const offline = await browser.newContext({viewport:{width:1280,height:900},reducedMotion:'reduce'});
  const offlinePage = await offline.newPage();
  await offlinePage.goto(baseURL,{waitUntil:'networkidle'});
  await offlinePage.evaluate(() => navigator.serviceWorker.ready.then(() => true));
  await offlinePage.reload({waitUntil:'networkidle'});
  check(await offlinePage.evaluate(() => !!navigator.serviceWorker.controller),'Service worker controls the application shell');
  await offline.setOffline(true);
  await offlinePage.reload({waitUntil:'domcontentloaded'});
  await offlinePage.locator('#overview-task-list .task-card').first().waitFor();
  check(await offlinePage.locator('#overview-task-list .task-card').count() === 3,'Application shell and objectives reopen offline');
  check(await offlinePage.evaluate(() => getComputedStyle(document.body).fontFamily.includes('Geist')),'Bundled font and stylesheet remain available offline');
  await offline.setOffline(false);
  await offline.close();

  console.log(`\n${checks.length} browser checks passed:`);
  checks.forEach((message,index) => console.log(`${index+1}. ${message}`));
} finally {
  await browser.close();
}
