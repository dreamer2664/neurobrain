import { chromium } from 'playwright';

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:4173';
const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
const checks = [];
const check = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };

try {
  const context = await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${baseURL}/?capture=1`,{waitUntil:'networkidle'});
  await page.evaluate(() => document.fonts.ready);
  check(await page.title() === 'Neurobrain — Neural Command Center','Pulse loads with the correct document title');
  check(await page.evaluate(() => window.Neurobrain?.version === '1.0.0'),'Public integration adapter is exposed');
  check(await page.locator('#overview-task-list .task-card').count() === 3,'Overview renders three task cards');

  for (const theme of ['atelier','terminal','pulse']) {
    await page.locator(`[data-theme-choice="${theme}"]`).click();
    check(await page.locator('html').getAttribute('data-theme') === theme,`${theme} concept switch works`);
  }

  await page.locator('.side-nav [data-nav="tasks"]').click();
  check(await page.locator('[data-panel="tasks"]').isVisible(),'Desktop task navigation works');
  await page.locator('[data-filter="done"]').click();
  check(await page.locator('#full-task-list .task-card').count() === 1,'Task status filter works');
  await page.locator('[data-filter="all"]').click();

  await page.locator('.side-nav [data-nav="overview"]').click();
  await page.locator('#overview-task-list [data-task-open="t-0248"]').click();
  check(await page.locator('#task-dialog').isVisible(),'Task detail dialog opens');
  check(await page.locator('#task-dialog .task-timeline li').count() === 5,'Task detail includes its observation plan');
  await page.locator('[data-task-action="toggle"]').click();
  check((await page.locator('#task-dialog-id').textContent()).includes('PAUSED'),'Pause control preserves and displays task state');
  await page.locator('#task-dialog [data-close-dialog="task-dialog"]').first().click();

  const objective = 'Verify the interactive UI test objective';
  await page.locator('#objective-input').fill(objective);
  await page.locator('#run-objective').click();
  check(await page.locator('#overview-task-list .task-card').first().locator('h3').textContent() === objective,'New objective enters the demo queue');
  check(await page.locator('#overview-task-list .task-card').count() === 3,'Overview remains capped at three recent objectives');

  await page.locator('.side-nav [data-nav="activity"]').click();
  check(await page.locator('#activity-feed .activity-event').count() >= 6,'New objective writes a visible activity event');
  await page.locator('.side-nav [data-nav="brain"]').click();
  check(await page.locator('.metric-card').count() === 4,'Brain view exposes four specifically labelled metrics');
  check((await page.locator('.metrics-note').textContent()).includes('no general-intelligence claim'),'Evaluation caveat is visible');

  await page.locator('.settings-link').click();
  check(await page.locator('#settings-dialog').isVisible(),'Runtime settings dialog opens');
  check((await page.locator('.settings-info').textContent()).includes('simulated'),'Runtime boundary honestly labels simulated execution');
  await page.locator('#settings-dialog [data-close-dialog="settings-dialog"]').first().click();
  check(errors.length === 0,`Desktop has no browser exceptions${errors.length ? ': '+errors.join('; ') : ''}`);
  await context.close();

  const mobile = await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,reducedMotion:'reduce'});
  const mobilePage = await mobile.newPage();
  mobilePage.on('pageerror', error => errors.push(error.message));
  await mobilePage.goto(`${baseURL}/?capture=1`,{waitUntil:'networkidle'});
  check(await mobilePage.locator('.mobile-nav').isVisible(),'Mobile bottom navigation is visible');
  check(!(await mobilePage.locator('.sidebar').isVisible()),'Desktop sidebar collapses on mobile');
  check(await mobilePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth),'Mobile overview has no horizontal overflow');
  await mobilePage.locator('.mobile-nav [data-nav="brain"]').click();
  check(await mobilePage.locator('[data-panel="brain"]').isVisible(),'Mobile brain navigation works');
  check(await mobilePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth),'Mobile brain view has no horizontal overflow');
  await mobilePage.locator('.mobile-nav [data-action="new-task"]').click();
  check(await mobilePage.locator('[data-panel="overview"]').isVisible(),'Mobile create button returns to the composer');
  check(errors.length === 0,`Mobile has no browser exceptions${errors.length ? ': '+errors.join('; ') : ''}`);
  await mobile.close();

  console.log(`\n${checks.length} browser checks passed:`);
  checks.forEach((message,index) => console.log(`${String(index+1).padStart(2,'0')}. ${message}`));
} finally {
  await browser.close();
}
