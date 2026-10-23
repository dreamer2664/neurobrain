import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'screenshots');
const baseURL = process.env.BASE_URL || 'http://127.0.0.1:4173';
await mkdir(output, { recursive: true });

const shots = [
  {file:'01-pulse-desktop.png', name:'Pulse — Desktop command center', theme:'pulse', width:1440, height:900},
  {file:'02-pulse-mobile.png', name:'Pulse — Mobile command center', theme:'pulse', width:390, height:844, mobile:true},
  {file:'03-atelier-desktop.png', name:'Atelier — Desktop calm workspace', theme:'atelier', width:1440, height:900},
  {file:'04-atelier-mobile.png', name:'Atelier — Mobile calm workspace', theme:'atelier', width:390, height:844, mobile:true},
  {file:'05-terminal-desktop.png', name:'Terminal — Desktop operator console', theme:'terminal', width:1440, height:900},
  {file:'06-terminal-mobile.png', name:'Terminal — Mobile operator console', theme:'terminal', width:390, height:844, mobile:true},
  {file:'07-pulse-task-detail.png', name:'Pulse — Task observation timeline', theme:'pulse', width:1440, height:900, detail:true},
  {file:'08-atelier-mobile-brain.png', name:'Atelier — Mobile model metrics', theme:'atelier', width:390, height:844, mobile:true, view:'brain'}
];

const browser = await chromium.launch({headless:true, args:['--no-sandbox']});
const manifest = [];
try {
  for (const shot of shots) {
    const context = await browser.newContext({
      viewport:{width:shot.width,height:shot.height},
      screen:{width:shot.width,height:shot.height},
      deviceScaleFactor:shot.mobile ? 2 : 1.5,
      isMobile:!!shot.mobile,
      hasTouch:!!shot.mobile,
      reducedMotion:'reduce',
      colorScheme:shot.theme === 'atelier' ? 'light' : 'dark'
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${baseURL}/?capture=1`, {waitUntil:'networkidle'});
    await page.evaluate(() => document.fonts.ready);
    if (shot.theme !== 'pulse') await page.locator(`[data-theme-choice="${shot.theme}"]`).click();
    if (shot.view) await page.locator(`.mobile-nav [data-nav="${shot.view}"]`).click();
    if (shot.detail) await page.locator('#overview-task-list [data-task-open="t-0248"]').click();
    await page.waitForTimeout(250);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    if (overflow) throw new Error(`Horizontal overflow in ${shot.file}`);
    if (errors.length) throw new Error(`${shot.file}: ${errors.join('; ')}`);
    await page.screenshot({path:path.join(output,shot.file), fullPage:false, animations:'disabled'});
    manifest.push({file:shot.file,name:shot.name,concept:shot.theme,viewport:`${shot.width}x${shot.height}`,device:shot.mobile?'mobile':'desktop',view:shot.view || (shot.detail?'task detail':'overview')});
    console.log(`captured ${shot.file}`);
    await context.close();
  }
  await writeFile(path.join(output,'manifest.json'),JSON.stringify({product:'Neurobrain',version:'1.0.0',capturedAt:new Date().toISOString(),screenshots:manifest},null,2)+'\n');
} finally {
  await browser.close();
}
