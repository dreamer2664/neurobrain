import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'screenshots');
const baseURL = process.env.BASE_URL || 'http://127.0.0.1:4173';
await mkdir(output, {recursive:true});

const shots = [
  {file:'01-linen-desktop.png',name:'Linen — Desktop overview',theme:'linen',width:1440,height:900},
  {file:'02-linen-mobile.png',name:'Linen — Mobile overview',theme:'linen',width:390,height:844,mobile:true},
  {file:'03-mist-desktop.png',name:'Mist — Desktop overview',theme:'mist',width:1440,height:900},
  {file:'04-mist-mobile.png',name:'Mist — Mobile overview',theme:'mist',width:390,height:844,mobile:true},
  {file:'05-rose-desktop.png',name:'Rose — Desktop overview',theme:'rose',width:1440,height:900},
  {file:'06-rose-mobile.png',name:'Rose — Mobile overview',theme:'rose',width:390,height:844,mobile:true},
  {file:'07-ink-desktop.png',name:'Ink — Desktop overview',theme:'ink',width:1440,height:900},
  {file:'08-ink-mobile.png',name:'Ink — Mobile overview',theme:'ink',width:390,height:844,mobile:true},
  {file:'09-settings-desktop.png',name:'Atelier — Settings and palette choices',theme:'linen',width:1440,height:900,view:'settings'},
  {file:'10-task-detail-desktop.png',name:'Atelier — Objective plan and observations',theme:'linen',width:1440,height:900,detail:true},
  {file:'11-objectives-mobile.png',name:'Atelier — Mobile objective queue',theme:'linen',width:390,height:844,mobile:true,view:'tasks'},
  {file:'12-brain-mobile.png',name:'Ink — Mobile model metrics',theme:'ink',width:390,height:844,mobile:true,view:'brain'}
];

const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
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
      colorScheme:shot.theme === 'ink' ? 'dark' : 'light'
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${baseURL}/?capture=1`,{waitUntil:'networkidle'});
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(theme => Neurobrain.setTheme(theme),shot.theme);
    if (shot.view) {
      const scope = shot.mobile ? '.mobile-nav' : (shot.view === 'settings' ? '.rail' : '.rail-nav');
      await page.locator(`${scope} [data-nav="${shot.view}"]`).click();
    }
    if (shot.detail) await page.locator('#overview-task-list [data-task-open="t-0248"]').click();
    await page.mouse.move(0,0);
    await page.waitForTimeout(150);
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error(`Horizontal overflow in ${shot.file}`);
    if (errors.length) throw new Error(`${shot.file}: ${errors.join('; ')}`);
    await page.screenshot({path:path.join(output,shot.file),fullPage:false,animations:'disabled'});
    manifest.push({
      file:shot.file,
      name:shot.name,
      palette:shot.theme,
      viewport:`${shot.width}x${shot.height}`,
      device:shot.mobile?'mobile':'desktop',
      view:shot.view || (shot.detail?'objective detail':'overview'),
      scale:shot.mobile?2:1.5
    });
    console.log(`captured ${shot.file}`);
    await context.close();
  }
  await writeFile(path.join(output,'manifest.json'),JSON.stringify({
    product:'Neurobrain Atelier',version:'2.0.0',captureMode:'static demo data; reduced motion',capturedAt:new Date().toISOString(),screenshots:manifest
  },null,2)+'\n');
} finally {
  await browser.close();
}
