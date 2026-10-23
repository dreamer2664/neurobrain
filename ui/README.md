# Neurobrain Atelier — UI Design Release 2.0.0

A quieter, responsive workspace for **Neurobrain / Brainbox**, delivered as an installable Progressive Web App. Atelier is the single product direction in this release. Four colour palettes are available in **Settings → Colour theme** and persist on the current device.

![Atelier palette overview](screenshots/00-design-overview.jpg)

## Four palettes, one workspace

| Palette | Ground | Text | Data accent | Character |
|---|---|---|---|---|
| **Linen** | `#f4f3f1` | `#20211f` | `#2f6f59` | Stone and forest |
| **Mist** | `#f2f4f7` | `#182028` | `#1d5fbf` | Blue-grey and cobalt |
| **Rose** | `#f6f3f4` | `#282126` | `#9c3f62` | Neutral rose and raspberry |
| **Ink** | `#101111` | `#f5f5f2` | `#8fb8a2` | Neutral dark and sage |

The layout and typography stay constant across palettes. Colour is reserved primarily for progress, grounding, and state rather than decorative chrome. There are no gradients, glows, glass panels, neural animations, or external images.

## What works

- Desktop, tablet, and phone layouts down to 320 CSS px without horizontal overflow
- Settings tab with four persistent palettes, objective budgets, reduced motion, and runtime endpoint preferences
- Objective composer, suggestions, filters, recent-objective queue, and keyboard shortcuts
- Objective detail dialogs with plans, observations, pause/resume, and an explicit undo-trail boundary
- Latest-observation interface fragment showing the grounded target and verified action
- Append-only activity view and specifically labelled model/evaluation metrics
- Installable PWA with bundled icons and an offline application-shell cache
- Real same-origin adapter for Brainbox `health`, `generate`, `score`, and `ground`
- Bundled Geist and Geist Mono fonts; no CDN, analytics, or external runtime assets

### Keyboard shortcuts

- `Ctrl/Cmd + K`: focus the objective composer
- `Ctrl/Cmd + Enter`: start the composed demo objective
- `1` / `2` / `3` / `4` / `5`: Overview / Objectives / Activity / Brain / Settings, when not typing

## Run it

No application build step or runtime npm dependencies are required. Python 3.10+ is enough:

```bash
cd ui
python3 serve.py
# open http://127.0.0.1:4173
```

On Windows, use `py serve.py` from this directory.

To connect the included proxy to the existing runtime, start Brainbox first from the repository root:

```bash
brainbox serve
# in another terminal:
cd ui && python3 serve.py
```

Then open **Settings → Local runtime → Test connection**. The proxy forwards only the fixed routes `/health`, `/generate`, `/score`, and `/ground` to `http://127.0.0.1:8321`; Brainbox itself remains loopback-only.

### Phone on the same network

```bash
cd ui
python3 serve.py --host 0.0.0.0 --port 4173
# on the phone: http://YOUR-PC-LAN-IP:4173
```

Use `0.0.0.0` only on a trusted network: it makes the UI and fixed local API proxy reachable by other devices. Browser installation and offline PWA features require localhost or HTTPS; an HTTP LAN page still runs as a responsive web app.

## Honest integration boundary

The current Brainbox HTTP API exposes model operations, not a task-lifecycle endpoint. Therefore:

- **Objective creation, progress, pause/resume, and activity entries are explicitly labelled demo simulations.**
- The runtime health check is real when a local Brainbox is available.
- `Neurobrain.api` exposes real request methods for the four existing model routes.
- A connected runtime does not turn simulated objectives into real desktop execution.

```js
const health = await Neurobrain.api.health();
const result = await Neurobrain.api.generate('def hello():\n', {maxTokens: 32});
const scores = await Neurobrain.api.score('The capital of France is', 5);
// const point = await Neurobrain.api.ground(pngBase64, 'click Save');
```

The next backend integration should add local task endpoints for create, list, events, pause/resume, and undo rather than repurposing `/generate` as a task runner.

## Screenshots

Twelve high-resolution PNG captures are included in [`screenshots/`](screenshots/):

1. `01-linen-desktop.png` / `02-linen-mobile.png`
2. `03-mist-desktop.png` / `04-mist-mobile.png`
3. `05-rose-desktop.png` / `06-rose-mobile.png`
4. `07-ink-desktop.png` / `08-ink-mobile.png`
5. `09-settings-desktop.png` — palette choices and preferences
6. `10-task-detail-desktop.png` — plan and observation timeline
7. `11-objectives-mobile.png` — mobile objective queue
8. `12-brain-mobile.png` — Ink model metrics and architecture

`00-design-overview.jpg` is the four-palette contact sheet. PNGs are rendered at 1.5× on desktop and 2× on mobile. [`screenshots/manifest.json`](screenshots/manifest.json) records the exact capture matrix.

## Reproduce checks and captures

With the UI server running on port 4173:

```bash
cd ui
npm install
npx playwright install chromium   # first machine only
npm test                         # 99 browser checks
npm run capture                  # 12 PNGs + manifest
python3 scripts/make_contact_sheet.py   # requires Pillow
```

On minimal Linux, `npx playwright install-deps chromium` may be needed first. `BASE_URL` can point the test and capture scripts at another server.

The visual pass uses the requested [`Leonxlnx/taste-skill`](https://github.com/Leonxlnx/taste-skill). This release records **0 lint failures / 0 warnings** and **0 browser-inspection failures / 0 warnings**. The inspector was run at 1440, 768, and 390 px, and its screenshots were reviewed manually.

## Source map

```text
ui/
├── index.html                 # Semantic app shell and inline SVG icons
├── styles.css                 # Four palettes, responsive layouts, reduced motion
├── app.js                     # Objectives, Settings, persistence, Brainbox adapter
├── serve.py                   # Static server + fixed loopback API proxy
├── manifest.webmanifest       # Install metadata and shortcuts
├── sw.js                      # Offline static application-shell cache
├── design/tokens.json         # Portable palettes, type, radii, breakpoints
├── assets/                    # Bundled fonts, OFL licenses, and PWA icons
├── scripts/                   # Playwright checks, captures, contact sheet
└── screenshots/               # Release imagery and capture manifest
```

## Implementation notes

- The visual direction is `achromatic-ledger`: low-chroma grounds, flat opaque surfaces, and one hue reserved for data.
- Geist is used for the interface; Geist Mono is reserved for literal code.
- The Brain view uses repository metrics with specific evaluation labels and a visible caveat; it does not make a general capability claim.
- API requests are same-origin. The Python proxy refuses non-loopback Brainbox URLs and unknown paths.
- Service-worker caching excludes `/api/*` and all non-GET requests.
- This is a PWA design release, not a native APK, IPA, or desktop binary.

See [`design/DIRECTIONS.md`](design/DIRECTIONS.md) for the design rationale and [`RELEASE.md`](RELEASE.md) for the release summary.
