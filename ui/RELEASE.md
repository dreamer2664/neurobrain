# Neurobrain Atelier v2.0.0

**Design release · 23 October 2026**

Atelier is now the single Neurobrain UI direction: a quiet, responsive local workspace with four colour palettes switchable in **Settings → Colour theme**.

## Included

- **Linen, Mist, Rose, and Ink** palettes with locally persisted selection
- Twelve high-resolution PNG screenshots covering every palette on desktop and mobile
- Settings, objective-detail, mobile-queue, and mobile-model-metric captures
- One JPG four-palette contact sheet
- Complete installable PWA source with no app build step
- Responsive layouts verified at desktop, tablet, 390 px, and 320 px
- Offline static cache, bundled Geist fonts, icons, and license files
- Objective composer, filters, observation timelines, activity, pause/resume, and demo progress
- Persistent step/time budgets, reduced-motion preference, and API endpoint
- Dependency-free Python server and fixed-route loopback Brainbox proxy
- Real adapter methods for health, generate, score, and ground
- Design tokens, rationale, and reproducible Playwright scripts
- **99 passing browser checks**
- Taste-skill lint and real-browser inspection: **0 failures / 0 warnings** in each

## Quick start

```bash
cd source/ui
python3 serve.py
# http://127.0.0.1:4173
```

See `source/ui/README.md` for mobile, Brainbox, testing, accessibility, and security instructions.

**Prototype boundary:** objective execution is deliberately labelled as simulated because the current Brainbox HTTP API has no task-lifecycle routes. Runtime health and the exposed model API methods are real when connected. This release does not include native APK, IPA, or desktop binaries.

The earlier multi-concept v1.0.0 release remains available separately. Atelier v2 replaces its concept switcher and decorative neural canvas with one consistent workspace and a concrete observation fragment.
