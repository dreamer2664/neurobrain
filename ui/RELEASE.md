# Neurobrain UI Concepts v1.0.0

**Design release · 23 October 2026**

Three responsive UI directions for the local-first Neurobrain / brainbox product:

- **Pulse** — cinematic neural command center (recommended flagship)
- **Atelier** — warm, calm productivity workspace
- **Terminal** — precise technical operator console

## Included

- Eight high-resolution PNG screenshots covering desktop, mobile, task detail, and model metrics
- One JPG design-overview contact sheet
- Complete installable PWA source with no app build step
- Desktop and mobile layouts, offline static cache, and bundled fonts/icons
- Interactive demo tasks, observation timelines, activity, filters, pause/resume, and concept switching
- Dependency-free Python static server and fixed-route loopback brainbox proxy
- Real adapter methods for health, generate, score, and ground
- Design tokens, rationale, font licenses, and reproducible Playwright scripts
- 26 passing browser checks

## Quick start

```bash
cd source/ui
python3 serve.py
# http://127.0.0.1:4173
```

See `source/ui/README.md` for mobile, brainbox, testing, and security instructions.

**Prototype boundary:** task execution is deliberately labeled as simulated because the current brainbox HTTP API has no task-lifecycle routes. Runtime health and the exposed model API methods are real when connected. This release does not include native APK/IPA/desktop binaries.

The recommended product direction is **Pulse**, with Atelier and Terminal retained as alternate modes.
