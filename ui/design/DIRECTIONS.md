# Neurobrain UI directions

## Product idea

Neurobrain should not look like another cloud chat client. Its differentiator is a small, local, inspectable loop: **observe → act → verify → recover**. The interface makes that loop visible and treats restraint as a feature.

Five principles guided all three directions:

1. **Local is visible.** Runtime state, memory envelope, checkpoint, and zero-external-call language are part of the main composition rather than buried in settings.
2. **Work, not chat.** The primary input is an objective. Tasks, observations, and outcomes dominate the information architecture.
3. **Honesty is a visual material.** Demo state and evaluation labels are explicit. The 30% verified research result is shown beside the 98% trained-curriculum result.
4. **Control stays close.** Pause, step budgets, history, and undo are accessible at the point of action.
5. **One brain, multiple expressions.** The information architecture remains consistent while tone, density, typography, geometry, and motion change.

## 01 — Pulse / recommended flagship

**Mood:** alive, precise, private, quietly powerful.

Pulse uses near-black blue surfaces, soft radial light, thin borders, acidic lime for action, and glacial cyan for ongoing work. Space Grotesk gives the product a distinctive voice without sacrificing readability. The neural field is the emotional center: a living topology rather than a stock robot or brain photograph.

Best for the main brand because it makes the from-scratch networks tangible while still presenting a friendly objective composer. Its motion language should remain subtle: traveling pulses, small state changes, and no perpetual large movement.

## 02 — Atelier / calm companion

**Mood:** warm, editorial, humane, focused.

Atelier shifts the product from mission control to a calm desk. Parchment and warm white replace black; iris marks intent and coral marks active work. Larger radii, softer shadows, and plain-language hero copy reduce the perceived technical burden. The dark neural card is deliberately retained as a window into the machine beneath the calm surface.

Best for a consumer-facing mode or a future “Focus” experience. It demonstrates that local AI does not need a neon/science-fiction aesthetic.

## 03 — Terminal / operator console

**Mood:** explicit, technical, deterministic.

Terminal uses IBM Plex Mono throughout, square geometry, dashed boundaries, scanlines, phosphor green, and amber. The neural core rotates into a technical diamond and the language becomes command-oriented. It is not a retro terminal imitation; it retains modern touch targets, layout, and accessible controls.

Best for developers, evaluations, debugging, and an advanced-operator mode.

## Responsive composition

### Desktop

A persistent sidebar contains workspace, primary navigation, privacy posture, runtime settings, and operator identity. The overview uses a two-column composition: objective + current work on the left; neural state + machine budget on the right. This gives the composer priority without hiding the brain.

### Compact desktop / tablet

The sidebar collapses to icons. The neural and system panels move below the main workflow; architecture becomes vertical. No horizontal scrolling is introduced.

### Phone

The sidebar becomes a five-item bottom bar with a central create action. The top bar retains brand, concept switching, and runtime status. Content becomes a single flow with the objective first, neural state second, and current tasks after it. Dialogs become bottom sheets and fixed navigation respects safe-area insets.

## Accessibility and trust

- Semantic buttons, navigation, sections, labels, dialogs, and live toast region
- Visible focus outlines using the secondary color
- Reduced-motion media query plus static deterministic capture mode
- Primary mobile navigation targets are at least 44 CSS px
- Status is never communicated by color alone; text labels accompany dots and bars
- The connection dialog states the exact prototype boundary
- No remote font, tracking, or image dependency

Before production, run a formal WCAG contrast audit and assistive-technology pass on every final component state. Micro-labels are intentionally secondary; core task titles and actions should not be reduced further.

## Recommended next increments

1. Choose **Pulse** as the flagship and retain Atelier/Terminal as alternate appearance modes.
2. Add a real task-lifecycle API contract; bind the existing task store and append-only event stream rather than synthesizing events in the browser.
3. Add a screenshot/observation viewer with a privacy redaction step and explicit retention controls.
4. Design approval sheets for destructive or externally visible actions; pair every recoverable action with its actual undo record.
5. Add empty, offline, model-missing, RAM-governor-refused, retry-exhausted, and partial-success states.
6. Package the same responsive front end in native shells only if platform distribution requires it; the current release is a PWA and does not claim APK/IPA/desktop-native binaries.
