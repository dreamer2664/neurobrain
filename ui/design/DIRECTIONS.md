# Neurobrain Atelier — design direction

## Product idea

Neurobrain should not look like another cloud chat client. Its differentiator is a small, local, inspectable loop: **observe → act → verify → recover**. Atelier makes that loop visible through ordinary product UI rather than science-fiction decoration.

Five principles guide the release:

1. **Local is visible.** Checkpoint, component sizes, CPU threads, memory envelope, and external-call count belong in the main composition.
2. **Work, not chat.** The primary input is an objective. Current work, observations, and outcomes dominate the information architecture.
3. **Honesty is a visual material.** Demo execution is labelled, evaluation results retain their specific scopes, and the weakest measured result remains visible.
4. **Control stays close.** Budgets, pause/resume, history, and undo explanations sit at the point of action.
5. **One workspace, several palettes.** Appearance changes do not rearrange the product or alter the meaning of its controls.

## Selected direction

**Atelier / `achromatic-ledger`**

Taste-skill direction seed: `1806656935`

Atelier uses low-chroma, desk-inspired grounds; opaque white or near-black surfaces; thin structural borders; and a single restrained data hue. Geist provides a neutral, highly legible interface voice. Geist Mono appears only in literal code.

The composition takes cues from a well-kept workspace: a clear objective at the top, current work below it, and a narrow evidence column beside it. It does not imitate paper, leather, or a physical desk.

## What was deliberately removed

The earlier multi-concept release was useful for exploration, but this revision removes the elements that made it feel generated rather than designed:

- Pulse and Terminal concepts and the concept switcher
- Animated neural topology and ambient canvas rendering
- Gradients, radial glows, glass surfaces, decorative shadows, and scanlines
- Marketing-style hero copy and oversized promotional typography
- Repeated badges, decorative index numbers, and inflated capability language
- Monospace as a general interface aesthetic
- Redundant “interactive demo” chrome; the runtime state and Settings boundary carry that information instead

The result is intentionally quieter. The signature element is not an illustration of AI; it is a concrete observation from the current objective.

## The observation fragment

The latest-observation card is an interface fragment inside the product interface. It shows:

- the source or surface being observed;
- the relevant content, kept brief;
- the grounded target with a visible accent outline;
- the verified action and its coordinates.

This makes the observe/act/verify loop tangible without pretending that a simulated action really ran. Empty and completed states use the same component rather than substituting decorative artwork.

## Palette system

| Palette | Ground | Text | Data accent | Use |
|---|---|---|---|---|
| Linen | `#f4f3f1` | `#20211f` | `#2f6f59` | Default stone/forest workspace |
| Mist | `#f2f4f7` | `#182028` | `#1d5fbf` | Cooler blue-grey workspace |
| Rose | `#f6f3f4` | `#282126` | `#9c3f62` | Neutral rose with raspberry data |
| Ink | `#101111` | `#f5f5f2` | `#8fb8a2` | Neutral dark with sage data |

The palettes change surfaces, text, borders, controls, and data accents together. They do not introduce different layouts or theme-specific features. Primary buttons remain near-black in light palettes and near-white in Ink. Status is always accompanied by text, never colour alone.

Palette selection lives in **Settings → Colour theme**, not in the global header. The selection, motion preference, budgets, and API endpoint use guarded local storage so restricted storage does not prevent the app from loading.

## Responsive composition

### Desktop

A 56 px icon rail keeps navigation available without turning the sidebar into a second content column. The overview uses a 2:1 main/evidence grid. The objective composer and current objectives balance the latest observation and Brainbox facts. Paired cards resolve on shared rows, and both columns finish at approximately the same level through substantive content rather than empty stretching.

### Compact desktop and tablet

The main workflow takes priority while evidence panels move into a secondary row. The icon rail remains until the phone breakpoint. No horizontal scrolling is introduced.

### Phone

The rail becomes a five-item bottom bar: Home, Objectives, Activity, Brain, and Settings. Content becomes one flow with the objective first and current work next. Observation and runtime evidence follow. Dialogs stay within the viewport, navigation respects safe-area insets, and layouts are tested down to 320 px.

## Settings

Settings is a full product view rather than a modal. It contains three groups:

1. **Colour theme:** four palettes and a small data-progress preview.
2. **Objective defaults:** step budget, time limit, and reduced motion.
3. **Local runtime:** proxy endpoint, real connection test, and the explicit simulation boundary.

Changes save automatically on the current device. Existing objectives retain their original budget; new objectives inherit the configured values.

## Accessibility and trust

- Semantic buttons, navigation, sections, labels, tables, dialogs, and live toast region
- Visible focus outlines and keyboard navigation shortcuts
- Reduced-motion media query, local preference, and static capture mode
- Mobile navigation targets remain at least 44 CSS px
- Supporting text is generally 13–14 px; 12 px is reserved for sublines and compact metadata
- Runtime state, demo state, and evaluation limits are written in plain language
- No remote font, tracking, image, or analytics dependency

Before production, still run a formal WCAG contrast audit and assistive-technology pass across every interactive state. Automated layout and source checks are not a substitute for that review.

## Validation in this release

- 99 Playwright browser checks covering themes, workflows, persistence, runtime contracts, offline reload, and responsive overflow
- Taste-skill source lint: **0 failures / 0 warnings**
- Taste-skill real-browser inspection at 1440, 768, and 390 px: **0 failures / 0 warnings**
- Manual review of desktop, tablet, mobile, all four palettes, Settings, and objective-detail captures

## Recommended next increments

1. Add a real task-lifecycle API contract and bind the objective store to actual runtime events.
2. Replace the simulated observation fragment with real screenshots, privacy redaction, and retention controls.
3. Add approval sheets for destructive or externally visible actions, paired with actual undo records.
4. Add empty, offline-runtime, model-missing, RAM-refused, retry-exhausted, and partial-success states.
5. Package the same responsive front end in native shells only if distribution requires it; this release remains a PWA.
