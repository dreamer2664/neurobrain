# Neurobrain Atelier v2 — app design notes

## Product inventory

- Neurobrain is a from-scratch, local desktop brain; the runtime is called brainbox.
- The user writes an objective, not a chat message.
- The runtime loop observes, acts, verifies, and recovers; a claim is not evidence.
- A task has an ID, a budget, append-only observations, a terminal state, and an undo trail.
- Current sample objective: research the boiling point of water and save a note in Docs.
- Current sample observation: source 2 of 3, step 8 of 12, 68% complete, grounded click at 812,405.
- Real repository snapshot: 4.25M LM, 0.44M GripNet, checkpoint 18,600, 312 MB sample RSS, 3.8 GB cap.
- Real evaluation labels: trained-curriculum QA 98%, widget hit 88%, runtime-verified research 30%.
- Operator: one person on one computer, checking progress briefly or leaving a task unattended.
- Main job: understand the current run and hand off the next objective.

## Screen questions and findings

- Overview: What is the brain doing, and what can I hand off next? Finding: research is on source 2 of 3.
- Objectives: Which run needs attention? Finding: one active, one queued, one complete in the sample data.
- Activity: What did the runtime actually observe and verify? Finding: a wrong result was recovered without a blind repeat.
- Brain: What is running, and how well does it honestly perform? Finding: research remains the weakest measured skill at 30%.
- Settings: How should Atelier look and connect on this device? Finding: appearance and runtime preferences save locally.

## Direction

- Style: achromatic-ledger, seed 1806656935.
- Reason: this is a daily working tool; a noun title, flat surfaces, readable 12/13/14/16/24 type, and one near-black action fit better than a greeting or marketing hero.
- Face: low-chroma grounds taken from a quiet desk (stone, blue-grey, neutral rose, true neutral dark), with one palette hue reserved for real data.
- Signature element: the latest visual-grounding observation, drawn as a real interface fragment with its target and coordinates, replacing the decorative neural canvas.
- Redraws: none.

## Palette system

- Linen: stone ground, ink chrome, forest data.
- Mist: blue-grey ground, ink chrome, cobalt data.
- Rose: neutral rose ground, ink chrome, raspberry data.
- Ink: true neutral dark ground, white chrome, sage data.
- Colour themes are selected only in Settings and persist locally.

## Composition

- Desktop: 56px icon rail + fluid canvas. Canvas max 1280px, 32px side gutters, 24px title offset.
- Overview: 2fr main column + 1fr observation rail, one 16px gap. Composer and objectives share the left edge; observation and runtime share the right edge.
- Main surfaces: white (or one dark lightness step), 1px border, 8px radius, no resting shadow or gradient.
- Type: Geist 12/13/14/16/24/30; Geist Mono only for literal code.
- Phone: 16px gutters, labelled bottom navigation with five destinations including Settings, no floating create button.
