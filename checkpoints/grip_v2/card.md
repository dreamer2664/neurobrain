# Checkpoint Card — `grip_v2` (`EyesGrounder`: Two-Stage Coarse FPN + Native CRNN Strip Refiner)

## 1. Summary & Architecture

- **Directory**: `checkpoints/grip_v2/` (`config.json`, `weights.pt`, `weights.int8.pt`, `tokenizer.json`, `card.md`)
- **Trained From Scratch**: Yes (`0` pretrained weights, pure PyTorch CPU, 2 threads)
- **Parameters**: **1,568,492** total
  - `Stage1CoarseNet` (`128x128` -> `32x32` ResNet-D anti-aliased FPN + 7-channel polarity-invariant stem + FiLM/4-head cosine cross-correlation): **882,460** params (`Top-7` widget-center recall = **99.8%**)
  - `Stage2RefineNet` (`32x160` 1:1 native-pixel candidate-pinned horizontal strip CRNN: `stride=1` stroke stem + `(8x1)` vertical glyph collapse + learned candidate-relative column embeddings + 1-layer Horizontal `BiGRU` + sub-pixel residual pointer): **686,032** params
- **Training Data & Schedule**:
  - **Stage 1**: `2,100` steps (`batch=16`) on `data/ui_hard_v1` (`50,000` SHA-256 verified Hard World v1 samples across 10 shards, `391 MB`) with online palette/polarity augmentation.
  - **Stage 2**: `3,550` steps (`8` screens/step × `6` sibling strips + `1` spatial-rejection negative = `~200k` strips / `~1.05M` pairwise in-screen OCR comparisons across `8,700` native `256x256` screens / `~71,000` active widgets) using `easy -> medium -> crowded` curriculum + online label-confusion hard-negative mining.

## 2. Headline Evaluation Metrics (2 CPU Threads)

| Split / Mode | Widget Hit Rate | Median Centre Error | p90 Error | Action Acc | Latency (2 thr) | Disk Size |
|---|---:|---:|---:|---:|---:|---:|
| **Hard World Seen (`N=1000`, `fp32`)** | **98.00%** (`980/1000`) | **0.68 px** | 1.87 px | **100.00%** | **31.54 ms** | 6.34 MB |
| **Hard World Unseen OOD (`N=500`, `fp32`)** | **91.00%** (`455/500`) | **0.81 px** | — | 100.00% | 31.54 ms | 6.34 MB |
| **Hard World Seen (`int8_storage`)** | **98.00%** (`+0.00 pts`) | **0.69 px** | 1.91 px | 100.00% | 31.49 ms (`-0.05 ms`) | **1.67 MB (`3.81x`)** |
| **Hard World Seen (`int8_dynamic`)** | **98.10%** (`+0.10 pts`) | **0.68 px** | 1.90 px | 100.00% | 28.26 ms (`-3.28 ms`) | **1.67 MB (`3.81x`)** |

- **Peak Runtime RSS**: `585.9 MB` (evaluation harness including pre-rendered split; standalone grounder process `< 290 MB`).
- **Recovery-Conditioned Grounding (`exclude_box`)**: `242/250` (**96.8%**) excluded-region avoidance; **96.8%** target hit when masking a distractor widget.

## 3. Generalisation Table — Seen vs Unseen Fonts, Themes, Widget Sizes, and Roles

### By Font Family (Seen vs Unseen OOD)
| Split | Font | N | Hit Rate | Median Centre Error |
|---|---|---:|---:|---:|
| Seen | `DejaVuSans` | 269 | **97.77%** | 0.67 px |
| Seen | `DejaVuSans-Bold` | 239 | **99.16%** | 0.72 px |
| Seen | `DejaVuSansMono` | 255 | **97.65%** | 0.69 px |
| Seen | `DejaVuSerif` | 237 | **97.47%** | 0.58 px |
| **Unseen OOD** | `DejaVuSansMono-BoldOblique` | 57 | **92.98%** | 0.86 px |
| **Unseen OOD** | `DejaVuSansMono-Oblique` | 64 | **95.31%** | 0.74 px |
| **Unseen OOD** | `DejaVuSerif-Bold` | 64 | **93.75%** | 0.75 px |
| **Unseen OOD** | `DejaVuSerif-Italic` | 62 | **93.55%** | 0.86 px |
| **Unseen OOD** | `FreeSans` | 42 | **95.24%** | 0.84 px |
| **Unseen OOD** | `FreeSerif` | 61 | **86.89%** | 0.80 px |
| **Unseen OOD** | `LiberationMono-Regular` | 51 | **82.35%** | 0.73 px |
| **Unseen OOD** | `LiberationSans-Regular` | 62 | **95.16%** | 0.82 px |
| **Unseen OOD** | `LiberationSansNarrow-Regular` | 37 | **78.38%** | 0.93 px |

### By Theme (Seen vs Unseen OOD Palettes)
| Split | Theme | N | Hit Rate | Median Centre Error |
|---|---|---:|---:|---:|
| Seen | `dark` | 351 | **97.15%** | 0.68 px |
| Seen | `high_contrast` | 331 | **97.28%** | 0.69 px |
| Seen | `light` | 318 | **99.69%** | 0.64 px |
| **Unseen OOD** | `high_contrast_amber` | 159 | **88.68%** | 0.85 px |
| **Unseen OOD** | `nord_slate` | 150 | **92.67%** | 0.92 px |
| **Unseen OOD** | `solarized_light` | 191 | **91.62%** | 0.70 px |

### By Widget Size (`small` / `medium` / `large`)
| Split | Size Bucket | N | Hit Rate | Median Centre Error |
|---|---|---:|---:|---:|
| Seen | `small` | 733 | **98.23%** | 0.62 px |
| Seen | `medium` | 197 | **96.45%** | 0.88 px |
| Seen | `large` | 70 | **100.00%** | 0.84 px |
| **Unseen OOD** | `small` | 372 | **91.94%** | 0.79 px |
| **Unseen OOD** | `medium` | 91 | **87.91%** | 0.90 px |
| **Unseen OOD** | `large` | 37 | **89.19%** | 0.93 px |

### By Widget Role
| Role | Seen Hit (`N=1000`) | Seen Med Err | Unseen OOD Hit (`N=500`) | OOD Med Err |
|---|---:|---:|---:|---:|
| `button` | **99.45%** (`n=181`) | 0.70 px | **81.11%** (`n=90`) | 0.99 px |
| `checkbox` | **96.91%** (`n=97`) | 0.80 px | **93.18%** (`n=44`) | 0.76 px |
| `field` | **98.10%** (`n=105`) | 0.78 px | **96.23%** (`n=53`) | 0.69 px |
| `icon` | **96.05%** (`n=152`) | 0.66 px | **95.52%** (`n=67`) | 0.86 px |
| `menu` | **97.45%** (`n=196`) | 0.68 px | **92.16%** (`n=102`) | 0.78 px |
| `slider` | **97.80%** (`n=91`) | 0.77 px | **90.00%** (`n=50`) | 1.07 px |
| `tab` | **99.44%** (`n=178`) | 0.52 px | **92.55%** (`n=94`) | 0.61 px |

## 4. Remaining Failure Modes (`2.0%` Seen / `9.0%` Unseen OOD)

- `0 / 1000` background misses (`background_far = 0`), `0 / 1000` border near-misses, `0 / 1000` wrong-role misses (`action_acc = 100.0%`).
- All `20 / 1000` seen misses are rare same-role label confusions on heavily JPEG-degraded `8 px` narrow widgets (`icon = 96.1%`, `checkbox = 96.9%`, `menu = 97.5%`) or 3-char truncated labels.
- On unseen OOD fonts (`FreeSerif` serif strokes & `FreeMono` wide fixed-pitch glyphs), `button` drops from `99.5% -> 81.1%` when wide monospace labels truncate to 3–4 letters, while `field` (`96.2%`), `icon` (`95.5%`), `checkbox` (`93.2%`), `tab` (`92.6%`), and `menu` (`92.2%`) all stay above `92%`.
