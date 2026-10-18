# Journal House Styles & Artwork Engineering Reference

This reference documents the journal-grade styling engine (`medstat.figures.journal`), vector typography rules, canvas targets, and publication quality assurance gates integrated into `shinystat`.

---

## 1. Supported Journal House Styles

| Style Code | Journal Target | Primary Palette (First 4) | Typography Spines | Inset / Feature Support |
| :--- | :--- | :--- | :--- | :--- |
| **`"NEJM"`** | *New England Journal of Medicine* | `#941100` (Burgundy), `#1A365D` (Navy), `#2D3748`, `#4A5568` | Crisp L-frame (left/bottom only), tick outward 3.5 pt, top-aligned metrics | Inset zoom for low-event curves (`//` axis break, 0–20% range) |
| **`"JAMA"`** | *Journal of the American Medical Association* | `#002D62` (Deep Navy), `#BA0C2F` (Crimson), `#5C768D`, `#C59B27` | Bold headers, 3-zone structured forest layout, uppercase column titles | Direct odds annotations, zero gridlines |
| **`"LANCET"`** | *The Lancet* | `#D62728` (Lancet Red), `#1F77B4` (Slate Blue), `#FF7F0E`, `#2CA02C` | Compact risk tables, explicit tick bounds, scientific neutral spines | Multi-panel cohort panel sync, zero-origin |
| **`"BMJ"`** | *The BMJ* (British Medical Journal) | `#004F9F` (BMJ Blue), `#E05A47` (Coral Red), `#2E8B57`, `#F4A261` | Minimalist L-frame, centered effect direction labels, sans-serif Sarabun | Clear favors headers, wrapped forest headers |
| **`"CLINICAL"`** | *Nature / General Clinical* | `#0072B2` (Okabe Blue), `#D55E00` (Vermilion), `#009E73`, `#CC79A7` | High-accessibility CVD-safe Okabe-Ito, clean neutral frame | Universal fallback, high-contrast |

---

## 2. Canvas Geometry & Target Dimensions

Figures are calibrated to standard journal print grids using `TARGET_WIDTHS`:

```python
TARGET_WIDTHS = {
    "single_col": 3.25,        # 1 column (82 mm) — ROC, calibration, univariate KM
    "one_and_half_col": 5.0,   # 1.5 column (127 mm) — Bland-Altman, 2-panel KM
    "double_col": 6.8,         # 2 columns / full width (173 mm) — Forest plots, CONSORT/PRISMA flows
    "slide_16_9": 11.5,        # Executive PowerPoint presentation (16:9 widescreen)
}
```

### Canvas Modes
- **`"manuscript"`** (default): Pure white background (`#FFFFFF`), standard black ink (`#222222`), suitable for direct submission to publisher peer review and printing.
- **`"editorial"`**: Warm off-white background (`#FDFBF7` for NEJM, `#F8F9FA` for BMJ), designed for online article previews, interactive HTML dashboards, and executive slides.

---

## 3. Vector Typography & Export Standards

All journal figures produce multi-format bundles:
1. **High-DPI Raster (PNG)**:
   - Statistical graphs: **300 DPI**.
   - Flow diagrams & medical illustrations: **600 DPI**.
2. **Vector Formats (PDF & SVG)**:
   - PDF exported with `matplotlib.rcParams["pdf.fonttype"] = 42` (TrueType / Type 42), guaranteeing text remains editable paths without font rasterization or substitution.
   - SVG generated for responsive web presentation and interactive web reports.
3. **Traceability CSV (`_source.csv`)**:
   - Statistical figures export all underlying plotted coordinates and 95% CIs.
   - Flow diagrams export bounding box coordinates `[x, y, w, h]`, text contents, sample counts, and reconciliation status.
   - Illustrations export semantic element IDs, types, anatomical coordinates, and seed metadata.

---

## 4. Standardized Medical Formatting Rules

The journal engine enforces consistent clinical reporting standards via `medstat.figures.journal`:

### Effect Estimates (`format_estimate`)
```python
from medstat.figures.journal import format_estimate

# Odds ratio with 95% CI
text = format_estimate(1.42, 1.12, 1.81, metric="OR", journal="NEJM")
# Returns: "OR, 1.42 (95% CI, 1.12-1.81)"
```
- In Lancet style: uses comma separators `"OR 1·42 (95% CI 1·12–1·81)"` (middle dot where required).
- In NEJM style: uses standard comma spacing `"OR, 1.42 (95% CI, 1.12-1.81)"`.

### P-Values (`format_p_value`)
```python
from medstat.figures.journal import format_p_value

format_p_value(0.00003, journal="NEJM")  # "P<0.001"
format_p_value(0.042, journal="NEJM")    # "P=0.04"
format_p_value(0.891, journal="JAMA")    # "P = .89" (JAMA omits leading zero)
```

---

## 5. Automated Publication Quality Assurance Gates

Every journal-grade figure is checked through `medstat.figures.qa`:

### Layout Collision Gate (`qa_layout`)
- Computes pixel bounding boxes across all text elements, tick labels, axis labels, legends, and annotations.
- Automatically calculates pairwise intersection-over-minimum (IoM). If overlap exceeds 5%, an issue is flagged.
- Flags unconstrained legend boxes or overlapping tick numbers.

### CVD Accessibility Gate (`cvd_check`)
- Simulates deuteranopia, protanopia, and tritanopia using Brettel (1997) / Machado (2009) LMS transformation matrices.
- Checks minimum Euclidean $\Delta E$ color difference across all distinct plotted series in CVD color space.
- Warns if contrast is insufficient for red-green color-blind reviewers.

### Zero-PHI Gate (`assert_zero_phi_text`)
- Automated regex audit of all title strings, axis labels, legend items, and text annotations against Hospital Numbers (HN), 13-digit Thai National IDs, phone numbers, and patient names.
- Raises immediate exception if unmasked patient data appears in figure text.
