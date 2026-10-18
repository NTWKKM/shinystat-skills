# Journal House Styles & Artwork Engineering Reference

Reached from `SKILL.md` Pillar 5 and **Output Format & House Style Alignment Gate** (#5). Values and formatters live in `medstat.figures.journal` (`JOURNAL_SPECS`, `TARGET_WIDTHS`, `apply_journal_theme`, `style_axis`, `format_estimate`, `format_p_value`, `format_num`) and `medstat.figures.qa` (`qa_layout`, `cvd_check`, `assert_zero_phi_text`).

---

## 1. Supported Journal House Styles (`JOURNAL_SPECS`)

> **Provenance**: The color palettes follow the widely used `ggsci` journal-inspired sets (plus Okabe-Ito for universal color-vision-deficiency accessibility), not official publisher color mandates. Journals specify resolution, vector typography, column widths, and file formats rather than fixed hex palettes; check the target journal's current author artwork guidelines for hard submission requirements.

| Style Code | Target / Inspiration | Palette (`JOURNAL_SPECS[style]["palette"]`) | Frame & Grid (`style_axis`) | Number & P-Value Rules (`format_estimate`, `format_p_value`) |
| :--- | :--- | :--- | :--- | :--- |
| **`"NEJM"`** | *New England Journal of Medicine* | `#BC3C29`, `#0072B5`, `#E18727`, `#20854E`, `#7876B1`, `#6F99AD`, `#FFDC91`, `#EE4C97` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, `grid_y=False`; zoomed inset (`nejm_inset_ylim`, `//` break) for low-event curves | `HR, 1.45 (95% CI, 1.12 to 1.88)`; `P=0.03`, `P<0.001` (keeps leading zero) |
| **`"JAMA"`** | *JAMA* / AMA style | `#374E55`, `#DF8F44`, `#00A1D5`, `#B24745`, `#79AF97`, `#6A6599`, `#80796B`, `#0073C2` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, dotted horizontal y-grid (`grid_y=True`, `#D9D9D9`, 0.6 pt), 3-zone forest layout | `HR, 1.45 (95% CI, 1.12-1.88)`; drops leading zero for bounded values: `P = .03`, `P < .001`, `P > .99` |
| **`"LANCET"`** | *The Lancet* | `#00468B`, `#ED0000`, `#42B540`, `#0099B4`, `#925E9F`, `#FDAF91`, `#AD002A`, `#ADB6B6` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, `grid_y=False`, mid-dot axis tick formatter | Raised mid-dot decimal (`\u00b7`): `HR 1·45 (95% CI 1·12–1·88)`; lowercase `p`: `p=0·01`, `p=0·0004`, `p<0·0001` |
| **`"BMJ"`** | *The BMJ* | `#003366`, `#D52B1E`, `#008080`, `#E87722`, `#6F2C91`, `#2D68C4`, `#708090`, `#118AB2` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, `grid_y=False`, 3-zone forest layout | `HR 1.45 (95% CI 1.12 to 1.88)`; spaced `P = 0.03`, `P < 0.001` |
| **`"CLINICAL"`** | `shinystat` canonical clinical | `#2563EB`, `#DC2626`, `#059669`, `#D97706`, `#7C3AED`, `#0891B2`, `#475569`, `#0F172A` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, dotted horizontal y-grid (`grid_y=True`, `#D9D9D9`, 0.6 pt) | `HR 1.45 (95% CI: 1.12–1.88)`; `P=0.03`, `P<0.001` |
| **`"OKABE_ITO"`** | Universal CVD-safe | `#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`, `#56B4E9`, `#F0E442`, `#000000` | L-frame (`spine_lw=0.8`), outward ticks 3.5 pt, `grid_y=False` | `HR 1.45 (95% CI: 1.12–1.88)`; `P=0.03`, `P<0.001` |

---

## 2. Canvas Geometry & Target Dimensions (`TARGET_WIDTHS`)

 Lock once per manuscript via `apply_journal_theme(journal="NEJM", canvas="manuscript", target="double_col")`:

```python
TARGET_WIDTHS = {
    "single_col": 3.5,     # 1 column (89 mm) — ROC, calibration, single-panel KM
    "one_half_col": 5.0,   # 1.5 column (127 mm) — Bland-Altman, cohort flow, methods pipeline
    "double_col": 7.0,     # 2 columns / full width (178 mm) — 3-zone forest, CONSORT/PRISMA/STARD
    "docx": 6.5,           # Word manuscript figure width (base_font = 10.0 pt)
    "pptx": 11.5,          # 16:9 widescreen slide width (base_font = 13.0 pt)
    "web": 7.5,            # Interactive HTML report width (base_font = 10.0 pt)
}
```

### Canvas Modes
- **`"manuscript"`** (default): Pure white background (`#FFFFFF`), `#222222` ink, calibrated for journal submission and print.
- **`"editorial"`**: Warm ivory background for `"NEJM"` (`#FBF7F0`, `#FFFFFF` for other styles), suited for slides and editorial previews.

---

## 3. Vector Typography & Export Standards

Every journal figure exports a synchronized bundle (`FigureResult`):
1. **High-DPI Raster (`png_path`)**:
   - Statistical graphs: **300 DPI**.
   - Study flow diagrams & **schematic** illustrations: **600 DPI**.
2. **Editable Vector Formats (`vector_path`, `svg_path`)**:
   - PDF exported with `mpl.rcParams["pdf.fonttype"] = 42` and `ps.fonttype = 42` (TrueType), plus `svg.fonttype = "none"` and Thai-capable sans-serif fallback chain (`Arial`, `Helvetica`, `DejaVu Sans`, `Sarabun`, `Thonburi`, `Sukhumvit Set`, `Tahoma`).
3. **Traceable Source CSV (`csv_path`)**:
   - Statistical figures export plotted coordinates and 95% CIs (plus `_risktable.csv` for survival).
   - Flow diagrams export **reconciled** step counts and equation checks (`_source.csv`).
   - **Schematic** illustrations export drawn element parameters and seed (`_source.csv`).

---

## 4. Standardized Medical Formatting (`medstat.figures.journal`)

```python
from medstat.figures.journal import format_estimate, format_num, format_p_value

format_estimate(1.42, 1.12, 1.81, metric="OR", journal="NEJM")    # "OR, 1.42 (95% CI, 1.12 to 1.81)"
format_estimate(1.42, 1.12, 1.81, metric="OR", journal="JAMA")    # "OR, 1.42 (95% CI, 1.12-1.81)"
format_estimate(1.42, 1.12, 1.81, metric="OR", journal="LANCET")  # "OR 1·42 (95% CI 1·12–1·81)"
format_estimate(1.42, 1.12, 1.81, metric="OR", journal="BMJ")     # "OR 1.42 (95% CI 1.12 to 1.81)"

format_p_value(0.00003, journal="NEJM")    # "P<0.001"
format_p_value(0.031, journal="JAMA")      # "P = .03" (bare=True -> ".03")
format_p_value(0.0004, journal="LANCET")   # "p=0·0004"
format_p_value(0.031, journal="BMJ")       # "P = 0.03"
```

---

## 5. Automated Publication Quality Assurance Gates (`medstat.figures.qa`)

1. **Layout Collision Gate (`qa_layout`)**: Flags text-box bounding box overlaps greater than `tol_px=2.0` pixels in both axes, canvas clipping (statistical plotters use `check_outside=False` because `bbox_inches='tight'` expands the canvas to encompass outer tick labels, while `Scene.qa` uses `check_outside=True`), and unreadable text (`min_font=6.0`), returning `FigureResult.qa["passed"]`. Inspect the rendered PNG visually after `qa["passed"] == True`, as bounding-box checks catch text collisions rather than aesthetic composition.
2. **CVD Accessibility Gate (`cvd_check`)**: Simulates deuteranopia, protanopia, and tritanopia (Machado 2009 matrices) using illustration tissue colors (`SEM`) by default and flags pairs with $\Delta E < 12.0$ (`needs_cue == True`) so hatches or direct labels can be added.
3. **Zero-PHI Gate (`assert_zero_phi_text`)**: Screens plotter-specific text lists (caller-provided titles, units, variable lists, labels, annotations, captions, notes, and rendered text artists across all plotters) against `medstat._phi.PHI_PATTERNS` (`HN`, 13-digit Thai IDs, phone numbers, DOB, honorific-prefixed English and Thai names) and raises `ValueError` on any match.
