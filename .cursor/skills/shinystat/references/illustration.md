# Vector Medical & Scientific Illustration Reference

This reference documents the vector-first scientific schematic engine (`medstat.figures.illustration`), designed to generate high-resolution anatomical structures, cellular biology, tissue cross-sections, and pathophysiological pathways using pure matplotlib and numpy without cloud dependencies.

---

## 1. Architectural Philosophy & Safety Standard

1. **Semantic Color Palette (`SEM`)**:
   - Every anatomical and pathological structure has a persistent semantic color:
     - `erythrocyte` / `rbc`: `#D9534F` (Erythrocyte red)
     - `platelet`: `#F0AD4E` (Thrombocyte amber)
     - `neutrophil`: `#E74C3C` (Multilobed granulocyte red)
     - `macrophage`: `#8E44AD` (Large phagocytic purple)
     - `fibroblast`: `#27AE60` (Spindle matrix green)
     - `bacteria`: `#2C3E50` (Dark rod/coccus navy)
     - `epidermis`: `#E8C4A2` (Stratified squamous keratin)
     - `dermis`: `#F4A261` (Vascular collagen matrix)
     - `hypodermis` / `adipose`: `#FCE8A6` (Adipose lipid yellow)
     - `muscle`: `#C0392B` (Striated myofiber deep red)
     - `bone`: `#E0E0E0` (Cortical osseous light gray)
     - `granulation`: `#E67E22` (Budding capillary orange-red)
     - `slough`: `#F4D03F` (Fibrinous exudative yellow)
     - `eschar`: `#1C1C1C` (Necrotic devitalized black)
2. **Schematic Disclaimer Invariant**:
   - Every exported illustration automatically displays an explicit disclaimer:
     `"Schematic illustration for conceptual mechanism only; not diagnostic patient imaging."`
3. **Traceability & Zero-PHI**:
   - Zero patient identifiers are permitted.
   - Vector PDF (Type 42 fonts), SVG, 600 DPI PNG, and element bounding coordinate CSV are automatically bundled.

---

## 2. Anatomical & Pathological Components

### Cellular Elements (`draw_cell`)
Renders biological cells with harmonic membrane noise, nuclei, and internal granules:
```python
from medstat.figures.illustration import Scene

scene = Scene(width=6.5, height=4.5, journal="NEJM")
# Red blood cell (biconcave disc appearance)
scene.draw_cell("rbc", cx=1.5, cy=2.0, r=0.4)
# Activated neutrophil with multilobed nucleus & granules
scene.draw_cell("neutrophil", cx=3.0, cy=2.0, r=0.6, lobes=3, granules=12)
# Phagocytosing macrophage with pseudopodia
scene.draw_cell("macrophage", cx=5.0, cy=2.0, r=0.8, irregularity=0.25)
```

### Tissue Cross-Sections & Wound Healing (`draw_skin_cross_section`, `draw_wound_phase`)
Renders full layered anatomical cross-sections (stratum corneum, epidermis, papillary/reticular dermis, subcutaneous adipose, fascia, muscle, and periosteum):
```python
from medstat.figures.illustration import Scene

scene = Scene(width=6.5, height=5.0, journal="LANCET")
# Cutaneous tissue cross section with ulcer crater
scene.draw_skin_cross_section(
    x0=0.5,
    x1=6.0,
    crater_depth="dermis",  # "epidermis", "dermis", "subcutis", "muscle", "bone"
    wound_bed=["granulation", "slough"],
)
```

### Pressure Injury Staging (NPUAP / EPUAP)
Visualizes standardized pressure injuries from Stage 1 (non-blanchable erythema) through Stage 4 (full thickness with exposed bone/tendon/muscle), Unstageable (slough/eschar obscured), and Deep Tissue Pressure Injury (DTPI).

### Vascular Anatomy & Hemodynamics (`draw_vessel_stenosis`, `draw_vessel_longitudinal`)
- **Stenosis Cross-Section**: Calculates lumen area reduction percentage and residual luminal diameter with eccentric/concentric atherosclerotic plaque.
- **Longitudinal Venule**: Demonstrates endothelial lining, selectin-mediated leukocyte rolling, adhesion, and trans-endothelial migration (diapedesis).

```python
from medstat.figures.illustration import Scene

scene = Scene(width=6.5, height=4.0, journal="JAMA")
# 75% luminal stenosis cross section
scene.draw_vessel_stenosis(
    cx=3.25,
    cy=2.0,
    radius=1.5,
    stenosis_pct=75.0,
    calcified=True,
    title="Left Anterior Descending (LAD) Stenosis",
)
```

### Signaling & Mechanistic Pathways (`draw_pathway`)
Assembles multi-step cellular cascades with receptor activation arrows (`->`), competitive inhibition bars (`-|`), phosphorylation badges, and downstream gene transcription boxes.
