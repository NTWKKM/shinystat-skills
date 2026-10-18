# Vector Medical & Scientific Illustration Reference (`medstat.figures.illustration`)

Reached from `SKILL.md` Pillar 5. Generates deterministic, seeded vector-first **schematic** illustrations (`Scene`, `SEM`) using `matplotlib` and `numpy` without cloud dependencies.

---

## 1. Semantic Palette (`SEM`) & Safety Invariants

1. **Semantic Color Palette (`SEM`)** (one biological meaning = one persistent color across figures):
   - **Tissue & Wound**: `epidermis` (`#E9B99D`), `dermis` (`#F3D3BC`), `hypodermis` (`#F7E7B0`), `muscle` (`#B8504F`), `bone` (`#EFE9DA`), `skin_surface` (`#F6DCC8`), `epithelial` (`#F2BDB5`), `granulation` (`#D8575A`), `slough` (`#E4CF74`), `necrotic` (`#3C302B`), `erythema` (`#F0B7A6`), `collagen` (`#E8DCC0`), `scar` (`#EADFD6`).
   - **Cells & Microbes**: `rbc` (`#D9433C`), `rbc_pallor` (`#EE8A84`), `platelet` (`#B58BD6`), `neutrophil` (`#F0E6F3`), `lymphocyte` (`#DCE6F4`), `macrophage` (`#EBCBA6`), `bacteria` (`#3E8E5B`), `nucleus` (`#5A4A9C`), `nucleus_dark` (`#3F3480`), `cyto` (`#F4DDEA`).
   - **Vessels & Fluid**: `blood` (`#B5282E`), `plasma` (`#F9E6DC`), `fibrin` (`#8A7F72`), `clot` (`#8E2B31`), `lipid` (`#F1D88A`), `cap` (`#E9DCCB`), `lumen` (`#FBEDEA`), `vessel_wall` (`#E7A2A0`), `adventitia` (`#EFE3CF`), `endothelium` (`#E8B0A8`), `outline` (`#3A2E2E`).
2. **Schematic Disclaimer & Geometric Verifiability**:
   - Quantitative features: `artery_cross_section` stenosis is geometrically constructed and logged (stenosis in `[0, 1)` validated before geometry calculation) but not verified against the requested value, whereas `wound_topdown` ring fractions are checked via `shoelace` against `tol` and logged to `_source.csv`.
   - `Scene.save(strict=True, ...)` enforces `assert_zero_phi_text` across all text elements and notes, checks layout QA (raising `RuntimeError` if failed when `strict=True`), appends the default **schematic** disclosure (`"Schematic illustration; not to scale; not derived from patient images."`) to `FigureResult.caption` (not drawn on canvas; add an on-figure disclaimer via `sc.text` if an on-canvas disclaimer is required), and exports 600 DPI PNG + TrueType PDF (`fonttype=42`) + SVG + `_source.csv`. Check `cvd_check()` pairs (`TISSUE_MUST_DIFFER`) and add hatches (`hatch=True`) or direct callouts where `needs_cue == True`.

---

## 2. `Scene` API & Anatomical Primitives

Initialize canvas (`x in [0, 100]`, `y in [0, 100 * aspect]`):
```python
from medstat.figures.illustration import SEM, Scene

sc = Scene(width="double_col", aspect=0.55, journal="NEJM", canvas="manuscript", seed=7)
```

### Cells (`rbc`, `platelet`, `neutrophil`, `lymphocyte`, `macrophage`, `fibroblast`, `bacterium`, `eukaryote`)
```python
sc.rbc(cx=12.0, cy=20.0, r=1.3, rot=0.0, tilt=0.55)
sc.platelet(cx=18.0, cy=20.0, r=0.7)
sc.neutrophil(cx=26.0, cy=20.0, r=2.2, lobes=3, squeeze=1.0, rot=0.0)
sc.lymphocyte(cx=34.0, cy=20.0, r=1.8)
sc.macrophage(cx=44.0, cy=20.0, r=3.2, vacuoles=4, engulfed=2)
sc.fibroblast(cx=56.0, cy=20.0, L=5.0, rot=0.1)
sc.bacterium(cx=66.0, cy=20.0, L=1.6, kind="bacillus")  # "bacillus"|"coccus"|"diplococcus"|"chain"|"cluster"
sc.eukaryote(cx=82.0, cy=20.0, r=9.0, organelles=True)
```

### Tissue Cross-Sections, Wound Phases & Pressure Injury Staging
```python
# Layered skin block: thick = (epidermis, dermis, hypodermis, muscle)
geo = sc.skin(x0=5.0, x1=95.0, ytop=46.0, thick=(3.0, 11.0, 9.0, 9.0), rete=0.9)

# Ulcer crater & wound healing phase ("hemostasis" | "inflammation" | "proliferation" | "remodeling")
cr = sc.crater(geo, xc=35.0, width=18.0, depth=10.0)
sc.wound_phase(geo, cr, xc=35.0, width=18.0, depth=10.0, phase="proliferation")

# NPIAP/EPUAP pressure injury stage (stage in {1, 2, 3, 4})
sc.pressure_injury(geo, xc=72.0, stage=3)

# Top-down wound bed with shoelace-verified tissue area fractions (tol=0.01)
ach = sc.wound_topdown(
    cx=80.0, cy=25.0, r=10.0,
    fractions={"epithelial": 0.15, "granulation": 0.50, "slough": 0.25, "necrotic": 0.10},
    hatch=True,
)
```

### Vessels & Hemodynamics (`artery_cross_section`, `vessel_longitudinal`, `selectin_lollipop`)
```python
# Eccentric atherosclerotic plaque with exact geometric area stenosis
info = sc.artery_cross_section(
    cx=30.0, cy=28.0, R=14.0, stenosis=0.70, lipid_core=True, rupture=True, thrombus=0.4
)

# Post-capillary venule longitudinal wall & lumen (for leukocyte extravasation)
ven = sc.vessel_longitudinal(x0=55.0, x1=95.0, y_floor=20.0, lumen_h=14.0, wall=3.0, tissue_h=12.0)
sc.selectin_lollipop(x=68.0, y=ven["floor"], h=1.8)
```

### Signaling & Mechanistic Pathways (`node`, `edge`, `cascade`)
```python
# kind in {"stimulus", "molecule", "cell", "process", "outcome"}
sc.cascade(
    rows=[
        [("lps", "Endotoxin (LPS)", "stimulus")],
        [("tlr4", "TLR4 / NF-κB", "molecule"), ("il6", "IL-6 / TNF-α", "molecule")],
        [("shock", "Endothelial Barrier Failure", "outcome")],
    ],
    y_top=46.0,
    y_bot=12.0,
)
sc.edge("lps", "tlr4", kind="activate", label="MyD88")
sc.edge("tlr4", "il6", kind="activate")
sc.edge("il6", "shock", kind="activate")
```

### Annotations & Bundle Export (`callout`, `label_column`, `legend_swatches`, `scale_bar`, `badge`, `save`)
```python
sc.badge(2.0, 52.0, "A")
sc.callout("Fibrous cap rupture", target=info["plaque_center"], text_xy=(48.0, 38.0))
res = sc.save(
    out_path="figures/mechanism_schematic.png",
    caption_title="Pathophysiological Mechanism Schematic",
    alt_text="Vector schematic of plaque rupture and inflammatory cascade",
    dpi=600,
    strict=True,
)
```
