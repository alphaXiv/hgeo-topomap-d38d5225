import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from statistics import mean, pstdev

    return mean, mo, pstdev


@app.cell
def _(mo):
    mo.md(r"""
    # HGeo-TopoMap: an evidence-first reproduction

    Road centerlines matter for autonomous driving, but they are often not painted. HGeo-TopoMap proposes to help a detector in two ways: give it an estimated road-area map (**GAL**) and make centerlines with similar orientations learn similar representations (**GCL**).

    **Verdict: partially reproduced.** On a compact public OpenLane-V2 split, GAL and scale-adjusted GCL each improved held-out centerline detection, their combination was best, and the combination was more robust to every missing camera. The experiment is directional evidence rather than a full-score replication because it uses 64 public-sample frames, a compact decoder, a substituted road segmenter, and a simplified GCL.
    """)
    return


@app.cell
def _(mean, pstdev):
    seed_scores = {
        "Baseline": [0.0701849409, 0.1532347574, 0.0914258821, 0.1022075756, 0.0785630905, 0.1498381233, 0.0722556723, 0.1255084259],
        "GAL": [0.1055297721, 0.1399767498, 0.1341318425, 0.0495309339, 0.1216136996, 0.1587149500, 0.0804259376, 0.1281390654],
        "GCL, β=0.001": [0.1301579234, 0.1519564683, 0.1126003188, 0.1072129934, 0.1479422780, 0.1564430834, 0.0728305950, 0.1183124963],
        "GAL + GCL": [0.1296618719, 0.1376164748, 0.1129086094, 0.1529571278, 0.1461901081, 0.1448004257, 0.0919986549, 0.1480278938],
    }
    summary_rows = [
        {
            "variant": name,
            "mean DET_l": round(mean(values), 4),
            "population SD": round(pstdev(values), 4),
            "seeds": len(values),
        }
        for name, values in seed_scores.items()
    ]
    return seed_scores, summary_rows


@app.cell
def _(seed_scores):
    _colors = ["#aab5c7", "#4f8cc9", "#8b6fc3", "#167d71"]
    _rows = []
    for _index, (_name, _values) in enumerate(seed_scores.items()):
        _score = sum(_values) / len(_values)
        _width = 100 * _score / 0.15
        _rows.append(
            f"""
            <div style="display:grid;grid-template-columns:140px 1fr 70px;gap:12px;align-items:center;margin:14px 0">
              <strong>{_name}</strong>
              <div style="height:30px;background:#eef1f6;border-radius:7px;overflow:hidden">
                <div style="width:{_width:.1f}%;height:100%;background:{_colors[_index]}"></div>
              </div>
              <strong>{_score:.4f}</strong>
            </div>
            """
        )
    headline_html = (
        '<section style="border:1px solid #dde3ec;border-radius:12px;padding:18px;background:#fbfcfe">'
        '<h3 style="margin-top:0">Held-out centerline detection</h3>'
        '<p>Compact DET_l; eight-seed means. Longer is better.</p>'
        + "".join(_rows)
        + "</section>"
    )
    return (headline_html,)


@app.cell
def _(headline_html, mo):
    mo.Html(headline_html)
    return


@app.cell
def _(mo, summary_rows):
    mo.vstack(
        [
            mo.md(
                r"""
                The paper's official full-benchmark DET_l rises from **29.7 to 31.3 (+1.6)**. This reproduction rises from **0.1054 to 0.1330 (+0.0276, +26.2%)** on a smaller Fréchet-AP scale. Those numbers are not directly comparable; the evidence is the ordering and positive module deltas.
                """
            ),
            mo.ui.table(summary_rows, pagination=False),
        ]
    )
    return


@app.cell
def _():
    missing_baseline = {
        "front center": 0.045059971,
        "front left": 0.056564111,
        "front right": 0.022708146,
        "side left": 0.069426056,
        "side right": 0.046277879,
        "rear left": 0.083748266,
        "rear right": 0.066478263,
    }
    missing_combined = {
        "front center": 0.054904441,
        "front left": 0.076650545,
        "front right": 0.023476832,
        "side left": 0.082474318,
        "side right": 0.059077518,
        "rear left": 0.093835614,
        "rear right": 0.081343064,
    }
    missing_rows = [
        {
            "camera removed": camera,
            "baseline": round(missing_baseline[camera], 4),
            "combined": round(missing_combined[camera], 4),
            "absolute gain": round(missing_combined[camera] - missing_baseline[camera], 4),
        }
        for camera in missing_baseline
    ]
    return missing_baseline, missing_combined, missing_rows


@app.cell
def _(mean, missing_baseline, missing_combined, missing_rows, mo):
    _base_macro = mean(missing_baseline.values())
    _combined_macro = mean(missing_combined.values())
    mo.vstack(
        [
            mo.md(
                f"""
                ## Robustness to a missing camera

                The combined model improves **all seven** single-camera-drop conditions. Its macro score rises from **{_base_macro:.4f}** to **{_combined_macro:.4f}**, a **{100 * (_combined_macro / _base_macro - 1):.1f}%** relative improvement.
                """
            ),
            mo.ui.table(missing_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    beta_scores = {
        "0.1 (paper-disclosed)": 0.0634481974,
        "0.01": 0.0772945926,
        "0.001 (selected)": 0.1246820196,
        "0.0001": 0.1144808259,
    }
    beta_picker = mo.ui.dropdown(
        options=list(beta_scores),
        value="0.001 (selected)",
        label="Inspect a GCL loss weight",
    )
    return beta_picker, beta_scores


@app.cell
def _(beta_picker, beta_scores, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## The consequential scaling choice

                GCL is added to classification and point-regression losses, so its coefficient depends on implementation-specific loss magnitudes. Select a weight to see its observed score.
                """
            ),
            beta_picker,
            mo.callout(
                mo.md(
                    f"Selected **β={beta_picker.value}** → DET_l **{beta_scores[beta_picker.value]:.4f}**. "
                    "Baseline is **0.1054**."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    control_rows = [
        {"control": "Baseline", "DET_l": 0.1054, "reading": "matched control"},
        {"control": "GCL, geometry groups", "DET_l": 0.1247, "reading": "orientation gain"},
        {"control": "GCL, arbitrary index groups", "DET_l": 0.1069, "reading": "near baseline"},
        {"control": "GAL, masked road prior", "DET_l": 0.1148, "reading": "positive standalone gain"},
        {"control": "GAL, masked zero-road prior", "DET_l": 0.1097, "reading": "road content contributes"},
        {"control": "GAL, global road attention", "DET_l": 0.1266, "reading": "mask not isolated alone"},
        {"control": "Combined, masked road prior", "DET_l": 0.1330, "reading": "best overall"},
        {"control": "Combined, zero-road prior", "DET_l": 0.1183, "reading": "road content contributes"},
        {"control": "Combined, global attention", "DET_l": 0.1299, "reading": "mask advantage is small"},
    ]
    mo.vstack(
        [
            mo.md(
                r"""
                ## Mechanism controls

                Arbitrary GCL groups erase the gain, which supports orientation consistency as the useful relation. Removing road occupancy weakens GAL and the combined model. However, global attention beats the GAL-only mask, so this bounded test supports useful road content more strongly than it supports the particular mask.
                """
            ),
            mo.ui.table(control_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experimental recipe

    1. Read 64 frames from the public OpenLane-V2 sample and split each of its two scenes temporally: 48 train, 16 validation.
    2. Cache seven-camera ResNet-18 features. Generate road masks with public SegFormer Cityscapes and project them into bird's-eye coordinates using the dataset calibration.
    3. Train a matched compact TopoLogic-style query decoder for 1,200 steps.
    4. For GAL, encode road occupancy plus position and apply distance-biased prior attention with `k=0.1`.
    5. For GCL, group matched centerlines by orientation/curvature and apply a supervised contrastive loss.
    6. Evaluate normal and seven single-camera-drop conditions with Fréchet-threshold average precision; pool eight seeds.

    **Compute.** Every qualifying run used Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs. Four GPUs were allocated per run; four simultaneous runs reached 16 GPUs. The observed Kubernetes campaign elapsed 46m 32s (0.775613 wall hours).

    **What remains.** A full replication needs the complete OpenLane-V2 benchmark, official TopoLogic evaluator, authors' geometry-aware decoder, and their Mask2Former/Mapillary road priors.

    [Read the illustrated report](https://github.com/alphaXiv/hgeo-topomap-d38d5225/blob/main/reports/hgeo-topomap/report.md) ·
    [Inspect the public repository](https://github.com/alphaXiv/hgeo-topomap-d38d5225)
    """)
    return


if __name__ == "__main__":
    app.run()
