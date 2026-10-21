#!/usr/bin/env python3
"""Build the TabPFGen reproduction presentation with speaker notes."""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Inches, Pt


REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "outputs" / "analysis"
OUT = REPO / "outputs" / "presentation" / "tabpfgen_reproduction_results_visual.pptx"

NAVY = RGBColor(15, 23, 42)
MUTED = RGBColor(71, 85, 105)
LIGHT = RGBColor(241, 245, 249)
GRID = RGBColor(226, 232, 240)
BLUE = RGBColor(37, 99, 235)
TEAL = RGBColor(14, 165, 166)
AMBER = RGBColor(245, 158, 11)
RED = RGBColor(220, 38, 38)
GREEN = RGBColor(22, 163, 74)
PURPLE = RGBColor(124, 58, 237)
WHITE = RGBColor(255, 255, 255)


def load_data():
    data = {}
    for name in [
        "table1_vs_paper",
        "synthetic_diagnostics_summary",
        "downstream_metrics_long",
        "ctgan_repeats_summary",
        "worst_delta_vs_original",
        "best_delta_vs_original",
    ]:
        path = ANALYSIS / f"{name}.csv"
        data[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    extra = REPO / "outputs" / "analysis_extra"
    privacy = REPO / "outputs" / "privacy_attacks"
    data["extra_synthetic_diagnostics_summary"] = (
        pd.read_csv(extra / "extra_synthetic_diagnostics_summary.csv")
        if (extra / "extra_synthetic_diagnostics_summary.csv").exists()
        else pd.DataFrame()
    )
    data["privacy_attacks_summary"] = (
        pd.read_csv(privacy / "privacy_attacks_summary.csv")
        if (privacy / "privacy_attacks_summary.csv").exists()
        else pd.DataFrame()
    )
    return data


def fmt(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def add_text(slide, text, x, y, w, h, size=16, color=NAVY, bold=False, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    if align:
        p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 0.55, 0.35, 12.0, 0.45, size=28, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.57, 0.86, 11.7, 0.30, size=12.5, color=MUTED)


def add_footer(slide, text="Source: current outputs/analysis CSVs and reproduction scripts"):
    add_text(slide, text, 0.55, 7.15, 8.5, 0.18, size=8.5, color=RGBColor(100, 116, 139))


def add_note_marker(slide, number, x, y, color=BLUE):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(0.33), Inches(0.33))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = color
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = str(number)
    run.font.name = "Arial"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = WHITE


def card(slide, x, y, w, h, title, value, color=BLUE, body=None):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = GRID
    shape.line.width = Pt(1)
    add_text(slide, value, x + 0.15, y + 0.16, w - 0.3, 0.36, size=22, color=color, bold=True)
    add_text(slide, title, x + 0.15, y + 0.58, w - 0.3, 0.28, size=10.5, color=MUTED)
    if body:
        add_text(slide, body, x + 0.15, y + 0.96, w - 0.3, h - 1.05, size=10.5, color=NAVY)


def table(slide, df, x, y, w, h, font=10.5, header_fill=NAVY):
    rows, cols = df.shape[0] + 1, df.shape[1]
    tbl = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for c, col in enumerate(df.columns):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.text = str(col)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(font)
                r.font.bold = True
                r.font.color.rgb = WHITE
    for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            if r_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font)
                    run.font.color.rgb = NAVY
    return tbl


def main_result_table(slide, t1, mode, value_col, x, y, w, h, font=7.2, title=None):
    generator_order = ["original", "smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"]
    model_order = ["xgb", "rf", "lr", "tabpfn"]
    sub = t1[t1["mode"] == mode].copy()
    sub["generator"] = pd.Categorical(sub["generator"], generator_order, ordered=True)
    sub["downstream_model"] = pd.Categorical(sub["downstream_model"], model_order, ordered=True)
    pivot = (
        sub.pivot_table(
            index="downstream_model",
            columns="generator",
            values=value_col,
            observed=False,
        )
        .reindex(index=model_order, columns=generator_order)
        .reset_index()
    )
    pivot["downstream_model"] = pivot["downstream_model"].map(lambda s: str(s).upper())
    for col in generator_order:
        pivot[col] = pivot[col].map(lambda v: "NA" if pd.isna(v) else f"{float(v):.3f}")
    pivot.columns = ["Model", "Original", "SMOTE", "CTGAN", "TVAE", "NF", "RTVAE", "TabDDPM", "TabPFGen"]
    if title:
        add_text(slide, title, x, y - 0.33, w, 0.25, size=12.2, color=NAVY, bold=True)
    return table(slide, pivot, x, y, w, h, font=font)


def delta_table(slide, t1, mode, x, y, w, h, font=7.0, title=None):
    generator_order = ["smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"]
    model_order = ["xgb", "rf", "lr", "tabpfn"]
    sub = t1[(t1["mode"] == mode) & (t1["generator"].isin(generator_order))].copy()
    sub["generator"] = pd.Categorical(sub["generator"], generator_order, ordered=True)
    sub["downstream_model"] = pd.Categorical(sub["downstream_model"], model_order, ordered=True)
    pivot = (
        sub.pivot_table(
            index="downstream_model",
            columns="generator",
            values="delta_vs_paper",
            observed=False,
        )
        .reindex(index=model_order, columns=generator_order)
        .reset_index()
    )
    pivot["downstream_model"] = pivot["downstream_model"].map(lambda s: str(s).upper())
    for col in generator_order:
        pivot[col] = pivot[col].map(lambda v: "NA" if pd.isna(v) else f"{float(v):+.3f}")
    pivot.columns = ["Model", "SMOTE", "CTGAN", "TVAE", "NF", "RTVAE", "TabDDPM", "TabPFGen"]
    if title:
        add_text(slide, title, x, y - 0.33, w, 0.25, size=12.2, color=NAVY, bold=True)
    return table(slide, pivot, x, y, w, h, font=font, header_fill=TEAL)


def bar_chart(slide, labels, values, x, y, w, h, max_value=None, color=BLUE, title=None, suffix=""):
    if title:
        add_text(slide, title, x, y - 0.28, w, 0.22, size=11.5, color=NAVY, bold=True)
    maxv = max_value or max(values or [1])
    row_h = h / max(len(labels), 1)
    for i, (lab, val) in enumerate(zip(labels, values)):
        yy = y + i * row_h
        add_text(slide, str(lab), x, yy + 0.02, 1.45, row_h * 0.75, size=9.5, color=MUTED)
        bg = slide.shapes.add_shape(1, Inches(x + 1.55), Inches(yy + 0.07), Inches(w - 2.15), Inches(row_h * 0.45))
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT
        bg.line.color.rgb = LIGHT
        bw = (w - 2.15) * (float(val) / maxv if maxv else 0)
        fg = slide.shapes.add_shape(1, Inches(x + 1.55), Inches(yy + 0.07), Inches(max(0.02, bw)), Inches(row_h * 0.45))
        fg.fill.solid()
        fg.fill.fore_color.rgb = color
        fg.line.color.rgb = color
        add_text(slide, f"{val:.3f}{suffix}", x + w - 0.52, yy, 0.55, row_h * 0.75, size=9.2, color=NAVY)


def make_deck(data):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    notes = []

    t1 = data["table1_vs_paper"]
    downstream = data["downstream_metrics_long"]
    synth = data["synthetic_diagnostics_summary"]
    ctgan = data["ctgan_repeats_summary"]
    extra = data["extra_synthetic_diagnostics_summary"]
    privacy = data["privacy_attacks_summary"]

    n_metrics = len(downstream)
    missing_cells = 13
    tabpfgen_gap = t1.loc[t1["generator"] == "tabpfgen", "abs_delta_vs_paper"].mean()
    repl_gap = t1.loc[t1["mode"] == "replacement", "abs_delta_vs_paper"].mean()

    slide = prs.slides.add_slide(blank)
    add_text(slide, "TabPFGen reproduction", 0.55, 0.55, 8.0, 0.75, size=40, bold=True)
    add_text(slide, "Current evidence, known failure modes, and next evaluation tracks", 0.58, 1.35, 8.5, 0.35, size=16, color=MUTED)
    card(slide, 0.7, 2.25, 2.65, 1.25, "downstream metric files", f"{n_metrics:,}", BLUE)
    card(slide, 3.65, 2.25, 2.65, 1.25, "missing planned cells", str(missing_cells), RED)
    card(slide, 6.6, 2.25, 2.65, 1.25, "TabPFGen mean gap", fmt(tabpfgen_gap), TEAL)
    card(slide, 9.55, 2.25, 2.65, 1.25, "replacement gap", fmt(repl_gap), AMBER)
    add_text(slide, "Updated with raw-logit sampling, CTGAN repeated seeds, gradient-health logging, extra fidelity diagnostics, and privacy attacks.", 0.7, 4.15, 11.4, 0.7, size=17, color=NAVY)
    add_text(slide, "No extra/privacy CSV results were present locally yet, so those slides describe the runnable evaluation tracks and expected readouts.", 0.7, 5.05, 11.4, 0.42, size=12.5, color=MUTED)
    add_footer(slide, "Source: outputs/analysis CSVs plus newly added scripts in scripts/")
    notes.append("Set up the story. The current reproduction has thousands of downstream metric files, but 13 planned cells still need targeted reruns. The new part is that the deck now includes the raw-logit implementation fix, CTGAN repeat results, and the extra evaluation tracks we just added.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Table 1 reproduction gap", "Mean absolute AUC difference against the paper, lower is closer")
    gen_gap = (
        t1.dropna(subset=["abs_delta_vs_paper"])
        .groupby("generator")["abs_delta_vs_paper"]
        .mean()
        .sort_values()
    )
    bar_chart(slide, gen_gap.index.tolist(), gen_gap.values.tolist(), 0.65, 1.65, 5.9, 4.4, color=BLUE, title="By generator")
    mode_gap = (
        t1.dropna(subset=["abs_delta_vs_paper"])
        .groupby("mode")["abs_delta_vs_paper"]
        .mean()
        .reindex(["augmentation", "replacement"])
    )
    bar_chart(slide, mode_gap.index.tolist(), mode_gap.values.tolist(), 7.1, 1.85, 4.8, 1.4, color=AMBER, title="By training mode")
    add_text(slide, "Closest: SMOTE and TabPFGen. Largest mismatch: CTGAN, mainly because replacement-mode performance is unstable and often much lower than the paper.", 7.1, 3.85, 5.0, 1.0, size=15, color=NAVY)
    add_footer(slide)
    notes.append("Use this slide to summarize where the reproduction agrees with the paper. The important point is not that every number matches exactly, but that TabPFGen is among the closer methods while CTGAN has the biggest gap.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Main table comparison: augmentation", "Average AUC in the paper versus the current reproduced summary")
    main_result_table(slide, t1, "augmentation", "paper_auc", 0.55, 1.35, 5.85, 1.95, title="Paper")
    main_result_table(slide, t1, "augmentation", "reproduced_auc_mean", 6.85, 1.35, 5.85, 1.95, title="My results")
    delta_table(slide, t1, "augmentation", 0.55, 4.25, 12.15, 1.95, title="Delta: my result minus paper")
    add_text(slide, "Positive delta means my reproduced AUC is higher than the paper. NA marks a cell that is not present in the current summary file.", 0.75, 6.55, 11.6, 0.34, size=11.8, color=MUTED)
    add_footer(slide, "Source: outputs/analysis/table1_vs_paper.csv")
    notes.append("This is the clearest augmentation comparison. First compare the paper block to my-results block, then use the delta block to see direction and size of mismatch. Remind the audience that positive means reproduced AUC is higher.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Main table comparison: replacement", "Average AUC when classifiers train only on synthetic data")
    main_result_table(slide, t1, "replacement", "paper_auc", 0.55, 1.35, 5.85, 1.95, title="Paper")
    main_result_table(slide, t1, "replacement", "reproduced_auc_mean", 6.85, 1.35, 5.85, 1.95, title="My results")
    delta_table(slide, t1, "replacement", 0.55, 4.25, 12.15, 1.95, title="Delta: my result minus paper")
    add_text(slide, "Replacement is the stricter comparison because the downstream classifier sees synthetic training rows only.", 0.75, 6.55, 11.6, 0.34, size=11.8, color=MUTED)
    add_footer(slide, "Source: outputs/analysis/table1_vs_paper.csv")
    notes.append("This slide mirrors the lower half of the paper table. It should be the main evidence slide for whether the reproduction matches the paper in the synthetic-only setting.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Augmentation and replacement", "The same generators answer two different questions")
    aug = t1[(t1["mode"] == "augmentation") & (t1["generator"].isin(["original", "tabpfgen", "ctgan", "smote"]))]
    repl = t1[(t1["mode"] == "replacement") & (t1["generator"].isin(["tabpfgen", "ctgan", "smote"]))]
    pivot = pd.concat([aug, repl])
    pivot = pivot.groupby(["mode", "generator"])["reproduced_auc_mean"].mean().reset_index()
    pivot["AUC"] = pivot["reproduced_auc_mean"].map(lambda x: fmt(x))
    display = pivot[["mode", "generator", "AUC"]].rename(columns={"mode": "Mode", "generator": "Generator"})
    table(slide, display, 0.7, 1.55, 5.1, 2.8, font=10.5)
    card(slide, 6.35, 1.55, 2.8, 1.35, "augmentation", "real + synthetic", BLUE, "Tests whether synthetic data improves a real training set.")
    card(slide, 9.45, 1.55, 2.8, 1.35, "replacement", "synthetic only", AMBER, "Tests whether the generator can stand in for real data.")
    add_text(slide, "Replacement is the harder and more diagnostic setting. A generator can help augmentation while still failing to capture enough structure to train a classifier by itself.", 6.35, 3.65, 5.8, 0.9, size=15, color=NAVY)
    add_footer(slide)
    notes.append("Explain the main result table from the paper. Top rows use synthetic data as augmentation. Bottom rows train classifiers only on synthetic data. Replacement is the stricter test of whether synthetic data contains useful structure.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "TabPFGen implementation status", "The sampler now uses raw logits, not probabilities")
    card(slide, 0.75, 1.45, 3.65, 1.35, "energy signal", "predict_logits", TEAL, "Raw logits preserve the prior energy scale used by SGLD.")
    card(slide, 4.85, 1.45, 3.65, 1.35, "gradient path", "differentiable input", BLUE, "SGLD needs gradients with respect to x_synth.")
    card(slide, 8.95, 1.45, 3.65, 1.35, "stability checks", "finite guards", RED, "Non-finite energy, gradient, or samples now fail loudly.")
    add_text(slide, "Why logits matter: softmax probabilities normalize away part of the evidence scale. The paper energy is based on the model score before that normalization, so raw logits better preserve the prior used during sampling.", 0.85, 3.45, 11.5, 0.85, size=15.5, color=NAVY)
    add_text(slide, "Current risk: diabetes can still fail during SGLD because its gradients become non-finite. That is now visible as a numerical-stability failure rather than silent empty CSV rows.", 0.85, 4.65, 11.5, 0.75, size=15.5, color=RED)
    add_footer(slide, "Source: src/tabpfgen/tabpfgen.py and scripts/train_table1_generators.py")
    notes.append("This slide answers what changed in the implementation. The important sentence is that logits preserve the energy scale, and differentiability means changes in x_synth must change the TabPFN output so autograd can compute SGLD updates.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Core and full TabPFGen", "The full objective asks synthetic data to explain the real data")
    card(slide, 0.75, 1.35, 3.6, 1.45, "core term", "real -> synthetic", BLUE, "Real rows form the TabPFN context. SGLD moves synthetic rows until TabPFN assigns their target labels high logits.")
    card(slide, 4.85, 1.35, 3.6, 1.45, "swap term", "synthetic -> real", TEAL, "Synthetic rows become the context. The same labels should help TabPFN classify the original real rows.")
    card(slide, 8.95, 1.35, 3.6, 1.45, "implementation", "weight = 1", PURPLE, "The current default uses the full paper objective. Weight 0 recovers the core ablation.")
    add_text(slide, "Why the swap helps", 0.85, 3.55, 3.2, 0.32, size=17, bold=True)
    add_text(slide, "The core term can produce rows that look highly class-plausible but do not form a useful training set. The swap term penalizes synthetic batches that fail to classify the original data when used as the in-context training set.", 0.85, 3.98, 11.35, 0.78, size=14.5, color=NAVY)
    add_text(slide, "Practical interpretation", 0.85, 5.05, 3.2, 0.32, size=17, bold=True)
    add_text(slide, "Full TabPFGen tests whether the synthetic set can stand in for the real set, which matters most in replacement mode.", 0.85, 5.48, 11.35, 0.45, size=14.5, color=NAVY)
    add_footer(slide, "Source: TabPFGen paper Algorithm 1 and full-method description")
    notes.append("Explain the swap slowly. First say that core asks whether a synthetic row is plausible given real data. Then say full also asks the reverse: if synthetic data becomes the context, can it classify real rows? That makes the synthetic batch useful as a replacement training set, not only plausible sample by sample.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Feature scaling decision", "Scaling is an implementation choice for SGLD stability")
    scaling = pd.DataFrame(
        [
            ["Paper detail", "External scaling is not clearly specified in TabPFGen."],
            ["Numerical issue", "SGLD step size and noise act directly in feature space."],
            ["Why scale", "Large-range features otherwise dominate gradients and noise."],
            ["How to report", "Main practical run uses scaling; no-scale is an ablation."],
        ],
        columns=["Point", "Interpretation"],
    )
    table(slide, scaling, 0.75, 1.35, 7.1, 3.1, font=11.2, header_fill=BLUE)
    add_text(slide, "Supervisor-facing justification", 8.35, 1.45, 3.7, 0.32, size=17, bold=True)
    add_text(slide, "Scaling does not change the conceptual TabPFGen objective. It makes a fixed SGLD step size and Gaussian noise level comparable across heterogeneous OpenML features.", 8.35, 1.9, 3.95, 1.05, size=13.8, color=NAVY)
    add_text(slide, "Caveat", 8.35, 3.45, 3.7, 0.32, size=17, bold=True)
    add_text(slide, "Because the paper underspecifies this preprocessing detail, strict reproduction should include a no-scale sensitivity run.", 8.35, 3.9, 3.95, 0.78, size=13.8, color=RED)
    add_footer(slide, "Run no-scale with: TABPFGEN_SCALE_FLAG=--tabpfgen-no-scale")
    notes.append("Use this slide when asked why scaling appears in the code. The honest answer is that the paper underspecifies external scaling, but gradient-based sampling is scale-sensitive. Present scaling as an engineering stabilization choice and the no-scale run as the check.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Hollmann protocol reference", "The cited TabPFN paper gives a procedure, not exact final settings")
    hollmann = pd.DataFrame(
        [
            ["What can be inferred", "Benchmark style, AUC evaluation, small tabular setting, and baseline tuning philosophy."],
            ["What cannot be inferred", "Exact per-dataset XGB, RF, LR, or TabPFN hyperparameters."],
            ["Downstream implication", "A faithful extension would add a tuning stage instead of inventing fixed values."],
            ["Current code stance", "Keep explicit defaults and document the deviation."],
        ],
        columns=["Question", "Answer"],
    )
    table(slide, hollmann, 0.65, 1.35, 12.0, 3.5, font=10.6, header_fill=NAVY)
    add_text(slide, "Bottom line: “following Hollmann” supports the evaluation protocol. It does not uniquely determine every downstream hyperparameter unless the TabPFGen authors release configs or selected values.", 0.85, 5.35, 11.4, 0.58, size=15, color=NAVY)
    add_footer(slide, "Source: Hollmann et al. TabPFN paper hyperparameter-search table")
    notes.append("This slide prevents overclaiming. Say that Hollmann provides search spaces and evaluation style, not a recoverable list of exact selected hyperparameters. Therefore the reproduction should either tune baselines or disclose fixed defaults.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "TabPFN model version", "The paper checkpoint differs from modern TabPFN releases")
    models = pd.DataFrame(
        [
            ["Paper checkpoint", "prior_diff_real_checkpoint_n_0_epoch_42.cpkt", "original TabPFN-era weights"],
            ["Current risk", "model_path='auto'", "may load newer package defaults"],
            ["v2.5 / v3 impact", "different pretrained prior", "can change logits, gradients, and AUC"],
            ["Strict reproduction", "pin checkpoint", "use the paper path when compatible"],
        ],
        columns=["Item", "Path or setting", "Meaning"],
    )
    table(slide, models, 0.65, 1.35, 12.0, 3.65, font=9.8, header_fill=PURPLE)
    add_text(slide, "Interpretation", 0.85, 5.38, 2.5, 0.3, size=17, bold=True)
    add_text(slide, "Using a v3 or other new TabPFN model is useful as a modern ablation, but it should not be described as the exact paper reproduction.", 2.8, 5.35, 9.5, 0.45, size=14.5, color=NAVY)
    add_footer(slide, "Source: TabPFGen paper checkpoint path and PriorLabs TabPFN model naming")
    notes.append("Use this slide to separate reproduction from extension. The paper path is an old TabPFN checkpoint. A v3/new-model run can be scientifically useful, but it changes the prior and therefore the sampler.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Diabetes failure mode", "The failed rows are informative rather than just missing output")
    add_text(slide, "Observed symptom", 0.75, 1.55, 3.0, 0.3, size=17, bold=True)
    add_text(slide, "Generated diabetes rows were effectively empty feature vectors with a target label, for example comma-separated blanks ending in tested_negative.", 0.75, 1.95, 5.2, 0.9, size=14, color=NAVY)
    add_text(slide, "Likely mechanism", 0.75, 3.15, 3.0, 0.3, size=17, bold=True)
    add_text(slide, "SGLD hit non-finite gradients on diabetes. Once numerical values become NaN or infinite, inverse scaling can turn the synthetic feature columns into invalid or blank output.", 0.75, 3.55, 5.2, 1.0, size=14, color=NAVY)
    card(slide, 6.65, 1.55, 2.5, 1.15, "current action", "ignore now", AMBER, "Keep the limitation explicit.")
    card(slide, 9.55, 1.55, 2.5, 1.15, "new logging", "gradient trace", BLUE, "Run health jobs without overwriting generations.")
    add_text(slide, "The new gradient-health script records energy finiteness, gradient finiteness, gradient norms, and sample magnitude per SGLD step.", 6.65, 3.35, 5.3, 0.8, size=14.5, color=NAVY)
    add_footer(slide, "Run: bash scripts/slurm/submit_tabpfgen_gradient_health.sh")
    notes.append("Be transparent about diabetes. The goal is not to hide this dataset. Say that we are excluding it temporarily, and the next diagnostic is gradient health logging to identify where sampling becomes numerically unstable.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Synthetic fidelity diagnostics", "Utility can look good while distribution matching remains weak")
    diag_cols = ["generator", "mean_feature_ks_mean", "correlation_mae_mean", "c2st_auc_mean", "exact_duplicate_fraction_mean"]
    diag = synth[diag_cols].copy()
    diag.columns = ["Generator", "KS", "Correlation MAE", "C2ST AUC", "Exact duplicate frac."]
    for c in diag.columns[1:]:
        diag[c] = diag[c].map(lambda x: fmt(x))
    table(slide, diag, 0.65, 1.35, 7.6, 4.1, font=9.2)
    add_text(slide, "Readout", 8.65, 1.35, 3.3, 0.3, size=17, bold=True)
    add_text(slide, "SMOTE is hard for C2ST to detect but has copy-like nearest-neighbor behavior. Neural generators often avoid exact copies but remain distinguishable from real data.", 8.65, 1.8, 3.85, 1.15, size=13.5, color=NAVY)
    add_text(slide, "The new extra-evaluation script extends this with conditional KS, PCA spectrum distance, partial correlations, feature-target mutual information, manifold precision/coverage, and minority coverage.", 8.65, 3.3, 3.85, 1.35, size=13.5, color=NAVY)
    add_footer(slide, "Run: sbatch scripts/slurm/run_extra_synthetic_evaluations.sh")
    notes.append("This slide explains why downstream AUC alone is not enough. Point out the contrast between fidelity, detectability, and copy risk, then mention the new diagnostics as a way to locate which part of the distribution each method loses.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Extra fidelity results", "TabPFGen preserves support well, but SMOTE still wins several low-level distribution metrics")
    if not extra.empty:
        extra_display = extra[
            [
                "generator",
                "conditional_ks_median",
                "pca_spectrum_l1_median",
                "partial_corr_mae_median",
                "feature_target_mi_mae_median",
                "precision_median",
                "coverage_median",
                "minority_coverage_median",
            ]
        ].copy()
        extra_display = extra_display.sort_values("conditional_ks_median")
        extra_display.columns = [
            "Generator",
            "Cond. KS",
            "PCA L1",
            "Partial corr.",
            "MI target",
            "Precision",
            "Coverage",
            "Minority cov.",
        ]
        for c in extra_display.columns[1:]:
            extra_display[c] = extra_display[c].map(lambda x: fmt(x))
        table(slide, extra_display, 0.55, 1.28, 7.8, 4.45, font=8.2)
        add_text(slide, "What to notice", 8.75, 1.35, 3.4, 0.3, size=17, bold=True)
        add_text(slide, "TabPFGen has the best median precision and coverage, and the best PCA spectrum match. That means its samples sit close to the real manifold and span much of it.", 8.75, 1.78, 3.8, 1.2, size=13.2, color=NAVY)
        add_text(slide, "SMOTE has the lowest conditional KS, partial correlation error, and target-MI error. That is expected because it interpolates real rows instead of learning a new distribution.", 8.75, 3.25, 3.8, 1.15, size=13.2, color=NAVY)
        add_text(slide, "Caution: partial-correlation means have huge outliers, so this slide uses medians.", 8.75, 4.75, 3.8, 0.6, size=12.4, color=RED)
    else:
        add_text(slide, "No extra diagnostics summary found.", 0.85, 1.6, 9.0, 0.5, size=16, color=RED)
    add_footer(slide, "Source: outputs/analysis_extra/extra_synthetic_diagnostics_summary.csv")
    notes.append("This is the main new fidelity slide. The key point is nuanced: TabPFGen looks good on manifold support and PCA spectrum, while SMOTE looks best on local distribution/dependency metrics because it interpolates real rows. Use medians because partial-correlation means are dominated by numerical outliers.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "How to read support and copying", "High support can coexist with membership risk")
    support = pd.DataFrame(
        [
            ["Precision", "Are synthetic rows close to real-data regions?"],
            ["Coverage", "Do synthetic rows cover many real-data regions?"],
            ["Minority coverage", "Do synthetic rows cover the minority-class region?"],
            ["Exact duplicate fraction", "How often a synthetic row exactly matches a train row after preprocessing/inverse output."],
        ],
        columns=["Metric", "Question it answers"],
    )
    table(slide, support, 0.7, 1.35, 6.1, 3.25, font=10.0, header_fill=BLUE)
    add_text(slide, "Manifold support", 7.35, 1.35, 3.4, 0.3, size=17, bold=True)
    add_text(slide, "The manifold is the region where realistic rows live. Good support means synthetic rows are near real rows and cover the same regions, instead of drifting into implausible feature combinations.", 7.35, 1.78, 4.9, 1.1, size=13.3, color=NAVY)
    add_text(slide, "Why TabPFGen has zero exact copies", 7.35, 3.25, 4.1, 0.3, size=17, bold=True)
    add_text(slide, "SGLD starts near training rows but adds continuous noise and gradient updates, so exact full-row matches are unlikely. Other generators can recreate exact rows after categorical decoding, rounding, clipping, or interpolation on small datasets.", 7.35, 3.68, 4.9, 1.25, size=13.3, color=NAVY)
    add_text(slide, "Key caveat: zero exact duplicates does not imply privacy. Membership attacks can still detect that TabPFGen rows sit closer to train members than to non-members.", 0.85, 5.55, 11.3, 0.55, size=14.5, color=RED)
    add_footer(slide, "Interpretation slide for outputs/analysis_extra and outputs/privacy_attacks")
    notes.append("Use this as the explainer slide. Manifold support means the generated rows lie in realistic regions and cover those regions. Exact duplicate fraction is stricter: it only counts exact row matches. TabPFGen has zero exact duplicates because SGLD perturbs continuous values, but membership attacks still show train rows are more exposed.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "CTGAN repeated seeds", "Best runs can look paper-like while the average remains unstable")
    ct = ctgan.copy()
    ct["paper_gap_mean"] = ct["mean_delta_vs_paper"]
    display = ct[["mode", "downstream_model", "mean_auc", "std_auc", "best_auc", "paper_auc"]].copy()
    display.columns = ["Mode", "Model", "Mean AUC", "Std.", "Best AUC", "Paper"]
    for c in ["Mean AUC", "Std.", "Best AUC", "Paper"]:
        display[c] = display[c].map(lambda x: fmt(x))
    table(slide, display, 0.55, 1.35, 7.55, 4.55, font=8.6)
    repl_ct = ct[ct["mode"] == "replacement"]
    vals = repl_ct.set_index("downstream_model")["mean_auc"].reindex(["xgb", "rf", "lr", "tabpfn"]).dropna()
    bar_chart(slide, vals.index.tolist(), vals.values.tolist(), 8.45, 1.72, 4.1, 2.15, max_value=1.0, color=RED, title="Replacement mean AUC")
    add_text(slide, "Why the best AUC can be very high: the best run often comes from easy datasets and seed selection. It is not a stable estimate of CTGAN performance.", 8.45, 4.35, 4.05, 0.85, size=13.5, color=NAVY)
    add_footer(slide, "Source: outputs/analysis/ctgan_repeats_summary.csv")
    notes.append("Use this to answer the question about surprisingly high CTGAN best AUC. Best-of-many is selection biased. The average and standard deviation are the honest stability signal.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Privacy attack results", "TabPFGen and SMOTE show the clearest membership signal")
    if not privacy.empty:
        privacy_display = privacy[
            [
                "generator",
                "dcr_membership_auc_median",
                "rf_membership_auc_median",
                "rf_membership_accuracy_median",
                "member_exact_fraction_median",
                "attribute_inference_accuracy_mean_median",
                "attribute_inference_mae_mean_median",
            ]
        ].copy()
        privacy_display = privacy_display.sort_values("rf_membership_auc_median", ascending=False)
        privacy_display.columns = [
            "Generator",
            "DCR AUC",
            "RF AUC",
            "RF acc.",
            "Exact frac.",
            "Attr. acc.",
            "Attr. MAE",
        ]
        for c in privacy_display.columns[1:]:
            privacy_display[c] = privacy_display[c].map(lambda x: fmt(x))
        table(slide, privacy_display, 0.55, 1.28, 7.75, 4.25, font=8.4, header_fill=RED)
        add_text(slide, "What to notice", 8.75, 1.35, 3.4, 0.3, size=17, bold=True)
        add_text(slide, "TabPFGen has the highest median membership AUC. Even without exact duplicate rows, the nearest-distance and RF attacks can distinguish train members from non-members.", 8.75, 1.78, 3.85, 1.2, size=13.2, color=NAVY)
        add_text(slide, "SMOTE also has high membership risk and nonzero exact-copy behavior on average. This matches the interpolation mechanism.", 8.75, 3.2, 3.85, 0.85, size=13.2, color=NAVY)
        add_text(slide, "Most neural baselines sit near 0.5 membership AUC, but that may reflect low fidelity rather than strong privacy.", 8.75, 4.35, 3.85, 0.78, size=13.2, color=MUTED)
    else:
        add_text(slide, "No privacy attack summary found.", 0.85, 1.6, 9.0, 0.5, size=16, color=RED)
    add_footer(slide, "Source: outputs/privacy_attacks/privacy_attacks_summary.csv")
    notes.append("This is the main privacy result. Say that TabPFGen has the strongest membership signal, even though exact duplicate fraction is zero. SMOTE is also risky, partly because interpolation keeps samples close to real training rows. A near-random privacy attack for weak generators should not automatically be celebrated because their fidelity is poor.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Evaluation map", "Which insights need only evaluation and which require more training")
    eval_only = pd.DataFrame(
        [
            ["Conditional fidelity", "run extra diagnostics"],
            ["Dependency structure", "partial corr, MI, PCA"],
            ["Manifold support", "precision and coverage"],
            ["Privacy risk", "train attack evaluators"],
            ["Paper comparison", "refresh summaries"],
        ],
        columns=["Evaluation-only", "Readout"],
    )
    train_needed = pd.DataFrame(
        [
            ["CTGAN seeds", "generator retraining"],
            ["TabPFGen hyperparameters", "new generations"],
            ["Gradient clipping", "new TabPFGen runs"],
            ["Label priors", "new synthetic labels"],
            ["Raw logits vs probabilities", "ablation generation"],
        ],
        columns=["Requires generation/training", "Why"],
    )
    table(slide, eval_only, 0.7, 1.45, 5.7, 3.15, font=10.2, header_fill=BLUE)
    table(slide, train_needed, 6.9, 1.45, 5.7, 3.15, font=10.2, header_fill=RED)
    add_text(slide, "This keeps the reproduction efficient: run evaluation-only diagnostics first, then spend GPU time only where the diagnostics point to a specific failure mode.", 0.9, 5.25, 11.2, 0.55, size=15, color=NAVY)
    add_footer(slide)
    notes.append("This slide turns the reproduction into a work plan. Evaluation-only items can run on existing outputs. Training items consume GPU time or rerun generators, so they should be prioritized after diagnostics.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Slurm commands", "The current runnable workflow")
    commands = pd.DataFrame(
        [
            ["Main analysis refresh", "bash scripts/slurm/submit_table1_analysis.sh"],
            ["Missing 13 cells", "bash scripts/slurm/submit_table1_missing_rows.sh"],
            ["TabPFGen only", "bash scripts/slurm/submit_table1_tabpfgen_pipeline.sh"],
            ["CTGAN repeated seeds", "bash scripts/slurm/submit_ctgan_repeats_pipeline.sh"],
            ["Extra diagnostics", "sbatch scripts/slurm/run_extra_synthetic_evaluations.sh"],
            ["Gradient health", "bash scripts/slurm/submit_tabpfgen_gradient_health.sh"],
            ["Privacy attacks", "sbatch scripts/slurm/run_privacy_attacks.sh"],
        ],
        columns=["Task", "Command"],
    )
    table(slide, commands, 0.65, 1.3, 12.0, 4.7, font=9.4, header_fill=NAVY)
    add_text(slide, "Gradient-health outputs are separate from table1_generators and skip existing traces unless OVERWRITE_FLAG=--overwrite is set.", 0.8, 6.2, 11.5, 0.4, size=13, color=MUTED)
    add_footer(slide, "Source: scripts/slurm/")
    notes.append("Use this as the operational slide. It gives the actual commands to run and reminds the audience that gradient health does not overwrite earlier generations.")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Interpretation to present", "What the reproduction currently supports")
    items = [
        ("1", "TabPFGen broadly matches the paper ranking, but exact numbers still depend on missing reruns."),
        ("2", "Replacement mode is the stronger test and exposes failures that augmentation can hide."),
        ("3", "CTGAN best runs are not enough; mean and variance over seeds show instability."),
        ("4", "Utility, fidelity, dependency preservation, and privacy are separate axes."),
        ("5", "Diabetes is the active TabPFGen stability case to debug next."),
    ]
    for i, (num, text) in enumerate(items):
        y = 1.45 + i * 0.78
        add_note_marker(slide, num, 0.85, y, BLUE if num != "5" else RED)
        add_text(slide, text, 1.35, y - 0.02, 10.8, 0.44, size=16, color=NAVY)
    add_text(slide, "The useful research direction is not only improving average AUC. The stronger direction is explaining when TabPFGen preserves decision-relevant structure without copying real rows or becoming numerically unstable.", 0.85, 5.85, 11.5, 0.65, size=15, color=MUTED)
    add_footer(slide)
    notes.append("Close with the research message. The reproduction is now useful because it identifies concrete axes for further work: stability, structure preservation, and privacy, not just aggregate AUC.")

    return prs, notes


def add_notes_to_pptx(path: Path, notes: list[str]) -> None:
    tmp = path.with_suffix(".notes_tmp.pptx")
    with zipfile.ZipFile(path, "r") as zin:
        parts = {item.filename: zin.read(item.filename) for item in zin.infolist()}

    content = parts["[Content_Types].xml"].decode("utf-8")
    if "notesMasters" not in content:
        content = content.replace(
            "</Types>",
            '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/></Types>',
        )
    for idx in range(1, len(notes) + 1):
        marker = f'/ppt/notesSlides/notesSlide{idx}.xml'
        if marker not in content:
            content = content.replace(
                "</Types>",
                f'<Override PartName="{marker}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>',
            )
    parts["[Content_Types].xml"] = content.encode("utf-8")

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(parts)

        if "ppt/notesMasters/notesMaster1.xml" not in names:
            parts["ppt/notesMasters/notesMaster1.xml"] = (
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notesMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:bg><p:bgPr><a:solidFill><a:schemeClr val="bg1"/></a:solidFill></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:notesStyle/></p:notesMaster>"""
            ).encode("utf-8")
            parts["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = (
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
            ).encode("utf-8")

        for idx, note in enumerate(notes, start=1):
            slide_rels = f"ppt/slides/_rels/slide{idx}.xml.rels"
            rel_id = f"rIdNotes{idx}"
            if slide_rels in names:
                rels = parts[slide_rels].decode("utf-8")
            else:
                rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
            if f"notesSlide{idx}.xml" not in rels:
                rels = rels.replace(
                    "</Relationships>",
                    f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{idx}.xml"/></Relationships>',
                )
                parts[slide_rels] = rels.encode("utf-8")

            note_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="685800" y="3886200"/><a:ext cx="7772400" cy="2057400"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1200"/><a:t>{escape(note)}</a:t></a:r><a:endParaRPr lang="en-US" sz="1200"/></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>"""
            parts[f"ppt/notesSlides/notesSlide{idx}.xml"] = note_xml.encode("utf-8")
            parts[f"ppt/notesSlides/_rels/notesSlide{idx}.xml.rels"] = (
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{idx}.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/></Relationships>"""
            ).encode("utf-8")
        for name, data in parts.items():
            zout.writestr(name, data)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prs, notes = make_deck(load_data())
    prs.save(args.output)
    add_notes_to_pptx(args.output, notes)
    print(args.output)


if __name__ == "__main__":
    main()
