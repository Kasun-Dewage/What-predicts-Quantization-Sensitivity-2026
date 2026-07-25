import json
import os
import math
import argparse
import glob
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


COMP_COLORS = {"Q": "#d73027", "K": "#4575b4", "V": "#1a9641", "O": "#762a83"}
COMP_MARKERS = {"Q": "o", "K": "s", "V": "^", "O": "D"}
COMP_ORDER = ["Q", "K", "V", "O"]

PROJ_GROUP_COLORS = {
    "QO_square": "#d73027",
    "KV_square": "#4575b4",
    "KV_rect":   "#ff7f00",
}
PROJ_GROUP_MARKERS = {
    "QO_square": "o",
    "KV_square": "s",
    "KV_rect":   "^",
}


def load_results(paths):
    all_data = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "model" in d and "layers" in d:
            all_data[d["model"]] = d
        else:
            for mk, v in d.items():
                if "layers" in v:
                    all_data[mk] = v
    return all_data


def proj_group_label(comp):
    if comp in ("Q", "O"):
        return "QO"
    return "KV"


def infer_proj_group_key(comp, shape):
    if comp in ("Q", "O"):
        return "QO_square"
    if shape and len(shape) == 2:
        m, n = shape[0], shape[1]
        gamma = max(m, n) / min(m, n) if min(m, n) > 0 else 1.0
        if gamma > 1.5:
            return "KV_rect"
    return "KV_square"


def flatten_records(model_data, bits):
    records = []
    baseline_ppl = model_data.get("baseline_ppl", float("nan"))
    for layer_key, comps in model_data["layers"].items():
        layer_idx = int(layer_key) if layer_key.isdigit() else -1
        for comp, info in comps.items():
            delta_key = "ppl_delta_{}bit".format(bits)
            recon_key = "recon_error_{}bit".format(bits)
            hsens_key = "hessian_sens_{}bit".format(bits)
            delta = info.get(delta_key, float("nan"))
            recon = info.get(recon_key, float("nan"))
            hsens = info.get(hsens_key, float("nan"))
            if not math.isfinite(delta):
                continue
            shape = info.get("shape", None)
            gamma_val = info.get("gamma", float("nan"))
            if (not math.isfinite(gamma_val)) and shape and len(shape) == 2:
                mn = min(shape[0], shape[1])
                mx = max(shape[0], shape[1])
                gamma_val = mx / mn if mn > 0 else float("nan")
            pg_key = infer_proj_group_key(comp, shape)
            records.append({
                "model": model_data["model"],
                "layer": layer_idx,
                "comp": comp,
                "shape": shape,
                "gamma": gamma_val,
                "proj_group": proj_group_label(comp),
                "proj_group_key": pg_key,
                "mp_energy_ratio": info.get("mp_energy_ratio", float("nan")),
                "entry_outlier_density": info.get("entry_outlier_density", float("nan")),
                "mp_n_outliers": info.get("mp_n_outliers", float("nan")),
                "kurtosis": info.get("kurtosis", float("nan")),
                "recon_error": recon,
                "hessian_sens": hsens,
                "ppl_delta": delta,
                "baseline_ppl": baseline_ppl,
                "ppl_delta_norm": delta / baseline_ppl if baseline_ppl > 0 else float("nan"),
            })
    return records


def corr_stats(x_arr, y_arr):
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho, p_s = sp_stats.spearmanr(x_arr[valid], y_arr[valid])
    r_lin, p_l = sp_stats.pearsonr(x_arr[valid], y_arr[valid])
    return float(rho), float(p_s), float(r_lin), float(p_l)


def bootstrap_r2_ci(x, y, n_boot=2000, seed=0):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    xv, yv = x[valid], y[valid]
    rng = np.random.default_rng(seed)
    n = len(xv)
    r2_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = xv[idx], yv[idx]
        if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            continue
        r, _ = sp_stats.pearsonr(xs, ys)
        r2_samples.append(r * r)
    if not r2_samples:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(r2_samples)
    r_full, _ = sp_stats.pearsonr(xv, yv)
    r2_point = float(r_full * r_full)
    return r2_point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def eta_squared_oneway(groups):
    flat = []
    labels = []
    for g, vals in enumerate(groups):
        for v in vals:
            flat.append(v)
            labels.append(g)
    flat = np.array(flat)
    labels = np.array(labels)
    if len(flat) < 2:
        return float("nan")
    grand = flat.mean()
    ss_between = 0.0
    for g in range(len(groups)):
        vals = flat[labels == g]
        if len(vals) == 0:
            continue
        ss_between += len(vals) * (vals.mean() - grand) ** 2
    ss_total = ((flat - grand) ** 2).sum()
    if ss_total < 1e-12:
        return float("nan")
    return ss_between / ss_total


def add_regression_band(ax, x, y, color="black", lw=1.2):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return
    xv, yv = x[valid], y[valid]
    slope, intercept, _, _, se = sp_stats.linregress(xv, yv)
    x_line = np.linspace(xv.min(), xv.max(), 300)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, lw=lw, ls="--", zorder=4)


def figure_per_model_scatter(all_data, bits, output_dir):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Outlier Density\n(|w| > 4 std)"),
        ("recon_error", "RTN Recon Error\n(relative Frobenius)"),
    ]
    for model_key, model_data in all_data.items():
        records = flatten_records(model_data, bits)
        if not records:
            continue
        n_cols = len(x_metrics)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5))
        for ci, (mkey, mlabel) in enumerate(x_metrics):
            ax = axes[ci]
            all_x, all_y = [], []
            for comp in COMP_ORDER:
                recs = [r for r in records if r["comp"] == comp]
                if not recs:
                    continue
                xs = np.array([r[mkey] for r in recs])
                ys = np.array([r["ppl_delta"] for r in recs])
                valid = np.isfinite(xs) & np.isfinite(ys)
                ax.scatter(xs[valid], ys[valid],
                           c=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                           s=18, alpha=0.8, label=comp, zorder=3)
                all_x.extend(xs[valid].tolist())
                all_y.extend(ys[valid].tolist())
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            rho, p_s, r_lin, _ = corr_stats(all_x, all_y)
            add_regression_band(ax, all_x, all_y)
            ax.set_xlabel(mlabel, fontsize=9)
            ax.set_ylabel("PPL Delta", fontsize=9)
            if math.isfinite(rho):
                ax.set_title(
                    "Spearman r={:.3f} (p={:.1e})\nPearson r={:.3f}".format(rho, p_s, r_lin),
                    fontsize=9,
                )
            ax.tick_params(labelsize=8)
            if ci == 0:
                ax.legend(title="Proj", fontsize=8, markerscale=1.4)
        baseline_ppl = model_data.get("baseline_ppl", float("nan"))
        fig.suptitle(
            "{} -- {}bit RTN sensitivity vs MP metrics  (baseline PPL={:.2f})".format(
                model_key, bits, baseline_ppl),
            fontsize=11,
        )
        fig.tight_layout()
        out = os.path.join(output_dir, "{}_scatter_{}bit.pdf".format(model_key, bits))
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("Saved: {}".format(out))


def figure_proj_split_scatter(all_data, bits, output_dir):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Outlier Density\n(|w| > 4 std)"),
        ("recon_error", "RTN Recon Error\n(relative Frobenius)"),
    ]
    group_labels = {
        "QO_square": "Q/O (square, MHA+GQA)",
        "KV_square": "K/V (square, MHA)",
        "KV_rect":   "K/V (rect, GQA)",
    }
    for model_key, model_data in all_data.items():
        records = flatten_records(model_data, bits)
        if not records:
            continue
        groups_present = sorted(set(r["proj_group_key"] for r in records))
        n_groups = len(groups_present)
        n_cols = len(x_metrics)
        fig, axes = plt.subplots(n_groups, n_cols,
                                 figsize=(5.5 * n_cols, 4.2 * n_groups),
                                 squeeze=False)
        for gi, gkey in enumerate(groups_present):
            grecs = [r for r in records if r["proj_group_key"] == gkey]
            for ci, (mkey, mlabel) in enumerate(x_metrics):
                ax = axes[gi][ci]
                all_x, all_y = [], []
                for comp in COMP_ORDER:
                    recs = [r for r in grecs if r["comp"] == comp]
                    if not recs:
                        continue
                    xs = np.array([r[mkey] for r in recs])
                    ys = np.array([r["ppl_delta"] for r in recs])
                    valid = np.isfinite(xs) & np.isfinite(ys)
                    if valid.sum() == 0:
                        continue
                    ax.scatter(xs[valid], ys[valid],
                               c=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                               s=18, alpha=0.8, label=comp, zorder=3)
                    all_x.extend(xs[valid].tolist())
                    all_y.extend(ys[valid].tolist())
                all_x = np.array(all_x)
                all_y = np.array(all_y)
                rho, p_s, r_lin, _ = corr_stats(all_x, all_y)
                add_regression_band(ax, all_x, all_y,
                                    color=PROJ_GROUP_COLORS.get(gkey, "black"))
                ax.set_xlabel(mlabel, fontsize=9)
                ax.set_ylabel("PPL Delta", fontsize=9)
                n_valid = int((np.isfinite(all_x) & np.isfinite(all_y)).sum())
                if math.isfinite(rho):
                    ax.set_title(
                        "[{}] {}\nSpearman r={:.3f} (p={:.1e}) n={}".format(
                            group_labels.get(gkey, gkey), mlabel, rho, p_s, n_valid),
                        fontsize=8,
                    )
                else:
                    ax.set_title("[{}] {} n={}".format(
                        group_labels.get(gkey, gkey), mlabel, n_valid), fontsize=8)
                ax.tick_params(labelsize=8)
                if ci == 0:
                    ax.legend(title="Proj", fontsize=7, markerscale=1.4)
        baseline_ppl = model_data.get("baseline_ppl", float("nan"))
        fig.suptitle(
            "{} -- {}bit RTN  projection-type split  (baseline PPL={:.2f})".format(
                model_key, bits, baseline_ppl),
            fontsize=11,
        )
        fig.tight_layout()
        out = os.path.join(output_dir, "{}_proj_split_scatter_{}bit.pdf".format(model_key, bits))
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("Saved: {}".format(out))


def figure_combined_scatter(all_data, bits, output_dir):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Outlier Density\n(|w| > 4 std)"),
        ("recon_error", "RTN Recon Error (relative Frobenius)"),
    ]
    MODEL_STYLES = {}
    cmap = plt.get_cmap("tab10")
    for i, mk in enumerate(all_data.keys()):
        MODEL_STYLES[mk] = {"color": cmap(i % 10), "alpha": 0.55}
    n_cols = len(x_metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5))
    all_records = []
    for model_key, model_data in all_data.items():
        all_records.extend(flatten_records(model_data, bits))
    for ci, (mkey, mlabel) in enumerate(x_metrics):
        ax = axes[ci]
        for mk in all_data:
            recs = [r for r in all_records if r["model"] == mk]
            xs = np.array([r[mkey] for r in recs])
            ys = np.array([r["ppl_delta"] for r in recs])
            valid = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[valid], ys[valid], s=10, alpha=MODEL_STYLES[mk]["alpha"],
                       c=[MODEL_STYLES[mk]["color"]] * valid.sum(), label=mk, zorder=3)
        xs_all = np.array([r[mkey] for r in all_records])
        ys_all = np.array([r["ppl_delta"] for r in all_records])
        add_regression_band(ax, xs_all, ys_all, color="black", lw=1.5)
        rho, p_s, r_lin, _ = corr_stats(xs_all, ys_all)
        ax.set_xlabel(mlabel, fontsize=9)
        ax.set_ylabel("PPL Delta", fontsize=9)
        if math.isfinite(rho):
            ax.set_title(
                "Spearman r={:.3f} (p={:.1e})\nPearson r={:.3f}  n={}".format(
                    rho, p_s, r_lin,
                    int((np.isfinite(xs_all) & np.isfinite(ys_all)).sum())),
                fontsize=9,
            )
        ax.tick_params(labelsize=8)
        if ci == 0:
            ax.legend(fontsize=6, markerscale=1.4, ncol=2)
    fig.suptitle("All models -- {}bit RTN sensitivity vs MP metrics".format(bits), fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "combined_scatter_{}bit.pdf".format(bits))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(out))


def figure_variance_decomposition(all_data, bits, output_dir):
    model_keys = list(all_data.keys())
    metrics = ["recon_error", "hessian_sens"]
    recon_r2 = []
    hsens_r2 = []
    comp_eta2 = []
    layer_r2 = []
    have_hsens = []
    for mk in model_keys:
        recs = flatten_records(all_data[mk], bits)
        if not recs:
            recon_r2.append(float("nan"))
            hsens_r2.append(float("nan"))
            comp_eta2.append(float("nan"))
            layer_r2.append(float("nan"))
            have_hsens.append(False)
            continue
        ys = np.array([r["ppl_delta"] for r in recs])
        xs_re = np.array([r["recon_error"] for r in recs])
        xs_hs = np.array([r["hessian_sens"] for r in recs])
        xs_ly = np.array([r["layer"] for r in recs], dtype=float)
        _, _, r_re, _ = corr_stats(xs_re, ys)
        _, _, r_ly, _ = corr_stats(xs_ly, ys)
        recon_r2.append(r_re * r_re if math.isfinite(r_re) else float("nan"))
        layer_r2.append(r_ly * r_ly if math.isfinite(r_ly) else float("nan"))
        if np.isfinite(xs_hs).any():
            _, _, r_hs, _ = corr_stats(xs_hs, ys)
            hsens_r2.append(r_hs * r_hs if math.isfinite(r_hs) else float("nan"))
            have_hsens.append(True)
        else:
            hsens_r2.append(float("nan"))
            have_hsens.append(False)
        groups = [[r["ppl_delta"] for r in recs if r["comp"] == c and math.isfinite(r["ppl_delta"])]
                  for c in COMP_ORDER]
        comp_eta2.append(eta_squared_oneway(groups))

    n = len(model_keys)
    x = np.arange(n)
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * n), 5))
    ax.bar(x - 1.5 * width, comp_eta2, width, label="Component type (eta^2)", color="#1b9e77")
    ax.bar(x - 0.5 * width, layer_r2, width, label="Layer position (R^2)", color="#7570b3")
    ax.bar(x + 0.5 * width, recon_r2, width, label="Recon error (R^2)", color="#d95f02")
    if any(have_hsens):
        ax.bar(x + 1.5 * width, hsens_r2, width, label="Hessian-trace sens (R^2)", color="#e7298a")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Variance explained")
    ax.set_title("Variance decomposition of per-projection PPL delta  ({}bit RTN)".format(bits))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, "variance_decomposition_{}bit.pdf".format(bits))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(out))


def figure_v_proj_natural_experiment(all_data, bits, output_dir):
    target_models = [m for m in all_data if any(
        r["comp"] == "V" for r in flatten_records(all_data[m], bits)
    )]
    target_models = target_models[:4] if target_models else []
    if not target_models:
        return
    fig, axes = plt.subplots(1, len(target_models), figsize=(4.5 * len(target_models), 4.0),
                             squeeze=False)
    for mi, mk in enumerate(target_models):
        recs = [r for r in flatten_records(all_data[mk], bits) if r["comp"] == "V"]
        recs.sort(key=lambda r: r["layer"])
        xs = np.array([r["layer"] for r in recs])
        ys_delta = np.array([r["ppl_delta"] for r in recs])
        ys_recon = np.array([r["recon_error"] for r in recs])
        ax = axes[0][mi]
        ax.bar(xs, ys_delta, color="#1a9641", alpha=0.7, label="PPL Delta")
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("PPL Delta", fontsize=9, color="#1a9641")
        ax.tick_params(labelsize=8)
        if np.isfinite(ys_recon).any():
            m_r = float(np.nanmean(ys_recon))
            s_r = float(np.nanstd(ys_recon))
            cv_pct = 100.0 * (s_r / m_r) if m_r > 1e-12 else float("nan")
            ax2 = ax.twinx()
            ax2.plot(xs, ys_recon, color="black", lw=1.6, marker="o", ms=3, label="Recon")
            ax2.set_ylabel("Recon Error", fontsize=9)
            ax2.tick_params(labelsize=8)
            ax.set_title("{}  V-proj  CV(recon)={:.1f}%".format(mk, cv_pct), fontsize=9)
        else:
            ax.set_title("{}  V-proj".format(mk), fontsize=9)
    fig.suptitle("V-projection natural experiment  ({}bit RTN): "
                 "similar recon error, divergent PPL delta".format(bits), fontsize=10)
    fig.tight_layout()
    out = os.path.join(output_dir, "v_proj_natural_experiment_{}bit.pdf".format(bits))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(out))


def figure_component_dominance(all_data, bits, output_dir):
    model_keys = list(all_data.keys())
    shares = {c: [] for c in COMP_ORDER}
    errs = {c: [] for c in COMP_ORDER}
    for mk in model_keys:
        recs = flatten_records(all_data[mk], bits)
        totals = {c: 0.0 for c in COMP_ORDER}
        for r in recs:
            if r["comp"] in totals and math.isfinite(r["ppl_delta"]) and r["ppl_delta"] > 0:
                totals[r["comp"]] += r["ppl_delta"]
        S = sum(totals.values())
        if S <= 0:
            for c in COMP_ORDER:
                shares[c].append(0.0)
                errs[c].append((0.0, 0.0))
            continue
        for c in COMP_ORDER:
            shares[c].append(100.0 * totals[c] / S)
        per_layer = defaultdict(lambda: {c: 0.0 for c in COMP_ORDER})
        for r in recs:
            if r["comp"] in COMP_ORDER and math.isfinite(r["ppl_delta"]) and r["ppl_delta"] > 0:
                per_layer[r["layer"]][r["comp"]] += r["ppl_delta"]
        layer_rows = list(per_layer.values())
        if len(layer_rows) >= 4:
            rng = np.random.default_rng(0)
            boots = {c: [] for c in COMP_ORDER}
            for _ in range(1000):
                idx = rng.integers(0, len(layer_rows), size=len(layer_rows))
                t = {c: 0.0 for c in COMP_ORDER}
                for i in idx:
                    for c in COMP_ORDER:
                        t[c] += layer_rows[i][c]
                Sb = sum(t.values())
                if Sb <= 0:
                    continue
                for c in COMP_ORDER:
                    boots[c].append(100.0 * t[c] / Sb)
            for c in COMP_ORDER:
                if boots[c]:
                    arr = np.array(boots[c])
                    errs[c].append((float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))))
                else:
                    errs[c].append((shares[c][-1], shares[c][-1]))
        else:
            for c in COMP_ORDER:
                errs[c].append((shares[c][-1], shares[c][-1]))
    n = len(model_keys)
    x = np.arange(n)
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * n), 5))
    for i, c in enumerate(COMP_ORDER):
        vals = shares[c]
        lo = [vals[j] - errs[c][j][0] for j in range(n)]
        hi = [errs[c][j][1] - vals[j] for j in range(n)]
        ax.bar(x + (i - 1.5) * width, vals, width, label=c,
               color=COMP_COLORS[c], alpha=0.85,
               yerr=[lo, hi], capsize=2.5, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Share of total positive PPL Delta (%)")
    ax.set_title("Component dominance with bootstrap 95% CI  ({}bit RTN)".format(bits))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, "component_dominance_ci_{}bit.pdf".format(bits))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(out))


def figure_layer_depth_profile(all_data, bits, output_dir):
    for model_key, model_data in all_data.items():
        records = flatten_records(model_data, bits)
        if not records:
            continue
        comps = COMP_ORDER
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for comp in comps:
            r_comp = [r for r in records if r["comp"] == comp]
            r_comp.sort(key=lambda z: z["layer"])
            if not r_comp:
                continue
            xs = [r["layer"] for r in r_comp]
            axes[0].plot(xs, [r["mp_energy_ratio"] for r in r_comp],
                         color=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                         ms=4, lw=1.2, label=comp)
            axes[1].plot(xs, [r["ppl_delta"] for r in r_comp],
                         color=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                         ms=4, lw=1.2, label=comp)
        axes[0].set_ylabel("MP Energy Ratio", fontsize=9)
        axes[0].legend(fontsize=8, ncol=4)
        axes[0].tick_params(labelsize=8)
        axes[1].set_ylabel("PPL Delta ({}bit)".format(bits), fontsize=9)
        axes[1].set_xlabel("Layer Index", fontsize=9)
        axes[1].tick_params(labelsize=8)
        fig.suptitle("{}: Layer-depth profile of MP metrics vs quantization sensitivity".format(model_key), fontsize=11)
        fig.tight_layout()
        out = os.path.join(output_dir, "{}_depth_profile_{}bit.pdf".format(model_key, bits))
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("Saved: {}".format(out))


def figure_comp_boxplot(all_data, bits, output_dir):
    for model_key, model_data in all_data.items():
        records = flatten_records(model_data, bits)
        if not records:
            continue
        comps = [c for c in COMP_ORDER if any(r["comp"] == c for r in records)]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ci, (ykey, ylabel) in enumerate([
            ("mp_energy_ratio", "MP Energy Ratio"),
            ("ppl_delta", "PPL Delta ({}bit)".format(bits)),
        ]):
            ax = axes[ci]
            data_boxes = []
            for comp in comps:
                vals = [r[ykey] for r in records if r["comp"] == comp and math.isfinite(r[ykey])]
                data_boxes.append(vals)
            bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                            medianprops={"color": "black", "lw": 1.5})
            for patch, comp in zip(bp["boxes"], comps):
                patch.set_facecolor(COMP_COLORS[comp])
                patch.set_alpha(0.75)
            ax.set_xticklabels(comps, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle("{}: Distribution of MP metrics and PPL sensitivity by projection type".format(model_key), fontsize=10)
        fig.tight_layout()
        out = os.path.join(output_dir, "{}_boxplot_{}bit.pdf".format(model_key, bits))
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("Saved: {}".format(out))


def print_within_component_r2_table(all_data, bits_list, output_dir, n_boot=2000):
    for bits in bits_list:
        sep = "=" * 100
        print("\n" + sep)
        print("WITHIN-COMPONENT R^2  recon_error vs PPL Delta   (bits={})  with bootstrap 95% CI".format(bits))
        print(sep)
        header = "  {:<22} {:>16} {:>16} {:>16} {:>16}".format(
            "Model", "Q", "K", "V", "O")
        print(header)
        print("-" * 100)
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Q} & \textbf{K} & \textbf{V} & \textbf{O} \\")
        lines.append(r"\midrule")
        all_vals = {c: [] for c in COMP_ORDER}
        for model_key in all_data:
            recs = flatten_records(all_data[model_key], bits)
            row_cells = []
            row = "  {:<22}".format(model_key)
            for comp in COMP_ORDER:
                grecs = [r for r in recs if r["comp"] == comp]
                xs = np.array([r["recon_error"] for r in grecs])
                ys = np.array([r["ppl_delta"] for r in grecs])
                r2, lo, hi = bootstrap_r2_ci(xs, ys, n_boot=n_boot)
                if math.isfinite(r2):
                    row += " {:>7.3f}[{:.2f},{:.2f}]".format(r2, lo, hi)
                    row_cells.append("{:.3f} [{:.2f}, {:.2f}]".format(r2, lo, hi))
                    all_vals[comp].append(r2)
                else:
                    row += "      --         "
                    row_cells.append("--")
            tex_row = model_key.replace("_", r"\_")
            for c, cell in zip(COMP_ORDER, row_cells):
                tex_row += " & " + cell
            lines.append(tex_row + r" \\")
            print(row)
        lines.append(r"\midrule")
        median_row = r"\textit{Median}"
        for c in COMP_ORDER:
            vals = all_vals[c]
            if vals:
                median_row += " & {:.3f}".format(float(np.median(vals)))
            else:
                median_row += " & --"
        lines.append(median_row + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Within-component $R^2$ between reconstruction error and "
                     r"PPL delta ({}bit RTN) with bootstrap 95\% CI.}}".format(bits))
        lines.append(r"\label{{tab:within_r2_{}bit}}".format(bits))
        lines.append(r"\end{table}")
        tex = "\n".join(lines)
        out = os.path.join(output_dir, "within_component_r2_{}bit.tex".format(bits))
        with open(out, "w") as f:
            f.write(tex)
        print("Saved: {}".format(out))
        print(sep)


def print_recon_cv_table(all_data, bits_list, output_dir):
    for bits in bits_list:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Q} & \textbf{K} & \textbf{V} & \textbf{O} \\")
        lines.append(r"\midrule")
        sep = "=" * 80
        print("\n" + sep)
        print("CV(recon_error) per component (%)  bits={}".format(bits))
        print(sep)
        print("  {:<22} {:>10} {:>10} {:>10} {:>10}".format("Model", "Q", "K", "V", "O"))
        print("-" * 80)
        for model_key in all_data:
            recs = flatten_records(all_data[model_key], bits)
            cv_row = []
            tex_row = model_key.replace("_", r"\_")
            print_row = "  {:<22}".format(model_key)
            for comp in COMP_ORDER:
                vals = np.array([r["recon_error"] for r in recs if r["comp"] == comp])
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 2 and vals.mean() > 1e-12:
                    cv = 100.0 * vals.std() / vals.mean()
                    cv_row.append(cv)
                    tex_row += " & {:.1f}".format(cv)
                    print_row += " {:>10.1f}".format(cv)
                else:
                    cv_row.append(float("nan"))
                    tex_row += " & --"
                    print_row += " {:>10}".format("--")
            print(print_row)
            lines.append(tex_row + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Coefficient of variation (\%) of reconstruction error across layers "
                     r"within each component type ({}bit).}}".format(bits))
        lines.append(r"\label{{tab:recon_cv_{}bit}}".format(bits))
        lines.append(r"\end{table}")
        tex = "\n".join(lines)
        out = os.path.join(output_dir, "recon_cv_{}bit.tex".format(bits))
        with open(out, "w") as f:
            f.write(tex)
        print("Saved: {}".format(out))


def print_variance_decomposition_table(all_data, bits_list, output_dir):
    for bits in bits_list:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Comp. type} & \textbf{Layer} & "
                     r"\textbf{Recon err} & \textbf{Hessian sens} \\")
        lines.append(r"& ($\eta^2$) & ($R^2$) & ($R^2$) & ($R^2$) \\")
        lines.append(r"\midrule")
        sep = "=" * 100
        print("\n" + sep)
        print("VARIANCE DECOMPOSITION  bits={}  (component eta^2 vs layer R^2 vs recon R^2 vs hsens R^2)".format(bits))
        print(sep)
        print("  {:<22} {:>12} {:>12} {:>12} {:>12}".format(
            "Model", "Comp eta^2", "Layer R^2", "Recon R^2", "Hsens R^2"))
        print("-" * 100)
        for mk in all_data:
            recs = flatten_records(all_data[mk], bits)
            if not recs:
                continue
            ys = np.array([r["ppl_delta"] for r in recs])
            groups = [[r["ppl_delta"] for r in recs if r["comp"] == c and math.isfinite(r["ppl_delta"])]
                      for c in COMP_ORDER]
            eta2 = eta_squared_oneway(groups)
            xs_ly = np.array([r["layer"] for r in recs], dtype=float)
            _, _, r_ly, _ = corr_stats(xs_ly, ys)
            r2_ly = r_ly * r_ly if math.isfinite(r_ly) else float("nan")
            xs_re = np.array([r["recon_error"] for r in recs])
            _, _, r_re, _ = corr_stats(xs_re, ys)
            r2_re = r_re * r_re if math.isfinite(r_re) else float("nan")
            xs_hs = np.array([r["hessian_sens"] for r in recs])
            if np.isfinite(xs_hs).any():
                _, _, r_hs, _ = corr_stats(xs_hs, ys)
                r2_hs = r_hs * r_hs if math.isfinite(r_hs) else float("nan")
            else:
                r2_hs = float("nan")

            def fmt(v):
                return "{:.3f}".format(v) if math.isfinite(v) else "--"

            print("  {:<22} {:>12} {:>12} {:>12} {:>12}".format(
                mk, fmt(eta2), fmt(r2_ly), fmt(r2_re), fmt(r2_hs)))
            tex_row = "{} & {} & {} & {} & {} \\\\".format(
                mk.replace("_", r"\_"), fmt(eta2), fmt(r2_ly), fmt(r2_re), fmt(r2_hs))
            lines.append(tex_row)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Variance decomposition of $\Delta$PPL at {}bit RTN. "
                     r"Component type ($\eta^2$ from one-way ANOVA) consistently exceeds "
                     r"reconstruction error and Hessian-trace sensitivity ($R^2$).}}".format(bits))
        lines.append(r"\label{{tab:variance_decomp_{}bit}}".format(bits))
        lines.append(r"\end{table}")
        tex = "\n".join(lines)
        out = os.path.join(output_dir, "variance_decomp_{}bit.tex".format(bits))
        with open(out, "w") as f:
            f.write(tex)
        print("Saved: {}".format(out))


def print_mp_summary_table(all_data, output_dir):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Dom.} & \textbf{U-4 (4.0b)} & "
                 r"\textbf{CA (3.25b)} & \textbf{Inv (3.75b)} & \textbf{U-3 (3.0b)} \\")
    lines.append(r"\midrule")
    sep = "=" * 100
    print("\n" + sep)
    print("MIXED-PRECISION RESULTS SUMMARY")
    print(sep)
    print("  {:<18} {:>5} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Dom", "U-4", "CA", "Inv", "U-3"))
    print("-" * 100)
    any_found = False
    for mk in all_data:
        md = all_data[mk]
        mp = md.get("mp_experiment", None)
        if not mp:
            continue
        any_found = True
        dom = md.get("mp_dominant_info", {}).get("used_dominant", "?")

        def delta_of(key_pat):
            for name, rec in mp.items():
                if key_pat in name:
                    return rec.get("delta", float("nan"))
            return float("nan")

        d_u4 = delta_of("uniform_4bit")
        d_u3 = delta_of("uniform_3bit")
        d_ca = delta_of("ca_")
        d_inv = delta_of("inv_")

        def fmt(v):
            return "{:.2f}".format(v) if math.isfinite(v) else "--"

        bold_ca = math.isfinite(d_ca) and math.isfinite(d_inv) and d_ca < d_inv
        ca_cell = r"\textbf{{{}}}".format(fmt(d_ca)) if bold_ca else fmt(d_ca)
        print("  {:<18} {:>5} {:>12} {:>12} {:>12} {:>12}".format(
            mk, dom, fmt(d_u4), fmt(d_ca) + ("*" if bold_ca else ""), fmt(d_inv), fmt(d_u3)))
        lines.append("{} & {} & {} & {} & {} & {} \\\\".format(
            mk.replace("_", r"\_"), dom, fmt(d_u4), ca_cell, fmt(d_inv), fmt(d_u3)))
    if not any_found:
        print("  (no MP experiment results found)")
        return
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{$\Delta$PPL for mixed-precision strategies under RTN. "
                 r"Bold CA entries indicate CA (3.25-bit avg) outperforms Inv (3.75-bit avg) "
                 r"despite using fewer total bits.}")
    lines.append(r"\label{tab:mp_summary}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    out = os.path.join(output_dir, "mp_summary.tex")
    with open(out, "w") as f:
        f.write(tex)
    print("Saved: {}".format(out))


def print_dominance_table(all_data, bits_list, output_dir):
    for bits in bits_list:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lccccl}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Q\%} & \textbf{K\%} & \textbf{V\%} & "
                     r"\textbf{O\%} & \textbf{Dom.} \\")
        lines.append(r"\midrule")
        sep = "=" * 100
        print("\n" + sep)
        print("COMPONENT DOMINANCE (% of total positive PPL Delta)  bits={}".format(bits))
        print(sep)
        print("  {:<22} {:>8} {:>8} {:>8} {:>8}  {:<6}".format(
            "Model", "Q%", "K%", "V%", "O%", "Dom"))
        print("-" * 100)
        for mk in all_data:
            recs = flatten_records(all_data[mk], bits)
            totals = {c: 0.0 for c in COMP_ORDER}
            for r in recs:
                if r["comp"] in totals and math.isfinite(r["ppl_delta"]) and r["ppl_delta"] > 0:
                    totals[r["comp"]] += r["ppl_delta"]
            S = sum(totals.values())
            if S <= 0:
                continue
            pct = {c: 100.0 * totals[c] / S for c in COMP_ORDER}
            dom = max(pct, key=lambda c: pct[c])
            print("  {:<22} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:<6}".format(
                mk, pct["Q"], pct["K"], pct["V"], pct["O"], dom))
            cells = []
            for c in COMP_ORDER:
                v = pct[c]
                if v > 35.0:
                    cells.append(r"\textbf{{{:.0f}}}".format(v))
                else:
                    cells.append("{:.0f}".format(v))
            lines.append("{} & {} & {} & {} & {} & {} \\\\".format(
                mk.replace("_", r"\_"), cells[0], cells[1], cells[2], cells[3], dom))
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Percentage of total positive $\Delta$PPL by component ({}bit). "
                     r"Bold indicates $>$35\%.}}".format(bits))
        lines.append(r"\label{{tab:dominance_{}bit}}".format(bits))
        lines.append(r"\end{table}")
        tex = "\n".join(lines)
        out = os.path.join(output_dir, "dominance_{}bit.tex".format(bits))
        with open(out, "w") as f:
            f.write(tex)
        print("Saved: {}".format(out))


def print_proj_split_correlation_report(all_data, bits_list):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Density"),
        ("mp_n_outliers", "N MP Outliers"),
        ("kurtosis", "Kurtosis"),
        ("recon_error", "Recon Error"),
        ("hessian_sens", "Hessian Sens"),
    ]
    group_keys = ["QO_square", "KV_square", "KV_rect"]
    group_desc = {
        "QO_square": "Q/O  (square, MP valid)",
        "KV_square": "K/V  (square, MHA)",
        "KV_rect":   "K/V  (rect,   GQA -- MP suspect)",
    }
    sep = "=" * 110
    print("\n" + sep)
    print("PROJECTION-TYPE SPLIT CORRELATION REPORT")
    print(sep)
    for bits in bits_list:
        print("\n  Quantization: {}bit RTN".format(bits))
        for model_key, model_data in all_data.items():
            records = flatten_records(model_data, bits)
            if not records:
                continue
            print("\n    Model: {}".format(model_key))
            print("    " + "-" * 100)
            header = "    {:<32} {:<28} {:>10} {:>10} {:>10} {:>6}".format(
                "Metric", "Proj Group", "Spearman r", "p-value", "Pearson r", "n")
            print(header)
            print("    " + "-" * 100)
            for gkey in group_keys:
                grecs = [r for r in records if r["proj_group_key"] == gkey]
                if not grecs:
                    continue
                for mkey, mlabel in x_metrics:
                    xs = np.array([r[mkey] for r in grecs])
                    ys = np.array([r["ppl_delta"] for r in grecs])
                    if not np.isfinite(xs).any():
                        continue
                    rho, p_s, r_lin, _ = corr_stats(xs, ys)
                    n = int((np.isfinite(xs) & np.isfinite(ys)).sum())
                    if math.isfinite(rho):
                        print("    {:<32} {:<28} {:>10.4f} {:>10.3e} {:>10.4f} {:>6}".format(
                            mlabel, group_desc[gkey], rho, p_s, r_lin, n))
    print("\n" + sep)


def print_latex_table_proj_split(all_data, bits_list, output_dir):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Density"),
        ("recon_error", "Recon Error"),
    ]
    group_keys = ["QO_square", "KV_square", "KV_rect"]
    group_short = {
        "QO_square": "Q/O",
        "KV_square": "K/V MHA",
        "KV_rect":   "K/V GQA",
    }
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    n_sub = len(group_keys) * 2
    n_metric_cols = len(x_metrics)
    col_spec = "l" + "r" * (n_metric_cols * n_sub * len(bits_list))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_top = "Model"
    for bits in bits_list:
        header_top += " & \\multicolumn{{{}}}{{c}}{{{}bit RTN}}".format(
            n_metric_cols * n_sub, bits)
    lines.append(header_top + r" \\")
    header_mid = ""
    for bits in bits_list:
        for _, mlabel in x_metrics:
            header_mid += " & \\multicolumn{{{}}}{{c}}{{{}}}".format(
                n_sub, mlabel.replace("_", " "))
    lines.append("" + header_mid + r" \\")
    header_sub = ""
    for bits in bits_list:
        for _ in x_metrics:
            for gkey in group_keys:
                header_sub += " & \\multicolumn{{2}}{{c}}{{{}}}".format(
                    group_short[gkey])
    lines.append("" + header_sub + r" \\")
    header_rho = ""
    for bits in bits_list:
        for _ in x_metrics:
            for _ in group_keys:
                header_rho += r" & $\rho$ & $r$"
    lines.append("" + header_rho + r" \\")
    lines.append(r"\midrule")
    for model_key, model_data in all_data.items():
        row = model_key.replace("_", r"\_")
        for bits in bits_list:
            records = flatten_records(model_data, bits)
            for mkey, _ in x_metrics:
                for gkey in group_keys:
                    grecs = [r for r in records if r["proj_group_key"] == gkey]
                    xs = np.array([r[mkey] for r in grecs])
                    ys = np.array([r["ppl_delta"] for r in grecs])
                    rho, p_s, r_lin, _ = corr_stats(xs, ys)
                    if math.isfinite(rho):
                        sig = "**" if p_s < 0.001 else ("*" if p_s < 0.05 else "")
                        row += " & {:.3f}{} & {:.3f}".format(rho, sig, r_lin)
                    else:
                        row += " & -- & --"
        lines.append(row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Spearman ($\rho$) and Pearson ($r$) correlations between MP spectral metrics "
        r"and per-layer RTN quantization sensitivity (PPL delta), split by projection group. "
        r"Q/O projections are always square; K/V are square in MHA models and rectangular in GQA models "
        r"where the MP random-matrix assumption is geometrically suspect. "
        r"$^{**}$: $p < 0.001$, $^{*}$: $p < 0.05$.}"
    )
    lines.append(r"\label{tab:quant_sensitivity_proj_split}")
    lines.append(r"\end{table}")
    latex_str = "\n".join(lines)
    out_path = os.path.join(output_dir, "correlation_table_proj_split.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print("Saved LaTeX proj-split table: {}".format(out_path))


def print_latex_table(all_data, bits_list, output_dir):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Density"),
        ("kurtosis", "Kurtosis"),
        ("recon_error", "Recon Error"),
        ("hessian_sens", "Hessian Sens"),
    ]
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    n_metric_cols = len(x_metrics)
    col_spec = "l" + "r" * (n_metric_cols * 2 * len(bits_list))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_top = "Model"
    for bits in bits_list:
        header_top += " & \\multicolumn{{{}}}{{c}}{{{}bit RTN}}".format(n_metric_cols * 2, bits)
    lines.append(header_top + r" \\")
    header_mid = ""
    for bits in bits_list:
        for _, mlabel in x_metrics:
            header_mid += " & \\multicolumn{{2}}{{c}}{{{}}}".format(mlabel.replace("_", " "))
    lines.append("" + header_mid + r" \\")
    header_sub = ""
    for bits in bits_list:
        for _ in x_metrics:
            header_sub += r" & $\rho$ & $r$"
    lines.append("" + header_sub + r" \\")
    lines.append(r"\midrule")
    for model_key, model_data in all_data.items():
        row = model_key.replace("_", r"\_")
        for bits in bits_list:
            records = flatten_records(model_data, bits)
            for mkey, _ in x_metrics:
                xs = np.array([r[mkey] for r in records])
                ys = np.array([r["ppl_delta"] for r in records])
                rho, p_s, r_lin, _ = corr_stats(xs, ys)
                if math.isfinite(rho):
                    sig = "*" if p_s < 0.05 else ""
                    sig2 = "**" if p_s < 0.001 else sig
                    row += " & {:.3f}{} & {:.3f}".format(rho, sig2, r_lin)
                else:
                    row += " & -- & --"
        lines.append(row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Spearman ($\rho$) and Pearson ($r$) correlations between MP spectral metrics "
                 r"(including Hessian-trace sensitivity) and per-layer RTN quantization sensitivity (PPL delta). "
                 r"$^{**}$: $p < 0.001$, $^{*}$: $p < 0.05$.}")
    lines.append(r"\label{tab:quant_sensitivity_correlation}")
    lines.append(r"\end{table}")
    latex_str = "\n".join(lines)
    out_path = os.path.join(output_dir, "correlation_table.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print("Saved LaTeX table: {}".format(out_path))


def print_full_correlation_report(all_data, bits_list):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Outlier Density"),
        ("mp_n_outliers", "N MP Outliers"),
        ("kurtosis", "Kurtosis"),
        ("recon_error", "Recon Error"),
        ("hessian_sens", "Hessian Sens"),
    ]
    sep = "=" * 110
    print("\n" + sep)
    print("FULL CORRELATION REPORT")
    print(sep)
    for bits in bits_list:
        print("\n  Quantization: {}bit RTN".format(bits))
        print("  " + "-" * 107)
        header = "  {:<25} {:<20} {:>10} {:>10} {:>10} {:>8}".format(
            "Model", "Metric", "Spearman r", "p-value", "Pearson r", "n")
        print(header)
        print("  " + "-" * 107)
        for model_key, model_data in all_data.items():
            records = flatten_records(model_data, bits)
            for mkey, mlabel in x_metrics:
                xs = np.array([r[mkey] for r in records])
                ys = np.array([r["ppl_delta"] for r in records])
                if not np.isfinite(xs).any():
                    continue
                rho, p_s, r_lin, _ = corr_stats(xs, ys)
                n = int((np.isfinite(xs) & np.isfinite(ys)).sum())
                if math.isfinite(rho):
                    print("  {:<25} {:<20} {:>10.4f} {:>10.3e} {:>10.4f} {:>8}".format(
                        model_key, mlabel, rho, p_s, r_lin, n))
        print()
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--output_dir", type=str, default="./output/quant_sensitivity/paper_figs")
    parser.add_argument("--split_proj_type", action="store_true", default=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--skip_per_model_figs", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    input_paths = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_paths.extend(expanded)
        else:
            input_paths.append(pattern)
    all_data = load_results(input_paths)
    if not all_data:
        print("No valid results loaded from: {}".format(input_paths))
        return
    print("Loaded {} model(s): {}".format(len(all_data), list(all_data.keys())))
    for bits in args.bits:
        print("\nGenerating figures for {}bit...".format(bits))
        if not args.skip_per_model_figs:
            figure_per_model_scatter(all_data, bits, args.output_dir)
            figure_layer_depth_profile(all_data, bits, args.output_dir)
            figure_comp_boxplot(all_data, bits, args.output_dir)
        figure_combined_scatter(all_data, bits, args.output_dir)
        figure_variance_decomposition(all_data, bits, args.output_dir)
        figure_component_dominance(all_data, bits, args.output_dir)
        figure_v_proj_natural_experiment(all_data, bits, args.output_dir)
        if args.split_proj_type:
            figure_proj_split_scatter(all_data, bits, args.output_dir)
    print_full_correlation_report(all_data, args.bits)
    print_latex_table(all_data, args.bits, args.output_dir)
    print_within_component_r2_table(all_data, args.bits, args.output_dir, n_boot=args.n_boot)
    print_recon_cv_table(all_data, args.bits, args.output_dir)
    print_variance_decomposition_table(all_data, args.bits, args.output_dir)
    print_dominance_table(all_data, args.bits, args.output_dir)
    print_mp_summary_table(all_data, args.output_dir)
    if args.split_proj_type:
        print_proj_split_correlation_report(all_data, args.bits)
        print_latex_table_proj_split(all_data, args.bits, args.output_dir)


if __name__ == "__main__":
    main()