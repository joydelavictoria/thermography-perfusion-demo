"""
Thermography heterogeneity demo (synthetic data).

Question: can spatial heterogeneity of skin temperature carry information that the
mean temperature does not, and how do we test it without fooling ourselves when
frames from the same animal are not independent?

Four steps, one function each:
  1. make_dataset      : synthetic thermal videos, two groups, mean temperature matched
  2. frame_features    : simple spatial and temporal descriptors per frame / per video
  3. animal_level_test : collapse to one value per animal, exact permutation test, BH correction
  4. simulate_false_positives : frame-level vs animal-level testing when there is NO true effect

All data are synthetic. Nothing here is a clinical result.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

# ---- assumptions (all synthetic, all adjustable) -------------------------------------
SIZE = 64            # frame is SIZE x SIZE pixels
N_FRAMES = 30        # frames per animal (a short video)
N_PER_GROUP = 6      # animals per group
ANIMAL_SD = 0.8      # between-animal SD of mean skin temperature, degC
NOISE_SD = 0.05      # pixel noise, degC
GRADIENT = 2.0       # distal cooling across the frame, degC
BASE_T = 31.0        # baseline mean skin temperature, degC
COLD_DELTA = 1.0     # "cold area" = pixel more than this far below the frame median, degC
BIN_WIDTH = 0.1      # fixed intensity bin width for entropy, degC (fixed, not data-driven)
GLCM_RANGE = (26.0, 36.0)  # fixed temperature range quantized into GLCM gray levels
GLCM_LEVELS = 32

FEATURES = ["mean_temp", "sd_temp", "cold_fraction", "entropy", "glcm_contrast", "temporal_change"]

COLORS = {"A": "#2a6f97", "B": "#c8553d"}  # A homogeneous, B heterogeneous


# ---- Step 1: synthetic data -------------------------------------------------------------
def make_animal(rng, group):
    """Return a (N_FRAMES, SIZE, SIZE) video in degC for one animal.

    Group A: mild patchiness (amplitude 0.3 degC). Group B: strong patchiness (1.5 degC).
    Every frame is shifted so its mean equals the animal's baseline: the groups differ
    in spatial heterogeneity but NOT in mean temperature.
    """
    amp, n_blobs = (0.3, 4) if group == "A" else (1.5, 6)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    baseline = BASE_T + rng.normal(0, ANIMAL_SD)             # between-animal offset
    gradient = -GRADIENT * (yy / SIZE)
    centers = rng.integers(10, SIZE - 10, size=(n_blobs, 2))  # fixed per animal
    phases = rng.uniform(0, 2 * np.pi, n_blobs)
    frames = np.empty((N_FRAMES, SIZE, SIZE))
    for t in range(N_FRAMES):
        field = gradient.copy()
        for (cy, cx), ph in zip(centers, phases):
            mod = 1 + 0.2 * np.sin(2 * np.pi * t / N_FRAMES + ph)  # slow temporal fluctuation
            field -= amp * mod * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2)))
        field += rng.normal(0, NOISE_SD, (SIZE, SIZE))
        field += baseline - field.mean()                      # match the mean
        frames[t] = field
    return frames


def make_dataset(seed=42):
    rng = np.random.default_rng(seed)
    videos = {}
    for g in ("A", "B"):
        for i in range(N_PER_GROUP):
            videos[(g, f"{g}{i + 1}")] = make_animal(rng, g)
    return videos


# ---- Step 2: features ------------------------------------------------------------------
def frame_features(img):
    """Spatial descriptors for one frame (2D array in degC)."""
    mean, sd = img.mean(), img.std(ddof=0)
    cold_fraction = float(np.mean(img < np.median(img) - COLD_DELTA))
    edges = np.arange(img.min(), img.max() + BIN_WIDTH, BIN_WIDTH)
    p, _ = np.histogram(img, bins=edges)
    p = p[p > 0] / p.sum()
    entropy = float(-(p * np.log2(p)).sum())
    lo, hi = GLCM_RANGE
    q = np.clip(((img - lo) / (hi - lo) * GLCM_LEVELS).astype(int), 0, GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, "contrast").mean())
    return dict(mean_temp=mean, sd_temp=sd, cold_fraction=cold_fraction,
                entropy=entropy, glcm_contrast=contrast)


def video_features(video):
    """Per-frame table plus one temporal descriptor (mean absolute frame-to-frame change)."""
    rows = [frame_features(f) for f in video]
    df = pd.DataFrame(rows)
    df["frame"] = np.arange(len(video))
    temporal = float(np.mean(np.abs(np.diff(video, axis=0))))
    return df, temporal


def build_tables(videos):
    frame_rows, animal_rows = [], []
    for (g, aid), v in videos.items():
        df, temporal = video_features(v)
        df.insert(0, "animal", aid)
        df.insert(0, "group", g)
        df["temporal_change"] = temporal            # same value repeated per frame of the animal
        frame_rows.append(df)
        animal = df[FEATURES].mean()
        animal["group"], animal["animal"] = g, aid
        animal_rows.append(animal)
    return pd.concat(frame_rows, ignore_index=True), pd.DataFrame(animal_rows)


# ---- Step 3: animal-level inference ----------------------------------------------------
def exact_permutation_p(a, b):
    """Two-sided exact permutation test on the difference of group means.

    Enumerates every relabeling of the pooled animals. With 6 + 6 animals there are
    C(12,6) = 924 labelings, so the smallest attainable p-value is 2/924 = 0.0022.
    """
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(a)
    obs = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(n), k):
        mask = np.zeros(n, bool)
        mask[list(idx)] = True
        d = abs(pooled[mask].mean() - pooled[~mask].mean())
        count += d >= obs - 1e-12
        total += 1
    return count / total


def animal_level_test(animals, frames):
    rows = []
    for f in FEATURES:
        a = animals.loc[animals.group == "A", f].astype(float).values
        b = animals.loc[animals.group == "B", f].astype(float).values
        fa = frames.loc[frames.group == "A", f].values   # naive: every frame is a sample
        fb = frames.loc[frames.group == "B", f].values
        rows.append(dict(
            feature=f,
            mean_A=a.mean(), mean_B=b.mean(),
            p_animal_permutation=exact_permutation_p(a, b),
            p_frame_level_naive_t=stats.ttest_ind(fa, fb, equal_var=False).pvalue,
        ))
    res = pd.DataFrame(rows)
    res["q_animal_BH"] = stats.false_discovery_control(res["p_animal_permutation"].values, method="bh")
    return res


# ---- Step 4: why animal-level? ---------------------------------------------------------
def simulate_false_positives(nsim=2000, n=N_PER_GROUP, frames=N_FRAMES,
                             sd_between=1.0, sd_within=0.2, seed=1):
    """Both groups come from the SAME distribution (no true effect). Count how often each
    analysis reports p < 0.05. Frames from one animal share that animal's offset."""
    rng = np.random.default_rng(seed)
    combos = np.array([c for c in itertools.combinations(range(2 * n), n)])
    sel = np.zeros((len(combos), 2 * n), bool)
    sel[np.arange(len(combos))[:, None], combos] = True
    fp_frame = fp_animal = 0
    for _ in range(nsim):
        off = rng.normal(0, sd_between, 2 * n)
        y = off[:, None] + rng.normal(0, sd_within, (2 * n, frames))   # animals x frames
        ga, gb = y[:n].ravel(), y[n:].ravel()
        fp_frame += stats.ttest_ind(ga, gb, equal_var=False).pvalue < 0.05
        m = y.mean(1)
        s1 = (sel * m).sum(1) / n
        s2 = ((~sel) * m).sum(1) / n
        d = np.abs(s1 - s2)
        obs = abs(m[:n].mean() - m[n:].mean())
        fp_animal += (d >= obs - 1e-12).mean() < 0.05
    return dict(nsim=nsim, animals_per_group=n, frames_per_animal=frames,
                fp_rate_frame_level=fp_frame / nsim, fp_rate_animal_level=fp_animal / nsim,
                min_attainable_p=2 / len(combos))


# ---- figures ---------------------------------------------------------------------------
def fig_frames(videos, path):
    a = videos[("A", "A1")][0]
    b = videos[("B", "B1")][0]
    # shift both to a common mean so the two panels share one color scale
    m = 0.5 * (a.mean() + b.mean())
    a, b = a - a.mean() + m, b - b.mean() + m
    vmin, vmax = min(a.min(), b.min()), max(a.max(), b.max())
    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.75, 1.3], wspace=0.15)
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[4])]
    cax = fig.add_subplot(gs[2])
    for x, img, t in ((ax[0], a, "Group A (mild patchiness)"), (ax[1], b, "Group B (strong patchiness)")):
        im = x.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
        x.set_title(t, fontsize=10); x.axis("off")
    fig.colorbar(im, cax=cax, label="skin temperature (degC)")
    bins = np.arange(vmin, vmax + 0.1, 0.1)
    ax[2].hist(a.ravel(), bins, alpha=0.6, color=COLORS["A"], label=f"A  mean {a.mean():.2f}  SD {a.std():.2f}")
    ax[2].hist(b.ravel(), bins, alpha=0.6, color=COLORS["B"], label=f"B  mean {b.mean():.2f}  SD {b.std():.2f}")
    ax[2].set_xlabel("temperature (degC)"); ax[2].set_ylabel("pixels"); ax[2].legend(fontsize=8, frameon=False)
    ax[2].set_title("Same mean, different spread", fontsize=10)
    for s in ("top", "right"): ax[2].spines[s].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_animal_level(animals, res, path):
    feats = ["mean_temp", "sd_temp", "glcm_contrast"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    rng = np.random.default_rng(0)
    for x, f in zip(ax, feats):
        for i, g in enumerate(("A", "B")):
            v = animals.loc[animals.group == g, f].astype(float).values
            x.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, color=COLORS[g], s=36, zorder=3)
            x.hlines(v.mean(), i - 0.22, i + 0.22, color="k", lw=1.5)
        p = res.loc[res.feature == f, "p_animal_permutation"].iloc[0]
        x.set_xticks([0, 1]); x.set_xticklabels(["A", "B"]); x.set_xlim(-0.5, 1.5)
        x.set_title(f"{f}\nexact permutation p = {p:.3f}", fontsize=9)
        for s in ("top", "right"): x.spines[s].set_visible(False)
    ax[0].set_ylabel("one dot = one animal")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_simulation(sim, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    vals = [sim["fp_rate_frame_level"], sim["fp_rate_animal_level"]]
    bars = ax.bar(["frames as samples", "one value\nper animal"], vals, color=["#c8553d", "#2a6f97"], width=0.55)
    ax.axhline(0.05, color="k", ls="--", lw=1); ax.text(0.5, 0.22, "dashed line = nominal 0.05", ha="center", fontsize=8)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("false-positive rate (no true effect)"); ax.set_ylim(0, 1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ---- main ------------------------------------------------------------------------------
def main(seed=42, nsim=2000, outdir="."):
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    videos = make_dataset(seed)
    frames, animals = build_tables(videos)
    res = animal_level_test(animals, frames)
    sim = simulate_false_positives(nsim=nsim)
    frames.to_csv(os.path.join(outdir, "results", "frame_features.csv"), index=False)
    animals.to_csv(os.path.join(outdir, "results", "animal_features.csv"), index=False)
    res.to_csv(os.path.join(outdir, "results", "group_comparison.csv"), index=False)
    pd.DataFrame([sim]).to_csv(os.path.join(outdir, "results", "simulation_summary.csv"), index=False)
    fig_frames(videos, os.path.join(outdir, "figures", "fig1_equal_mean_different_heterogeneity.png"))
    fig_animal_level(animals, res, os.path.join(outdir, "figures", "fig2_animal_level_comparison.png"))
    fig_simulation(sim, os.path.join(outdir, "figures", "fig3_false_positive_simulation.png"))
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:.4f}")
    print("\nGroup comparison (6 vs 6 animals; seed %d)" % seed)
    print(res.to_string(index=False))
    print("\nSimulation, no true effect:")
    for k, v in sim.items(): print(f"  {k}: {v}")
    return res, sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    main(a.seed, a.nsim, a.outdir)
