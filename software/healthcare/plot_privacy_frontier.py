#!/usr/bin/env python3
"""Privacy frontier plot: model size (log MB) vs FILM attack recall.

Goal: show 001 sits in the lower-left (small + private) corner alone, while
Subword sits in the upper-right (large + leaky), with EmByte in between.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '.'
HC = os.path.join(ROOT, 'results/healthcare')
DM = os.path.join(ROOT, 'results/domains')
FILM = os.path.join(ROOT, 'results/healthcare_film')

DATASETS = [
    ('ade',       HC, 'ADE'),
    ('pubmedqa',  HC, 'PubMedQA'),
    ('medmcqa',   HC, 'MedMCQA'),
    ('finance',   DM, 'Finance'),
    ('education', DM, 'OpenBookQA'),
    ('legal',     DM, 'LEDGAR'),
    ('mental',    DM, 'GoEmotions'),
]
METHODS = ['standard', 'byte_seb', 'binary_seb']
LABELS = {'standard': 'Subword', 'byte_seb': 'EmByte', 'binary_seb': '001'}
COLORS = {'standard': '#d62728', 'byte_seb': '#ff7f0e', 'binary_seb': '#2ca02c'}
MARKERS = {'standard': 'o', 'byte_seb': 's', 'binary_seb': '^'}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Total model size vs FILM recall
for m in METHODS:
    xs, ys, labels = [], [], []
    for short, base, full in DATASETS:
        acc = load(os.path.join(base, short, f'{short}_{m}.json'))
        film = load(os.path.join(FILM, f'{short}_{m}_film.json'))
        if acc and film:
            size_mb = acc['total_params'] * 4 / 1024 / 1024  # fp32 bytes
            recall = film['metrics']['recall']
            xs.append(size_mb); ys.append(recall); labels.append(full)
    ax1.scatter(xs, ys, label=LABELS[m], color=COLORS[m], marker=MARKERS[m],
                s=120, alpha=0.85, edgecolor='black', linewidth=0.5)
    for x, y, lbl in zip(xs, ys, labels):
        ax1.annotate(lbl, (x, y), textcoords='offset points', xytext=(6, 4),
                     fontsize=8, color=COLORS[m])

ax1.set_xscale('log')
ax1.set_xlabel('Total model size (MB, log scale)', fontsize=12)
ax1.set_ylabel('FILM attack recall — fraction of input tokens recovered', fontsize=12)
ax1.set_title('Privacy frontier — total model', fontsize=13)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(alpha=0.3)
ax1.legend(loc='center right', fontsize=11, framealpha=0.9)
ax1.axhline(0.05, linestyle='--', color='gray', alpha=0.5)
ax1.text(ax1.get_xlim()[0]*1.5, 0.06, 'effective privacy threshold', fontsize=8, color='gray')

# Plot 2: Embedding-only size vs FILM recall (where 001 dominates)
for m in METHODS:
    xs, ys, labels = [], [], []
    for short, base, full in DATASETS:
        acc = load(os.path.join(base, short, f'{short}_{m}.json'))
        film = load(os.path.join(FILM, f'{short}_{m}_film.json'))
        if acc and film:
            emb_mb = acc['embed_params'] * 4 / 1024 / 1024
            recall = film['metrics']['recall']
            xs.append(emb_mb); ys.append(recall); labels.append(full)
    ax2.scatter(xs, ys, label=LABELS[m], color=COLORS[m], marker=MARKERS[m],
                s=120, alpha=0.85, edgecolor='black', linewidth=0.5)
    for x, y, lbl in zip(xs, ys, labels):
        ax2.annotate(lbl, (x, y), textcoords='offset points', xytext=(6, 4),
                     fontsize=8, color=COLORS[m])

ax2.set_xscale('log')
ax2.set_xlabel('Embedding parameters (MB, log scale)', fontsize=12)
ax2.set_ylabel('FILM attack recall — fraction of input tokens recovered', fontsize=12)
ax2.set_title('Privacy frontier — embedding only', fontsize=13)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(alpha=0.3)
ax2.legend(loc='center right', fontsize=11, framealpha=0.9)
ax2.axhline(0.05, linestyle='--', color='gray', alpha=0.5)

# 001's "lower-left" highlight
for ax in (ax1, ax2):
    ax.axvspan(ax.get_xlim()[0], ax.get_xlim()[1] / 6, alpha=0.05, color='green')

plt.suptitle("001 occupies the 'small + private' frontier", fontsize=14, y=1.02)
plt.tight_layout()
out = os.path.join(ROOT, 'results/privacy_frontier.png')
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"wrote {out}")

# Also save a single-panel version (embedding-only) for the paper figure
fig2, ax = plt.subplots(figsize=(8, 6))
for m in METHODS:
    xs, ys, labels = [], [], []
    for short, base, full in DATASETS:
        acc = load(os.path.join(base, short, f'{short}_{m}.json'))
        film = load(os.path.join(FILM, f'{short}_{m}_film.json'))
        if acc and film:
            emb_mb = acc['embed_params'] * 4 / 1024 / 1024
            recall = film['metrics']['recall']
            xs.append(emb_mb); ys.append(recall); labels.append(full)
    ax.scatter(xs, ys, label=LABELS[m], color=COLORS[m], marker=MARKERS[m],
               s=180, alpha=0.85, edgecolor='black', linewidth=0.7)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(8, 6),
                    fontsize=9, color=COLORS[m])
ax.set_xscale('log')
ax.set_xlabel('Embedding memory (MB, log scale)', fontsize=13)
ax.set_ylabel('FILM attack recall (lower = more private)', fontsize=13)
ax.set_title('001 is the only method in the small-and-private corner', fontsize=14)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=12, loc='center right')
ax.axhline(0.05, linestyle='--', color='gray', alpha=0.5)
ax.text(ax.get_xlim()[0]*2, 0.07, 'effective privacy threshold (recall < 5%)',
        fontsize=9, color='gray')
plt.tight_layout()
out2 = os.path.join(ROOT, 'results/privacy_frontier_single.png')
plt.savefig(out2, dpi=140, bbox_inches='tight')
print(f"wrote {out2}")
