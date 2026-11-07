#!/usr/bin/env python3
"""Plot the num_bits ablation with a shaded mean+/-std band (real per-seed data)."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--subword', type=float, default=None,
                   help='subword baseline acc to draw as a reference line')
    args = p.parse_args()
    d = json.load(open(args.json))
    B = sorted(int(k) for k in d)
    mean = np.array([d[str(b)]['mean'] for b in B])
    std = np.array([d[str(b)]['std'] for b in B])

    plt.figure(figsize=(4.2, 2.8))
    plt.plot(B, mean, '-o', color='#1f4e79', lw=1.6, ms=4, label='SpinHash (Spin-Random)')
    plt.fill_between(B, mean - std, mean + std, color='#1f4e79', alpha=0.18,
                     label=r'$\pm$1 s.d. (3 seeds)')
    if args.subword is not None:
        plt.axhline(args.subword, ls='--', color='gray', lw=1.0, label='Subword table')
    bstar = B[int(np.argmax(mean))]
    plt.axvline(bstar, ls=':', color='crimson', lw=1.0)
    plt.text(bstar, plt.ylim()[0], f' B*={bstar}', color='crimson', fontsize=8, va='bottom')
    plt.xscale('log', base=2); plt.xticks(B, [str(b) for b in B], fontsize=7, rotation=0)
    plt.xlabel('spin-code length $B$'); plt.ylabel('test accuracy')
    plt.legend(fontsize=7, frameon=False, loc='lower right')
    plt.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.savefig(args.out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
