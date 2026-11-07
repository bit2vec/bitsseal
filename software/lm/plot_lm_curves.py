"""Plot training loss, train PPL, and valid PPL vs epoch for the three LM
methods (Subword, EmByte, 001). Final test PPL is shown as a horizontal
marker since per-epoch test PPL wasn't tracked.

Data sources:
- Subword: extracted from results/language_modeling/log.txt
- EmByte/001: results/language_modeling/results_{byte_seb,binary_seb}.json
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE = 'results/language_modeling'


def parse_subword_log(path):
    train_ppls, valid_ppls, train_losses, valid_losses = [], [], [], []
    with open(path) as f:
        text = f.read()
    blocks = re.finditer(
        r'Train Loss:\s*([\d.]+),\s*Train PPL:\s*([\d.]+)\s*\n\s*Valid Loss:\s*([\d.]+),\s*Valid PPL:\s*([\d.]+)',
        text,
    )
    for m in blocks:
        train_losses.append(float(m.group(1)))
        train_ppls.append(float(m.group(2)))
        valid_losses.append(float(m.group(3)))
        valid_ppls.append(float(m.group(4)))
    return {'train_losses': train_losses, 'train_ppls': train_ppls,
            'valid_losses': valid_losses, 'valid_ppls': valid_ppls}


def main():
    subword = parse_subword_log(os.path.join(BASE, 'log.txt'))
    byte = json.load(open(os.path.join(BASE, 'results_byte_seb.json')))
    binary = json.load(open(os.path.join(BASE, 'results_binary_seb.json')))

    test_ppls = {
        'Subword': None,                  # not computed in log
        'EmByte': byte.get('test_ppl'),
        '001 (binary)': binary.get('test_ppl'),
    }

    curves = {
        'Subword':         subword,
        'EmByte':          byte,
        '001 (binary)':    binary,
    }
    colors = {'Subword': 'C0', 'EmByte': 'C1', '001 (binary)': 'C3'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Train loss ---
    ax = axes[0]
    for name, c in curves.items():
        y = c['train_losses']
        ax.plot(range(1, len(y) + 1), y, label=name, color=colors[name], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (avg cross-entropy)')
    ax.set_title('Train Loss per Epoch')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Train PPL ---
    ax = axes[1]
    for name, c in curves.items():
        y = c['train_ppls']
        ax.plot(range(1, len(y) + 1), y, label=name, color=colors[name], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Perplexity')
    ax.set_yscale('log')
    ax.set_title('Train PPL per Epoch')
    ax.grid(alpha=0.3, which='both')
    ax.legend()

    # --- Valid PPL ---
    ax = axes[2]
    for name, c in curves.items():
        y = c['valid_ppls']
        ax.plot(range(1, len(y) + 1), y, label=name, color=colors[name], marker='o')
        if test_ppls[name] is not None:
            ax.axhline(test_ppls[name], color=colors[name], linestyle='--', alpha=0.5,
                       label=f'{name} test PPL = {test_ppls[name]:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Valid Perplexity')
    ax.set_title('Valid PPL per Epoch\n(dashed = final test PPL)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('LM Training Curves on WikiText-103 (GPT-2 BPE vocab, 15-epoch runs)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    out = os.path.join(BASE, 'lm_curves.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"wrote {out}")
    # Also print a summary table
    print()
    print(f"{'method':<15}{'final train PPL':>18}{'final valid PPL':>18}{'test PPL':>12}")
    for name, c in curves.items():
        tp = f"{test_ppls[name]:.2f}" if test_ppls[name] is not None else "n/a"
        print(f"{name:<15}{c['train_ppls'][-1]:>18.2f}{c['valid_ppls'][-1]:>18.2f}{tp:>12}")


if __name__ == '__main__':
    main()
