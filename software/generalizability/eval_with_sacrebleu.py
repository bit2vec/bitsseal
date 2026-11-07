"""Post-hoc sacrebleu word-level BLEU eval on dumped dev predictions.

Reads dev_preds_{method}_{lang}.txt and dev_refs_{method}_{lang}.txt that
generalizability_coling.py emits at best-BLEU epoch when run with the patched
finetune loop. No model loading required.
"""

import argparse
import json
import os
import sys

import sacrebleu


def score_one(method: str, lang: str, dump_dir: str):
    preds_path = os.path.join(dump_dir, f'dev_preds_{method}_{lang}.txt')
    refs_path = os.path.join(dump_dir, f'dev_refs_{method}_{lang}.txt')
    if not os.path.isfile(preds_path) or not os.path.isfile(refs_path):
        return {'error': f'missing files preds={preds_path} refs={refs_path}'}

    with open(preds_path, encoding='utf-8') as f:
        preds = [line.rstrip('\n') for line in f]
    with open(refs_path, encoding='utf-8') as f:
        refs = [line.rstrip('\n') for line in f]

    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]

    bleu_word = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf_word = sacrebleu.corpus_chrf(preds, [refs], beta=2).score

    # Char-level BLEU (split into chars) for cross-comparison with internal metric.
    preds_chars = [' '.join(list(p)) for p in preds]
    refs_chars = [' '.join(list(r)) for r in refs]
    bleu_char = sacrebleu.corpus_bleu(preds_chars, [refs_chars],
                                      tokenize='none').score

    return {
        'bleu_word': bleu_word,
        'chrf_word': chrf_word,
        'bleu_char': bleu_char,
        'n_examples': n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dump_dir', default='./results/generalizability_v2')
    p.add_argument('--output_file',
                   default='./results/generalizability_v2/sacrebleu_results.json')
    args = p.parse_args()

    results = {}
    for method in ['standard', 'byte_seb', 'binary_seb']:
        results[method] = {}
        for lang in ['quy', 'gn']:
            res = score_one(method, lang, args.dump_dir)
            results[method][lang] = res
            print(f"{method} / {lang}: {res}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_file}")

    # Pretty summary table
    print("\n" + "=" * 80)
    print(f"{'method':<14} {'quy BLEU_w':<12} {'quy ChrF_w':<12} "
          f"{'gn BLEU_w':<12} {'gn ChrF_w':<12}")
    print('-' * 80)
    for m in ['standard', 'byte_seb', 'binary_seb']:
        q = results[m].get('quy', {})
        g = results[m].get('gn', {})
        print(f"{m:<14} {q.get('bleu_word',0):<12.2f} {q.get('chrf_word',0):<12.2f} "
              f"{g.get('bleu_word',0):<12.2f} {g.get('chrf_word',0):<12.2f}")


if __name__ == '__main__':
    main()
