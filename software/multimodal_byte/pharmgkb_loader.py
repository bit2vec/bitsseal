"""PharmGKB clinical annotations loader.

Predict Phenotype Category (Efficacy / Toxicity / Dosage / Metabolism / Other)
from variant + gene + drug + phenotype + specialty + evidence-level modalities.
"""
import csv, os, random
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'pharmgkb_cache')


def _read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


# Map raw category to a normalized label set; fold rare-class -> Other
PHENOTYPE_CATEGORIES = {
    'Efficacy':   'Efficacy',
    'Toxicity':   'Toxicity',
    'Dosage':     'Dosage',
    'Metabolism/PK': 'Metabolism',
    'Other':      'Other',
}


def normalize_category(s):
    s = (s or '').strip()
    return PHENOTYPE_CATEGORIES.get(s, 'Other')


def load_pharmgkb(seed=42):
    """Returns: list of dicts with multimodal fields + label_idx, plus label_set."""
    ann_path = os.path.join(CACHE, 'clinical_annotations.tsv')
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f'{ann_path} missing. Download from https://api.pharmgkb.org/v1/download/file/data/clinicalAnnotations.zip')
    rows = _read_tsv(ann_path)
    examples = []
    for r in rows:
        cat = normalize_category(r.get('Phenotype Category', ''))
        if not cat: continue
        # Skip records with empty key fields
        gene = (r.get('Gene', '') or '').strip()
        variant = (r.get('Variant/Haplotypes', '') or '').strip()
        drug = (r.get('Drug(s)', '') or '').strip()
        phenotype = (r.get('Phenotype(s)', '') or '').strip()
        specialty = (r.get('Specialty Population', '') or '').strip()
        level = (r.get('Level of Evidence', '') or '').strip()
        if not (gene and variant): continue
        examples.append({
            'variant':      variant,
            'gene':         gene,
            'drug':         drug,
            'phenotype':    phenotype,
            'specialty':    specialty if specialty else 'general',
            'evidence_level': f'level_{level}' if level else 'unknown',
            'label':        cat,
        })
    label_set = sorted(set(e['label'] for e in examples))
    label2idx = {l: i for i, l in enumerate(label_set)}
    for ex in examples: ex['label_idx'] = label2idx[ex['label']]

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples, label_set


if __name__ == '__main__':
    exs, labs = load_pharmgkb()
    print(f'Loaded {len(exs)} examples, {len(labs)} classes: {labs}')
    print('Class distribution:', Counter(e['label'] for e in exs))
    print('\nExample 0:')
    for k, v in exs[0].items():
        print(f'  {k}: {str(v)[:120]}')
