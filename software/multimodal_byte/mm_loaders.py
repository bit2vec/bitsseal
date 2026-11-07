"""Pluggable multimodal-fusion dataset loaders for run_mm_generic.py.

Each loader returns (examples, modalities, label_names):
  examples    list[dict]  modality-name -> text, plus 'label_idx'
  modalities  list[(name, token_budget)]
  label_names list[str]

Datasets chosen so each fuses >=2 heterogeneous high-cardinality symbol fields,
the regime where a Subword union vocabulary explodes but Bits stays constant.
"""
import io, os, random, urllib.request


# ---------------------------------------------------------------- code-text
def load_codesearchnet(n_target=20000, seed=42):
    """code + docstring fusion. Task: code-comment MATCHING (binary).
    Positive = (code, its own docstring); negative = (code, a mismatched
    docstring). Both modalities are essential and high-vocab (identifiers)."""
    from datasets import load_dataset
    langs = ['python', 'java', 'go', 'javascript', 'php', 'ruby']
    rows = []
    for lg in langs:
        try:
            ds = load_dataset('sentence-transformers/codesearchnet', split='train',
                              streaming=True)
        except Exception:
            ds = None
        if ds is not None:
            # generic 2-column (comment, code) parquet; language-agnostic
            for i, ex in enumerate(ds):
                rows.append((ex.get('comment') or ex.get('docstring') or '',
                             ex.get('code') or ''))
                if len(rows) >= n_target: break
            break
    if not rows:  # fallback: per-language parquet mirror
        from datasets import load_dataset
        for lg in langs:
            try:
                ds = load_dataset('claudios/code_search_net', lg, split='train',
                                  streaming=True, trust_remote_code=True)
            except Exception:
                continue
            for ex in ds:
                rows.append((ex.get('func_documentation_string', ''),
                             ex.get('func_code_string', '')))
                if len(rows) >= n_target: break
            if rows: break
    rows = [(d, c) for d, c in rows if d.strip() and c.strip()]
    rng = random.Random(seed); rng.shuffle(rows)
    rows = rows[:n_target]
    docs = [d for d, _ in rows]
    examples = []
    for i, (doc, code) in enumerate(rows):
        examples.append({'code': code, 'docstring': doc, 'label_idx': 1})       # match
        neg = docs[(i + 1 + rng.randint(0, len(docs) - 2)) % len(docs)]
        examples.append({'code': code, 'docstring': neg, 'label_idx': 0})       # mismatch
    modalities = [('code', 256), ('docstring', 64)]
    return examples, modalities, ['mismatch', 'match']


# ---------------------------------------------------------------- molecule-text
def load_molecule(n_target=20000, seed=42):
    """compound name (text) + SMILES (chemical symbol language) fusion.
    Task: blood-brain-barrier penetration (binary). MoleculeNet BBBP."""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
    cache = os.path.join(os.path.dirname(__file__), 'BBBP.csv')
    if not os.path.exists(cache):
        urllib.request.urlretrieve(url, cache)
    import csv
    examples = []
    with open(cache, newline='') as f:
        for r in csv.DictReader(f):
            smi = (r.get('smiles') or '').strip()
            name = (r.get('name') or '').strip()
            lab = (r.get('p_np') or '').strip()
            if not smi or lab not in ('0', '1'):
                continue
            # space out SMILES so the tokenizer sees chemical sub-symbols
            smi_tok = ' '.join(smi)
            examples.append({'name': name, 'smiles': smi_tok, 'label_idx': int(lab)})
    modalities = [('name', 48), ('smiles', 256)]
    return examples[:n_target], modalities, ['non_penetrant', 'penetrant']


# ---------------------------------------------------------------- protein-text
def load_protein(n_target=20000, seed=42):
    """protein sequence + name/keywords text fusion. Task: predict broad
    functional family (top-N keyword classes). Swiss-Prot parquet mirror."""
    from datasets import load_dataset
    ds = None
    for name in ['khairi/uniprot-swissprot', 'tattabio/swissprot']:
        try:
            ds = load_dataset(name, split='train', streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError('no Swiss-Prot mirror loadable')
    from collections import Counter
    raw = []
    for ex in ds:
        seq = ex.get('sequence') or ex.get('Sequence') or ''
        name = ex.get('name') or ex.get('protein_name') or ex.get('Protein names') or ''
        fam = (ex.get('family') or ex.get('Protein families')
               or (ex.get('EC number') or '').split('.')[0] or '')
        if seq and fam:
            # space out residues into tokens
            raw.append({'sequence': ' '.join(seq[:1000]), 'name': str(name),
                        'family': str(fam).split(',')[0].split(';')[0].strip()})
        if len(raw) >= n_target * 2:
            break
    fams = [r['family'] for r in raw]
    top = [f for f, _ in Counter(fams).most_common(10)]
    f2i = {f: i for i, f in enumerate(top)}
    examples = [{'sequence': r['sequence'], 'name': r['name'],
                 'label_idx': f2i[r['family']]} for r in raw if r['family'] in f2i]
    modalities = [('sequence', 256), ('name', 32)]
    return examples[:n_target], modalities, top


LOADERS = {
    'codesearchnet': load_codesearchnet,
    'molecule': load_molecule,
    'protein': load_protein,
}
