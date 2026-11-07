"""Byte-level classification loaders. Each returns (rows, fields, label_names):
  rows        list[dict]  field-name -> text, plus 'label_idx'
  fields      list[str]   which fields to (byte-)encode, in order
  label_names list[str]
"""
import random


def load_code_langid(n_target=20000, seed=42):
    """Programming-language identification from source code (6-class).
    High-cardinality identifier vocab -> BPE table grows; byte/binary constant."""
    from datasets import load_dataset
    langs = ['python', 'java', 'go', 'javascript', 'php', 'ruby']
    per = max(200, n_target // len(langs))
    rows = []
    for li, lg in enumerate(langs):
        ds = None
        for name, kw in [('claudios/code_search_net', dict(name=lg, trust_remote_code=True)),
                         ('code_search_net', dict(name=lg, trust_remote_code=True))]:
            try:
                ds = load_dataset(name, split='train', streaming=True, **kw); break
            except Exception:
                continue
        if ds is None:
            continue
        n = 0
        for ex in ds:
            code = ex.get('func_code_string') or ex.get('whole_func_string') or ex.get('code') or ''
            if not code.strip():
                continue
            rows.append({'code': code, 'label_idx': li})
            n += 1
            if n >= per: break
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], ['code'], langs


def load_xnli(n_target=20000, seed=42):
    """Multilingual NLI (15 langs): premise + hypothesis -> entailment (3-class).
    Byte-level handles all 15 languages with one fixed encoder."""
    from datasets import load_dataset
    rows = []
    ds = None
    for name in ['facebook/xnli', 'xnli']:
        try:
            ds = load_dataset(name, 'all_languages', split='validation'); break
        except Exception:
            try:
                ds = load_dataset(name, 'all_languages', split='train', streaming=True); break
            except Exception:
                continue
    if ds is None:
        raise RuntimeError('XNLI not loadable')
    for ex in ds:
        # all_languages: premise is dict lang->text, hypothesis has language+translation lists
        prem = ex['premise']; hyp = ex['hypothesis']
        if isinstance(prem, dict):
            langs = list(prem.keys())
            hmap = {l: t for l, t in zip(hyp['language'], hyp['translation'])}
            for l in langs:
                if l in hmap:
                    rows.append({'premise': prem[l], 'hypothesis': hmap[l],
                                 'label_idx': int(ex['label'])})
        else:
            rows.append({'premise': prem, 'hypothesis': hyp, 'label_idx': int(ex['label'])})
        if len(rows) >= n_target:
            break
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], ['premise', 'hypothesis'], ['entailment', 'neutral', 'contradiction']


def load_dna_text(n_target=20000, seed=42, n_classes=12):
    """DNA + text fusion: gene description (English) + RefSeq RNA nucleotide
    sequence -> GO Biological-Process category (non-leaky label, not stated in
    the description). Demonstrates byte-level fusion of a natural language and a
    nucleotide language in ONE vocabulary-independent encoder.

    Data (NCBI, prebuilt under data/dna_text/):
      human.*.rna.fna.gz       RefSeq RNA: header (desc + (SYMBOL)) + sequence
      Homo_sapiens.gene_info.gz  Symbol -> GeneID
      gene2go.gz               GeneID -> GO terms (filter tax 9606, Process)
    """
    import csv, glob, gzip, os, re
    from collections import Counter
    D = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dna_text')

    # Symbol -> GeneID (human)
    sym2gid = {}
    with gzip.open(os.path.join(D, 'Homo_sapiens.gene_info.gz'), 'rt') as f:
        r = csv.reader(f, delimiter='\t')
        next(r, None)
        for row in r:
            if len(row) > 9:
                sym2gid[row[2]] = row[1]

    # GeneID -> a single GO Biological-Process term (most frequent overall)
    gid2bp = {}
    bp_counter = Counter()
    with gzip.open(os.path.join(D, 'gene2go.gz'), 'rt') as f:
        r = csv.reader(f, delimiter='\t')
        next(r, None)
        for row in r:
            if len(row) < 8 or row[0] != '9606':
                continue
            if row[7] == 'Process':
                gid2bp.setdefault(row[1], []).append(row[5])  # GO_term
                bp_counter[row[5]] += 1
    top = {t for t, _ in bp_counter.most_common(n_classes)}
    labels = sorted(top)
    lab2idx = {t: i for i, t in enumerate(labels)}

    def gene_go_label(gid):
        for t in gid2bp.get(gid, []):
            if t in top:
                return t
        return None

    biotype_re = re.compile(r',\s*(mRNA|.*?RNA|.*?non-coding.*|transcript variant.*)$', re.I)
    sym_re = re.compile(r'\(([^)]+)\)[^()]*$')
    rows = []
    for fa in sorted(glob.glob(os.path.join(D, 'human.*.rna.fna.gz'))):
        with gzip.open(fa, 'rt') as f:
            header, seq = None, []
            def flush(header, seq):
                if not header:
                    return
                m = sym_re.search(header)
                if not m:
                    return
                sym = m.group(1)
                gid = sym2gid.get(sym)
                lab = gene_go_label(gid) if gid else None
                if lab is None:
                    return
                # description = header minus accession, minus trailing biotype (anti-leak)
                desc = header.split(' ', 1)[1] if ' ' in header else header
                desc = biotype_re.sub('', desc).replace('Homo sapiens', '').strip()
                rows.append({'text': desc, 'sequence': ''.join(seq)[:600],
                             'label_idx': lab2idx[lab]})
            for line in f:
                if line.startswith('>'):
                    flush(header, seq)
                    header = line[1:].strip(); seq = []
                else:
                    seq.append(line.strip())
            flush(header, seq)
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], ['text', 'sequence'], labels


def load_wili(n_target=40000, seed=42):
    """A2 — WiLI-2018 language identification, 235 languages / scripts.
    Single text field; huge multilingual vocab for a Subword baseline, constant
    for byte/Bits. label = language (235-class)."""
    from datasets import load_dataset
    ds = load_dataset('MartinThoma/wili_2018', split='train')
    rows = [{'text': ex['sentence'], '_lab': ex['label']} for ex in ds]
    labs = sorted({r['_lab'] for r in rows})
    l2i = {l: i for i, l in enumerate(labs)}
    for r in rows:
        r['label_idx'] = l2i[r['_lab']]
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], ['text'], [str(l) for l in labs]


def load_mimic_fusion(n_target=20000, seed=42):
    """A3 — MIMIC-IV demo medical-code fusion (real codes, no credentialing).
    Per admission fuse ICD dx + ICD proc + drug names -> predict admission_type.
    Demonstrates byte-level fusion of multiple high-cardinality code systems."""
    import csv, gzip, os
    from collections import defaultdict, Counter
    D = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'medcode_efficiency')

    def col(fn):
        with gzip.open(os.path.join(D, fn), 'rt') as f:
            return list(csv.DictReader(f))

    dx, proc, rx, adm = (col('diagnoses_icd.csv.gz'), col('procedures_icd.csv.gz'),
                         col('prescriptions.csv.gz'), col('admissions.csv.gz'))
    dxm, prm, rxm = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in dx:   dxm[r['hadm_id']].append(r['icd_code'])
    for r in proc: prm[r['hadm_id']].append(r['icd_code'])
    for r in rx:
        if r.get('drug'): rxm[r['hadm_id']].append(r['drug'].strip())
    rows = []
    for a in adm:
        h = a['hadm_id']
        rows.append({'dx': ' '.join(dxm.get(h, [])),
                     'proc': ' '.join(prm.get(h, [])),
                     'drug': ' '.join(sorted(set(rxm.get(h, [])))[:40]),
                     '_lab': a['admission_type']})
    labs = [l for l, _ in Counter(r['_lab'] for r in rows).most_common()]
    l2i = {l: i for i, l in enumerate(labs)}
    for r in rows:
        r['label_idx'] = l2i[r['_lab']]
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], ['dx', 'proc', 'drug'], labs


def _load_code_stack(level, n_target=20000, seed=42):
    """Parallel POJ-104 corpus at one compilation level (src / asm / bin).
    SAME programs + SAME labels across the three levels; only the byte
    representation differs -- the one-axis control for testing whether the
    vocabulary-independent (byte/binary) advantage over BPE grows as you descend
    the compilation stack (source -> assembly -> machine-code bytes). The corpus
    is prebuilt by code_stack_probe/build_corpus.py (compiled with g++ on alien).
    For 'bin', the field holds the .text machine-code bytes latin-1-decoded 1:1,
    so BPE tokenizes the identical byte stream that byte_seb/binary_seb consume."""
    import json, os
    path = os.path.join(os.path.dirname(__file__), 'code_stack_probe', 'corpus.json')
    with open(path) as f:
        data = json.load(f)
    rows = [{level: p[level], 'label_idx': p['label_idx']} for p in data['programs']]
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:n_target], [level], data['label_names']


def load_code_src(n_target=20000, seed=42):
    return _load_code_stack('src', n_target, seed)


def load_code_asm(n_target=20000, seed=42):
    return _load_code_stack('asm', n_target, seed)


def load_code_bin(n_target=20000, seed=42):
    return _load_code_stack('bin', n_target, seed)


LOADERS = {
    'code_src': load_code_src,
    'code_asm': load_code_asm,
    'code_bin': load_code_bin,
    'code_langid': load_code_langid,
    'xnli': load_xnli,
    'dna_text': load_dna_text,
    'wili': load_wili,
    'mimic_fusion': load_mimic_fusion,
}
