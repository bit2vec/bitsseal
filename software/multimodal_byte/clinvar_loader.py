"""ClinVar loader (NCBI eutils, no API key needed for low-volume queries).

Pulls ~3000 variants with (clinical_significance, condition, gene, variant_name,
review_status). Multi-modal: text-like fields (condition, review_status) +
gene + variant + label (clinical significance).
"""
import json, os, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'clinvar_cache')
os.makedirs(CACHE, exist_ok=True)

EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


def _cached(url, key):
    fn = os.path.join(CACHE, f'{key}')
    if os.path.exists(fn):
        return open(fn, 'rb').read()
    print(f'  GET {url[:120]}')
    r = urllib.request.urlopen(url, timeout=60)
    data = r.read()
    open(fn, 'wb').write(data)
    return data


def search_clinvar(term, retmax=3000):
    """esearch returns up to retmax UIDs."""
    url = (f'{EUTILS}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(term)}'
           f'&retmax={retmax}&retmode=json')
    data = json.loads(_cached(url, f'esearch_{term[:20].replace(" ","_")}.json'))
    return data['esearchresult']['idlist']


def summary_clinvar(uids, batch=200):
    """esummary returns rich metadata per UID. Returns list of dicts."""
    out = []
    for i in range(0, len(uids), batch):
        chunk = uids[i:i+batch]
        url = (f'{EUTILS}/esummary.fcgi?db=clinvar&id={",".join(chunk)}'
               f'&retmode=json')
        try:
            data = json.loads(_cached(url, f'esum_{i}_{len(chunk)}.json'))
        except Exception as e:
            print(f'  batch {i}: {e}'); continue
        result = data.get('result', {})
        for uid in chunk:
            d = result.get(uid)
            if not d or 'error' in d: continue
            out.append(d)
        time.sleep(0.4)  # be nice to NCBI rate limit
    return out


# Top-5 significance categories
LABEL_SET = ['Pathogenic', 'Likely pathogenic', 'Uncertain significance',
             'Likely benign', 'Benign']


def normalize_significance(s):
    s = (s or '').strip()
    for label in LABEL_SET:
        if label.lower() in s.lower():
            return label
    return None


def extract_fields(record):
    """ClinVar esummary record -> our modality dict + label."""
    germline = record.get('germline_classification', {}) or {}
    sig = normalize_significance(germline.get('description'))
    if sig is None: return None

    review_status = germline.get('review_status', '')
    last_eval = germline.get('last_evaluated', '')
    trait_set = germline.get('trait_set', [])
    conditions = '; '.join(t.get('trait_name', '') for t in trait_set if t.get('trait_name'))

    var_set = record.get('variation_set', [])
    if not var_set: return None
    var = var_set[0]
    variant_name = var.get('variation_name', '') or var.get('cdna_change', '')
    var_type = var.get('variant_type', '')
    genes = var.get('genes', [])
    gene_symbol = genes[0].get('symbol') if genes else ''
    canonical_spdi = var.get('canonical_spdi', '')

    return {
        'clinvar_id':       record.get('uid', ''),
        'condition':        conditions,
        'gene':             gene_symbol,
        'variant':          variant_name,
        'variant_type':     var_type,
        'review_status':    review_status,
        'last_evaluated':   last_eval,
        'canonical_spdi':   canonical_spdi,
        'label':            sig,
    }


def load_clinvar(term='cancer[disease]', n_target=3000):
    uids = search_clinvar(term, retmax=n_target)
    print(f'  ClinVar {term}: {len(uids)} UIDs returned')
    records = summary_clinvar(uids)
    examples = []
    for rec in records:
        ex = extract_fields(rec)
        if ex is not None:
            examples.append(ex)
    print(f'  Parsed {len(examples)} examples')
    return examples, LABEL_SET


if __name__ == '__main__':
    import random
    examples, labels = load_clinvar(n_target=3000)
    random.seed(0); random.shuffle(examples)
    print('\nExample 0:')
    for k, v in examples[0].items(): print(f'  {k}: {v[:120] if isinstance(v, str) else v}')
    from collections import Counter
    print('\nLabel distribution:', Counter(e['label'] for e in examples))
