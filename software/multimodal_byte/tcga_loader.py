"""TCGA pan-cancer 5-class loader via cBioPortal API.

Pulls clinical metadata + mRNA expression for BRCA / KIRC / LUAD / COADREAD /
UCEC PanCancer Atlas studies. Produces per-patient (text, structured,
gene_top_K) tuples + cancer-type label.

Caches the JSON pulls to disk. First run takes ~5-10 min for the API queries;
subsequent runs use the cache.
"""
import json, os, urllib.request, urllib.parse
from collections import defaultdict, Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(THIS_DIR, 'tcga_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

STUDIES = {
    'BRCA':     'brca_tcga_pan_can_atlas_2018',
    'KIRC':     'kirc_tcga_pan_can_atlas_2018',
    'LUAD':     'luad_tcga_pan_can_atlas_2018',
    'COADREAD': 'coadread_tcga_pan_can_atlas_2018',
    'UCEC':     'ucec_tcga_pan_can_atlas_2018',
}

# Clinical attributes to use as text vs structured
CLINICAL_TEXT_ATTRS = [
    'ICD_10', 'ICD_O_3_HISTOLOGY', 'ICD_O_3_SITE',
    'PRIMARY_SITE_PATIENT', 'TUMOR_TYPE', 'TUMOR_TISSUE_SITE',
    'HISTOLOGICAL_DIAGNOSIS', 'SUBTYPE',
]
CLINICAL_STRUCT_ATTRS = [
    'AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'GRADE',
    'ETHNICITY', 'RACE', 'GENETIC_ANCESTRY_LABEL',
    'SMOKING_HISTORY', 'PRIOR_DX', 'HISTORY_NEOADJUVANT_TRTYN',
    'OS_STATUS', 'DFS_STATUS',
    'TMB_NONSYNONYMOUS', 'BUFFA_HYPOXIA_SCORE',
    'MSI_SCORE_MANTIS', 'ANEUPLOIDY_SCORE',
]

# Ranking of genes is more meaningful for some genes; pull a fixed set known to vary across cancer types.
# Use a stable list of ~200 cancer-relevant genes. Pulling all 20k genes via API is too slow.
CANCER_GENES = [
    "TP53","KRAS","EGFR","BRCA1","BRCA2","PIK3CA","BRAF","MYC","RB1","PTEN",
    "APC","SMAD4","NRAS","ATM","ARID1A","CDKN2A","ERBB2","CTNNB1","FGFR2","FGFR3",
    "JAK2","NF1","NF2","KIT","ALK","RET","ROS1","MET","NOTCH1","NOTCH2",
    "MTOR","STK11","CDH1","VHL","BAP1","FBXW7","SMARCB1","TET2","DNMT3A","FLT3",
    "IDH1","IDH2","HIST1H3B","H3F3A","KMT2D","KMT2C","SETD2","ARID2","ATRX","PBRM1",
    "BAP1","KDM6A","ARID1B","CREBBP","EP300","BCL2","BCL6","CCND1","CDK4","CDK6",
    "CCNE1","MDM2","MDM4","FOXA1","FOXP1","FOXM1","ESR1","AR","HOXA9","HOXB13",
    "MYCN","MYCL","WNT5A","CTLA4","PDCD1","PDL1","CD274","VEGFA","HIF1A","TGFB1",
    "TGFBR2","TLR4","STAT3","STAT5A","NRF2","KEAP1","NFKB1","RELA","TRAF6","TNF",
    "IL6","IL10","IL8","IFNG","FAS","CASP3","CASP8","CASP9","BAX","BCL2L1",
    "ATG5","ATG7","BECN1","HMGB1","SIRT1","SIRT3","FOXO1","FOXO3","FOXO4","SOX2",
    "SOX9","SOX10","NANOG","POU5F1","KLF4","LIN28","TET1","TET3","WT1","PAX5",
    "PAX8","SF3B1","U2AF1","SRSF2","ZRSR2","DDX3X","CIC","FUBP1","DAXX","SMARCA4",
    "ARID4A","ARID4B","HRAS","RHOA","RAC1","CDC42","ROCK1","ROCK2","MYLK","ITGB1",
    "MMP2","MMP9","TIMP1","TIMP2","COL1A1","COL3A1","FN1","VIM","CDH2","SNAI1",
    "SNAI2","ZEB1","ZEB2","TWIST1","TWIST2","BMI1","RUNX1","RUNX2","RUNX3","TBX1",
    "TBX2","SHH","SMO","GLI1","GLI2","PTCH1","SUFU","NOTCH3","DLL4","JAG1",
    "HES1","HEY1","NUMB","DTX1","WNT1","WNT3","WNT3A","CTNNB1","AXIN1","AXIN2",
    "GSK3B","DKK1","SFRP1","RSPO2","LGR5","BMP2","BMP4","BMPR1A","BMPR2","SMAD2",
    "SMAD3","SMAD7","TGFA","INSR","IGF1R","IGF2","IGFBP3","FGF19","FGF21","FGF23",
]


def _cached_get(url, key):
    fn = os.path.join(CACHE_DIR, f'{key}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    print(f'  GET {url[:120]}')
    r = urllib.request.urlopen(url, timeout=60)
    data = json.loads(r.read())
    with open(fn, 'w') as f: json.dump(data, f)
    return data


def _post_json(url, payload, key):
    fn = os.path.join(CACHE_DIR, f'{key}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    print(f'  POST {url[:120]}')
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, method='POST',
                                  headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=120)
    data = json.loads(r.read())
    with open(fn, 'w') as f: json.dump(data, f)
    return data


def fetch_clinical(study_id):
    url = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType=PATIENT'
    return _cached_get(url, f'clin_{study_id}')


def fetch_mrna(study_id, hugo_genes=CANCER_GENES, max_patients=None):
    """Pull mRNA RPKM/RSEM expression for a fixed gene panel via the molecular-data endpoint."""
    # First find the mRNA molecular profile id
    url = f'https://www.cbioportal.org/api/studies/{study_id}/molecular-profiles'
    profs = _cached_get(url, f'profs_{study_id}')
    mrna_prof = None
    for p in profs:
        if p.get('molecularAlterationType') == 'MRNA_EXPRESSION' \
           and 'rna_seq_v2' in p.get('molecularProfileId', ''):
            mrna_prof = p['molecularProfileId']
            break
    if mrna_prof is None:
        for p in profs:
            if p.get('molecularAlterationType') == 'MRNA_EXPRESSION':
                mrna_prof = p['molecularProfileId']; break
    if mrna_prof is None:
        return {}, mrna_prof

    # Find a sample list
    list_url = f'https://www.cbioportal.org/api/studies/{study_id}/sample-lists'
    lists = _cached_get(list_url, f'lists_{study_id}')
    sample_list_id = None
    for sl in lists:
        if sl.get('sampleListId', '').endswith('_rna_seq_v2_mrna'):
            sample_list_id = sl['sampleListId']; break
    if sample_list_id is None:
        for sl in lists:
            if 'mrna' in sl.get('sampleListId', ''):
                sample_list_id = sl['sampleListId']; break

    # Look up Entrez IDs for the panel
    gene_url = 'https://www.cbioportal.org/api/genes/fetch?geneIdType=HUGO_GENE_SYMBOL'
    gene_data = _post_json(gene_url, hugo_genes, f'genes_panel')
    entrez = [g['entrezGeneId'] for g in gene_data]

    # Pull molecular data (sampleListId goes in the body, not the URL)
    mol_url = (f'https://www.cbioportal.org/api/molecular-profiles/{mrna_prof}/'
               f'molecular-data/fetch')
    body = {'sampleListId': sample_list_id, 'entrezGeneIds': entrez}
    mol_data = _post_json(mol_url, body, f'mrna_{study_id}')

    # Reshape: per-patient dict of {gene: value}
    by_patient = defaultdict(dict)
    entrez_to_hugo = {g['entrezGeneId']: g['hugoGeneSymbol'] for g in gene_data}
    for r in mol_data:
        pid = r['patientId']
        gene = entrez_to_hugo.get(r['entrezGeneId'])
        try:
            by_patient[pid][gene] = float(r['value'])
        except Exception:
            pass
    return dict(by_patient), mrna_prof


def patients_to_examples(study_label, study_id, top_k_genes=64,
                         text_attrs=CLINICAL_TEXT_ATTRS,
                         struct_attrs=CLINICAL_STRUCT_ATTRS):
    """Returns list of dicts: {'text', 'structured', 'gene_top_k', 'label'}."""
    clinical = fetch_clinical(study_id)
    by_patient = defaultdict(dict)
    for d in clinical:
        by_patient[d['patientId']][d['clinicalAttributeId']] = d['value']
    mrna, _ = fetch_mrna(study_id)

    examples = []
    for pid, attrs in by_patient.items():
        text_parts = []
        for a in text_attrs:
            v = attrs.get(a, '').strip()
            if v: text_parts.append(f'{a.lower()}:{v}')
        text = ' '.join(text_parts) if text_parts else f'study:{study_label}'

        struct_parts = []
        for a in struct_attrs:
            v = attrs.get(a, '').strip()
            if v:
                # Bin numeric values
                try:
                    fv = float(v)
                    if a == 'AGE': v_bin = f'{int(fv//10)*10}'
                    elif 'SCORE' in a: v_bin = f'{round(fv, 1)}'
                    else: v_bin = f'{fv:.2f}'
                    struct_parts.append(f'{a.lower()}={v_bin}')
                except ValueError:
                    struct_parts.append(f'{a.lower()}={v.replace(" ", "_")[:30]}')
        structured = ' '.join(struct_parts)

        # Top-K expressed genes
        if pid in mrna:
            ranked = sorted(((g, v) for g, v in mrna[pid].items() if v == v),
                            key=lambda x: x[1], reverse=True)[:top_k_genes]
            gene_top = ' '.join(g for g, _ in ranked)
        else:
            gene_top = ''

        examples.append({
            'text':       text,
            'structured': structured,
            'gene_top':   gene_top,
            'label':      study_label,
        })
    return examples


def load_tcga_pan5(top_k_genes=64):
    """Returns: list of (text, structured, gene_top, label) examples + label list."""
    all_examples = []
    for label, study in STUDIES.items():
        ex = patients_to_examples(label, study, top_k_genes=top_k_genes)
        print(f'  {label}: {len(ex)} patients')
        all_examples.extend(ex)
    labels_list = list(STUDIES.keys())
    return all_examples, labels_list


if __name__ == '__main__':
    import random
    examples, labels = load_tcga_pan5()
    print(f'\nTotal: {len(examples)} patients across {len(labels)} types')
    random.seed(0); random.shuffle(examples)
    print('\nExample 0:')
    for k in ['text', 'structured', 'gene_top', 'label']:
        v = examples[0][k]
        print(f'  {k}: {v[:200]}{"..." if len(v) > 200 else ""}')
