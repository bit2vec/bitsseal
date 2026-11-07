#!/usr/bin/env python3
"""Tier-A medical-code modality-scaling efficiency curve (CPU-only, no GPU).

Headline efficiency figure for the multimodal+efficiency paper:
fuse ICD-10-CM dx + ICD-10-PCS proc + RxNorm/NDC meds + drug-name text + clinical
text through one shared encoder. Each modality is a controlled terminology with
10K-350K codes, so the Subword baseline's *summed* embedding table grows through
GB scale as modalities are added, while Bits stays at one constant ~33K encoder.

Two grounding levels:
  (1) DEMO   - unique codes actually present in MIMIC-IV demo (100 patients).
               Small, but proves the modalities are real and disjoint.
  (2) DEPLOY - full public terminology sizes (the real deployment vocab).
               This is what determines embedding-table memory in practice; the
               sizes are public facts (CMS / NLM / SNOMED International), so the
               GB-scale claim needs no credentialed data.

Outputs results/medcode_efficiency/summary.json + curve.csv.
"""
import csv, gzip, json, os

DATA = os.path.join(os.path.dirname(__file__), "../../data/medcode_efficiency")
OUT  = os.path.join(os.path.dirname(__file__), "../../results/medcode_efficiency")
os.makedirs(OUT, exist_ok=True)

EMB_DIM_BERT = 768      # typical encoder width
EMB_DIM_LLM  = 4096     # LLM width
BYTES_FP32   = 4
BITS_PARAMS  = 33_000   # constant Bits encoder (byte->random 64-bit code->FFN)


def col(path, name):
    with gzip.open(os.path.join(DATA, path), "rt") as f:
        r = csv.DictReader(f)
        return [row[name] for row in r]


def uniq(vals):
    return len({v for v in vals if v not in ("", "NULL", None)})


# ---- (1) real codes present in the MIMIC-IV demo ----
dx   = col("diagnoses_icd.csv.gz", "icd_code")
proc = col("procedures_icd.csv.gz", "icd_code")
ndc  = col("prescriptions.csv.gz", "ndc")
drug = col("prescriptions.csv.gz", "drug")

demo = [
    ("clinical text (subword)",  3000),          # ~note-subword est. (no notes in demo)
    ("+ ICD dx codes",           uniq(dx)),
    ("+ ICD proc codes",         uniq(proc)),
    ("+ NDC med codes",          uniq(ndc)),
    ("+ drug-name tokens",       uniq(drug)),
]

# ---- (2) full public deployment terminology sizes ----
deploy = [
    ("clinical text (subword)",  50_000),
    ("+ ICD-10-CM dx",           74_000),
    ("+ ICD-10-PCS proc",        78_000),
    ("+ RxNorm meds",            280_000),
    ("+ SNOMED CT",              350_000),
    ("+ NDC drug names",         60_000),
]


def build_curve(mods):
    rows, cum = [], 0
    for i, (m, v) in enumerate(mods, 1):
        cum += v
        rows.append({
            "n_modalities":   i,
            "added":          m,
            "added_vocab":    v,
            "cumulative_vocab": cum,
            "subword_params_d768":  cum * EMB_DIM_BERT,
            "subword_gb_d768":      round(cum * EMB_DIM_BERT * BYTES_FP32 / 1e9, 4),
            "subword_gb_d4096":     round(cum * EMB_DIM_LLM  * BYTES_FP32 / 1e9, 4),
            "bits_params":          BITS_PARAMS,
            "ratio_params_d768":    round(cum * EMB_DIM_BERT / BITS_PARAMS, 1),
        })
    return rows


summary = {
    "description": "Tier-A medical-code modality-scaling efficiency curve (CPU-only).",
    "config": {"emb_dim_bert": EMB_DIM_BERT, "emb_dim_llm": EMB_DIM_LLM,
               "bits_params": BITS_PARAMS, "mimic_source": "mimic-iv-demo v2.2 (100 patients, open)"},
    "demo_curve": build_curve(demo),
    "deploy_curve": build_curve(deploy),
}

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# flat CSV for plotting
with open(os.path.join(OUT, "curve.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["level", "n_modalities", "added", "added_vocab", "cumulative_vocab",
                "subword_gb_d768", "subword_gb_d4096", "bits_params", "ratio_params_d768"])
    for lvl, rows in (("demo", summary["demo_curve"]), ("deploy", summary["deploy_curve"])):
        for r in rows:
            w.writerow([lvl, r["n_modalities"], r["added"], r["added_vocab"],
                        r["cumulative_vocab"], r["subword_gb_d768"], r["subword_gb_d4096"],
                        r["bits_params"], r["ratio_params_d768"]])

# console report
for lvl, rows in (("DEMO (codes present in MIMIC-IV demo)", summary["demo_curve"]),
                  ("DEPLOY (full public terminology sizes)", summary["deploy_curve"])):
    print(f"\n=== {lvl} ===")
    print(f'{"#":>2} {"added":>24} {"+vocab":>9} {"cum":>9} {"Subword GB d768":>16} {"d4096":>9} {"ratio":>9}')
    for r in rows:
        print(f'{r["n_modalities"]:>2} {r["added"]:>24} {r["added_vocab"]:>9,} {r["cumulative_vocab"]:>9,} '
              f'{r["subword_gb_d768"]:>14.3f}GB {r["subword_gb_d4096"]:>7.2f}GB {r["ratio_params_d768"]:>8,.0f}x')
print(f"\nBits encoder constant: {BITS_PARAMS:,} params at every step.")
print(f"Wrote {OUT}/summary.json + curve.csv")
