import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
from huggingface_hub import HfApi
arm=sys.argv[1]; tok=sys.stdin.readline().strip()
api=HfApi(token=tok); repo="bit2vec/bitsseal"; base="/root/traj/s3_1B"
for attempt in range(8):
    try:
        api.upload_folder(folder_path=f"{base}/{arm}", repo_id=repo, path_in_repo=f"qwen2.5-7b_c4_1B/{arm}",
            allow_patterns=["checkpoint-*/*.safetensors","checkpoint-*/*.json","checkpoint-*/bf16.log"], commit_message=f"{arm}: all checkpoints")
        print("ARM_UPLOADED", arm, time.strftime("%H:%M:%S"), flush=True); break
    except Exception as e:
        print("RETRY", arm, attempt, repr(e)[:300], flush=True); time.sleep(300)
else:
    print("GAVEUP", arm, flush=True)
if arm=="s42_bits":
    api.upload_folder(folder_path=base, repo_id=repo, path_in_repo="qwen2.5-7b_c4_1B/logs", allow_patterns=["*.log"], ignore_patterns=["**/checkpoint-*/**","**/runs/**"], commit_message="logs")
    api.upload_folder(folder_path="/root", repo_id=repo, path_in_repo="qwen2.5-7b_c4_1B/scripts", allow_patterns=["bits_llm.py","bits_lm_traj.py","eval_ppl.py","eval_seed.sh","run_3seeds_1B.sh","to_bf16.py","hf_arm.py","heldout_c4val_4000x512.pt"], commit_message="scripts")
    print("LOGS_SCRIPTS_DONE", flush=True)
