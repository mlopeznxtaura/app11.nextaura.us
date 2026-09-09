"""FineWeb pretrain for ~50M GPT.

5090 (32 GB) Tuesday:
  source .gpu.env
  TRAIN_DEVICE=cuda TRAIN_BATCH_SIZE=128 MAX_STEPS=20000 TARGET_TOKENS=1000000000 \\
    python train_gpu.py
Stop on MAX_STEPS or TARGET_TOKENS (not 10 GB). COS save at end only.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone

import ibm_boto3
from ibm_botocore.client import Config

from train_engine import TrainEngine, should_save_model

DATASET_ID = os.environ.get("DATASET_ID", "HuggingFaceFW/fineweb")
CONFIG = os.environ.get("FINEWEB_CONFIG", "sample-10BT")
TARGET_GB = float(os.environ.get("TARGET_GB", "0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20000"))
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "1000000000"))
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
IBM_API_KEY = os.environ.get("IBM_CLOUD_API_KEY", "").strip()
COS_CRN = os.environ.get(
    "COS_CRN",
    "crn:v1:bluemix:public:cloud-object-storage:global:a/ee54102c4e17411fa08552596d94e53d:49c90492-ab55-4cba-90f4-589623751191::",
)
COS_BUCKET = os.environ.get("COS_BUCKET", "nextaura-fineweb-stage1")
COS_ENDPOINT = os.environ.get(
    "COS_ENDPOINT",
    "https://s3.us-south.cloud-object-storage.appdomain.cloud",
)
CHECKPOINT_KEY = os.environ.get("COS_CHECKPOINT_KEY", "stage1/checkpoint.json")
MODEL_KEY = os.environ.get("COS_MODEL_KEY", "stage1/model/latest.pt")
ENGLISH_ONLY = os.environ.get("ENGLISH_ONLY", "1") != "0"
MIN_TOKEN_COUNT = int(os.environ.get("MIN_TOKEN_COUNT", "32"))
HOST = "app2.nextaura.us"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cos_client():
    if not IBM_API_KEY:
        raise SystemExit("IBM_CLOUD_API_KEY missing")
    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=IBM_API_KEY,
        ibm_service_instance_id=COS_CRN,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT,
    )


def accept_row(row: dict) -> bool:
    text = row.get("text")
    if not isinstance(text, str) or len(text.strip()) < 80:
        return False
    if ENGLISH_ONLY:
        lang = str(row.get("language") or "").lower()
        score = row.get("language_score")
        if lang and lang != "en":
            return False
        if isinstance(score, (int, float)) and score < 0.85:
            return False
    tok = row.get("token_count")
    if isinstance(tok, int) and tok < MIN_TOKEN_COUNT:
        return False
    return True


def fmt_consumed(n_bytes: int) -> str:
    mb = n_bytes / (1024**2)
    if TARGET_GB > 0:
        gb = n_bytes / (1024**3)
        if gb < 0.01:
            return f"{mb:.2f} MB / {TARGET_GB:g} GB"
        return f"{gb:.3f}/{TARGET_GB:g} GB"
    return f"{mb:.2f} MB"


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN missing")
    from datasets import load_dataset
    import torch

    device = pick_device()
    os.environ.setdefault("TRAIN_BATCH_SIZE", "128" if device == "cuda" else ("32" if device == "mps" else "2"))
    # Re-import batch size after env set: engine reads BATCH_SIZE at import time.
    # Recreate engine module values by constructing after setting env — import already happened.
    # Override via TrainEngine internals: feed uses module BATCH_SIZE. Set before first use:
    import train_engine as te

    te.BATCH_SIZE = int(os.environ["TRAIN_BATCH_SIZE"])

    eng = TrainEngine(device=device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cos = cos_client()
    try:
        cos.head_bucket(Bucket=COS_BUCKET)
    except Exception:
        cos.create_bucket(Bucket=COS_BUCKET)

    target_bytes = int(TARGET_GB * (1024**3)) if TARGET_GB > 0 else 0
    rows = skipped = bytes_written = stream_index = 0
    print(
        f"[CYAN] pretrain device={device} cuda={torch.cuda.is_available()} "
        f"mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()} "
        f"params={eng.param_count:,} batch={eng.batch_size} "
        f"max_steps={MAX_STEPS} target_tokens={TARGET_TOKENS:,} target_gb={TARGET_GB:g}",
        flush=True,
    )

    stream = load_dataset(
        DATASET_ID,
        name=CONFIG,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    def snapshot(status: str) -> dict:
        return {
            "version": 2,
            "status": status,
            "config": CONFIG,
            "target_gb": TARGET_GB,
            "model_params": eng.param_count,
            "rows": rows,
            "skipped": skipped,
            "bytes_written": bytes_written,
            "stream_index": stream_index,
            "train_step": eng.step,
            "last_loss": eng.last_loss,
            "last_tok_s": 0.0,
            "model_ready": eng.step > 0,
            "model_key": MODEL_KEY,
            "dataset": DATASET_ID,
            "host": HOST,
            "device": device,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_all(status: str, force_model: bool = False) -> None:
        if force_model or should_save_model(eng.step):
            if eng.step > 0:
                cos.put_object(
                    Bucket=COS_BUCKET,
                    Key=MODEL_KEY,
                    Body=eng.state_bytes(),
                    ContentType="application/octet-stream",
                )
                print(f"[CYAN] model checkpoint saved step {eng.step}", flush=True)
        cos.put_object(
            Bucket=COS_BUCKET,
            Key=CHECKPOINT_KEY,
            Body=json.dumps(snapshot(status), indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    try:
        for row in stream:
            if MAX_STEPS and eng.step >= MAX_STEPS:
                break
            if TARGET_TOKENS and eng.total_tokens >= TARGET_TOKENS:
                break
            if target_bytes and bytes_written >= target_bytes:
                break
            stream_index += 1
            if not accept_row(row):
                skipped += 1
                continue
            text = str(row.get("text") or "").strip()
            text_bytes = len(text.encode("utf-8"))
            if target_bytes and bytes_written + text_bytes > target_bytes:
                break
            steps = eng.feed_text(text)
            bytes_written += text_bytes
            rows += 1
            for st in steps:
                if st.step == 1 or st.step % 50 == 0:
                    print(
                        f"[GREEN] step {st.step:,} loss {st.loss:.4f} · "
                        f"{fmt_consumed(bytes_written)} · {rows:,} docs · {st.tok_s:.0f} tok/s",
                        flush=True,
                    )
                if MAX_STEPS and st.step >= MAX_STEPS:
                    break
            if MAX_STEPS and eng.step >= MAX_STEPS:
                break
            if TARGET_TOKENS and eng.total_tokens >= TARGET_TOKENS:
                break
        save_all("complete", force_model=True)
        print(
            f"[GREEN] complete — step {eng.step:,} · {fmt_consumed(bytes_written)} · {device}",
            flush=True,
        )
    except Exception as exc:
        print(f"[RED] error: {exc}", flush=True)
        print(traceback.format_exc()[-800:], flush=True)
        try:
            save_all("interrupted", force_model=True)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
