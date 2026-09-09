"""UpCloud Managed Object Storage (S3-compatible) for checkpoint archive."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

UPCLOUD_S3_ENDPOINT = os.environ.get("UPCLOUD_S3_ENDPOINT", "https://hei5u.upcloudobjects.com")
UPCLOUD_S3_BUCKET = os.environ.get("UPCLOUD_S3_BUCKET", "nextaura-models")
UPCLOUD_S3_PREFIX = os.environ.get("UPCLOUD_S3_PREFIX", "app7/checkpoints/").rstrip("/") + "/"
UPCLOUD_S3_REGION = os.environ.get("UPCLOUD_S3_REGION", "us-1")
UPCLOUD_S3_VERIFY_SSL = os.environ.get("UPCLOUD_S3_VERIFY_SSL", "false").strip().lower() in ("1", "true", "yes")


def upcloud_configured() -> bool:
    return bool(
        os.environ.get("UPCLOUD_S3_ACCESS_KEY", "").strip()
        and os.environ.get("UPCLOUD_S3_SECRET_KEY", "").strip()
    )


def _upcloud_client():
    if not upcloud_configured():
        return None
    import boto3
    from botocore.client import Config

    if not UPCLOUD_S3_VERIFY_SSL:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    return boto3.client(
        "s3",
        endpoint_url=UPCLOUD_S3_ENDPOINT,
        aws_access_key_id=os.environ["UPCLOUD_S3_ACCESS_KEY"].strip(),
        aws_secret_access_key=os.environ["UPCLOUD_S3_SECRET_KEY"].strip(),
        region_name=UPCLOUD_S3_REGION,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path", "payload_signing_enabled": False},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
        verify=UPCLOUD_S3_VERIFY_SSL,
    )


def upcloud_key_for_checkpoint(name: str) -> str:
    base = os.path.basename(name.strip())
    return f"{UPCLOUD_S3_PREFIX}{base}"


def upload_checkpoint_file(local_path: str, *, name: str | None = None) -> dict[str, Any]:
    """Upload a local .pt file to UpCloud MOS."""
    client = _upcloud_client()
    if not client:
        return {"ok": False, "error": "UpCloud S3 not configured"}
    filename = name or os.path.basename(local_path)
    key = upcloud_key_for_checkpoint(filename)
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as fh:
        client.put_object(
            Bucket=UPCLOUD_S3_BUCKET,
            Key=key,
            Body=fh,
            ContentType="application/octet-stream",
        )
    return {
        "ok": True,
        "bucket": UPCLOUD_S3_BUCKET,
        "key": key,
        "name": filename,
        "size_bytes": size,
        "endpoint": UPCLOUD_S3_ENDPOINT,
    }


def download_checkpoint_bytes(key: str | None = None, *, name: str | None = None) -> bytes:
    client = _upcloud_client()
    if not client:
        raise RuntimeError("UpCloud S3 not configured")
    obj_key = key or upcloud_key_for_checkpoint(name or "")
    resp = client.get_object(Bucket=UPCLOUD_S3_BUCKET, Key=obj_key)
    return resp["Body"].read()


def list_remote_checkpoints() -> list[dict[str, Any]]:
    client = _upcloud_client()
    if not client:
        return []
    rows: list[dict[str, Any]] = []
    try:
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": UPCLOUD_S3_BUCKET, "Prefix": UPCLOUD_S3_PREFIX}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents") or []:
                key = obj.get("Key") or ""
                if not key.endswith(".pt"):
                    continue
                name = os.path.basename(key)
                rows.append(
                    {
                        "name": name,
                        "key": key,
                        "source": "upcloud",
                        "size_mb": round(int(obj.get("Size") or 0) / (1024 * 1024), 2),
                        "modified": obj.get("LastModified"),
                    }
                )
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    except Exception:
        return []
    rows.sort(key=lambda r: r["name"], reverse=True)
    return rows


def storage_status() -> dict[str, Any]:
    return {
        "upcloud_configured": upcloud_configured(),
        "upcloud_endpoint": UPCLOUD_S3_ENDPOINT if upcloud_configured() else None,
        "upcloud_bucket": UPCLOUD_S3_BUCKET if upcloud_configured() else None,
        "upcloud_prefix": UPCLOUD_S3_PREFIX if upcloud_configured() else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
