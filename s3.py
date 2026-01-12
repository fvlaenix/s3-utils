from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

_PROPERTIES_ENCODING = "utf-8"


@dataclass
class S3Config:
    """S3 connection settings."""

    bucket: str
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


def _parse_properties(path: str) -> Dict[str, str]:
    props: Dict[str, str] = {}
    with open(path, "r", encoding=_PROPERTIES_ENCODING) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            props[key.strip()] = value.strip()
    return props


def load_s3_config(path: Optional[str] = None) -> S3Config:
    """
    Load S3Config from a .properties file.

    Resolution order:
    1. Explicit path argument, if provided.
    2. S3_PROPERTIES env var.
    3. 's3.properties' in the current working directory.
    """
    if path is None:
        path = os.environ.get("S3_PROPERTIES", "s3.properties")
    if not os.path.exists(path):
        raise FileNotFoundError(f"S3 properties file not found: {path}")

    props = _parse_properties(path)

    bucket = props.get("s3.bucket") or props.get("bucket")
    if not bucket:
        raise RuntimeError("S3 properties must contain s3.bucket")

    region = props.get("s3.region") or props.get("region")
    access_key = props.get("s3.accessKey") or props.get("accessKey")
    secret_key = props.get("s3.secretKey") or props.get("secretKey")

    return S3Config(
        bucket=bucket,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
    )


def create_s3_client(cfg: S3Config):
    """
    Create a boto3 S3 client from S3Config.

    If access_key / secret_key are not provided in the config, standard
    AWS credential resolution is used (env vars, shared credentials file, etc.).
    """
    boto_config = BotoConfig(
        signature_version="s3v4",
        max_pool_connections=16,
    )

    session_kwargs = {}
    client_kwargs = {}

    if cfg.region:
        client_kwargs["region_name"] = cfg.region

    if cfg.access_key and cfg.secret_key:
        session_kwargs["aws_access_key_id"] = cfg.access_key
        session_kwargs["aws_secret_access_key"] = cfg.secret_key

    session = boto3.Session(**session_kwargs)
    return session.client(
        "s3",
        config=boto_config,
        **client_kwargs,
    )


class _ProgressBar:
    def __init__(self, desc: str, total: Optional[int]) -> None:
        self._bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)
        self._total = total
        self._seen_so_far = 0

    def __call__(self, bytes_amount: int) -> None:
        self._seen_so_far += bytes_amount
        self._bar.update(bytes_amount)
        if self._total is not None and self._seen_so_far >= self._total:
            self._bar.close()

    def close(self) -> None:
        self._bar.close()


def _zip_directory(src_dir: str) -> str:
    src_dir = os.path.abspath(src_dir)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        archive_path = tmp.name

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, start=src_dir)
                zf.write(full_path, rel_path)

    return archive_path


def _upload_file(
    s3_client,
    bucket: str,
    local_file: str,
    s3_key: str,
) -> bool:
    total_size = os.path.getsize(local_file)
    progress = _ProgressBar(desc=local_file, total=total_size)
    try:
        s3_client.upload_file(local_file, bucket, s3_key, Callback=progress)
        return True
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] Failed to upload {local_file} -> s3://{bucket}/{s3_key}: {exc}")
        return False
    finally:
        progress.close()


def upload_to_s3(cfg: S3Config, local_path: str, s3_path: str) -> bool:
    """
    Upload a file or directory to S3.

    If local_path is a directory, it is zipped before upload and s3_path
    must end with .zip.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path not found: {local_path}")

    s3_client = create_s3_client(cfg)

    if os.path.isdir(local_path):
        if not s3_path.endswith(".zip"):
            raise ValueError("Uploading a directory requires s3_path to end with .zip")
        archive_path = _zip_directory(local_path)
        try:
            return _upload_file(s3_client, cfg.bucket, archive_path, s3_path)
        finally:
            try:
                os.remove(archive_path)
            except OSError:
                pass

    return _upload_file(s3_client, cfg.bucket, local_path, s3_path)


def _get_object_size(s3_client, bucket: str, key: str) -> Optional[int]:
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        return resp.get("ContentLength")
    except (BotoCoreError, ClientError):
        return None


def download_from_s3(cfg: S3Config, s3_path: str, local_dir: str) -> bool:
    """
    Download an S3 object into a local directory.

    If the downloaded object is a zip, it is extracted into local_dir
    and the zip file is removed.
    """
    if not s3_path or s3_path.endswith("/"):
        raise ValueError("s3_path must be an object key, not a prefix")

    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(s3_path)
    dest_path = os.path.join(local_dir, filename)

    s3_client = create_s3_client(cfg)
    total_size = _get_object_size(s3_client, cfg.bucket, s3_path)
    progress = _ProgressBar(desc=dest_path, total=total_size)

    try:
        s3_client.download_file(cfg.bucket, s3_path, dest_path, Callback=progress)
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] Failed to download s3://{cfg.bucket}/{s3_path} -> {dest_path}: {exc}")
        return False
    finally:
        progress.close()

    if zipfile.is_zipfile(dest_path):
        with zipfile.ZipFile(dest_path, "r") as zf:
            zf.extractall(local_dir)
        try:
            os.remove(dest_path)
        except OSError:
            pass

    return True
