from __future__ import annotations

import json
import os
import posixpath
import shutil
import tarfile
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

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None


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

    access_key = props.get("s3.accessKey") or props.get("accessKey")
    secret_key = props.get("s3.secretKey") or props.get("secretKey")
    session_token = props.get("s3.sessionToken")

    return S3Config(
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
    )


def create_s3_client(cfg: S3Config, region: Optional[str] = None):
    """
    Create a boto3 S3 client from S3Config.

    If access_key / secret_key are not provided in the config, standard
    AWS credential resolution is used (env vars, shared credentials file, etc.).
    Region can be provided per client call.
    """
    boto_config = BotoConfig(
        signature_version="s3v4",
        max_pool_connections=16,
    )

    session_kwargs = {}
    client_kwargs = {}

    if region:
        client_kwargs["region_name"] = region

    if cfg.access_key and cfg.secret_key:
        session_kwargs["aws_access_key_id"] = cfg.access_key
        session_kwargs["aws_secret_access_key"] = cfg.secret_key

    if cfg.session_token:
        session_kwargs["aws_session_token"] = cfg.session_token

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


def upload_to_s3(
    cfg: S3Config,
    bucket: str,
    region: Optional[str],
    local_path: str,
    s3_path: str,
) -> bool:
    """
    Upload a file or directory to S3.

    bucket and region are supplied per call.
    If local_path is a directory, it is zipped before upload and s3_path
    must end with .zip.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path not found: {local_path}")

    s3_client = create_s3_client(cfg, region)

    if os.path.isdir(local_path):
        if not s3_path.endswith(".zip"):
            raise ValueError("Uploading a directory requires s3_path to end with .zip")
        archive_path = _zip_directory(local_path)
        try:
            return _upload_file(s3_client, bucket, archive_path, s3_path)
        finally:
            try:
                os.remove(archive_path)
            except OSError:
                pass

    return _upload_file(s3_client, bucket, local_path, s3_path)


def _get_object_size(s3_client, bucket: str, key: str) -> Optional[int]:
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        return resp.get("ContentLength")
    except (BotoCoreError, ClientError):
        return None


def _download_object(s3_client, bucket: str, key: str, dest_path: str) -> bool:
    total_size = _get_object_size(s3_client, bucket, key)
    progress = _ProgressBar(desc=dest_path, total=total_size)
    try:
        s3_client.download_file(bucket, key, dest_path, Callback=progress)
        return True
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] Failed to download s3://{bucket}/{key} -> {dest_path}: {exc}")
        return False
    finally:
        progress.close()


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:
    base_dir = os.path.abspath(dest_dir)
    for member in tar.getmembers():
        if member.islnk() or member.issym():
            raise ValueError(f"Symlink not allowed in tar: {member.name}")
        member_path = os.path.abspath(os.path.join(base_dir, member.name))
        if os.path.commonpath([base_dir, member_path]) != base_dir:
            raise ValueError(f"Unsafe path in tar: {member.name}")
    tar.extractall(dest_dir)


def download_from_s3(
    cfg: S3Config,
    bucket: str,
    region: Optional[str],
    s3_path: str,
    local_dir: str,
) -> bool:
    """
    Download an S3 object into a local directory.

    bucket and region are supplied per call.
    If the downloaded object is a zip, it is extracted into local_dir
    and the zip file is removed.
    """
    if not s3_path or s3_path.endswith("/"):
        raise ValueError("s3_path must be an object key, not a prefix")

    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(s3_path)
    dest_path = os.path.join(local_dir, filename)

    s3_client = create_s3_client(cfg, region)
    total_size = _get_object_size(s3_client, bucket, s3_path)
    progress = _ProgressBar(desc=dest_path, total=total_size)

    try:
        s3_client.download_file(bucket, s3_path, dest_path, Callback=progress)
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] Failed to download s3://{bucket}/{s3_path} -> {dest_path}: {exc}")
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


def download_snapshot_from_s3(
    cfg: S3Config,
    bucket: str,
    region: Optional[str],
    manifest_s3_path: str,
    local_dir: str,
) -> bool:
    """
    Download snapshot chunks listed in manifest.json and merge them into local_dir.

    The output contains a merged manifest.jsonl and the merged images/tags/scores folders.
    """
    if not manifest_s3_path or manifest_s3_path.endswith("/"):
        raise ValueError("manifest_s3_path must be an object key, not a prefix")

    if os.path.exists(local_dir) and os.listdir(local_dir):
        raise ValueError(f"local_dir must be empty or non-existent: {local_dir}")

    os.makedirs(local_dir, exist_ok=True)

    s3_client = create_s3_client(cfg, region)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        manifest_path = tmp.name

    try:
        if not _download_object(s3_client, bucket, manifest_s3_path, manifest_path):
            return False
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    finally:
        try:
            os.remove(manifest_path)
        except OSError:
            pass

    chunk_paths = manifest.get("chunk_paths")
    if not isinstance(chunk_paths, list) or not chunk_paths:
        raise ValueError("manifest.json must include a non-empty chunk_paths list")

    prefix = posixpath.dirname(manifest_s3_path)
    merged_manifest_path = os.path.join(local_dir, "manifest.jsonl")

    for chunk_path in chunk_paths:
        if not isinstance(chunk_path, str) or not chunk_path:
            raise ValueError("chunk_paths must contain non-empty strings")
        if posixpath.isabs(chunk_path):
            raise ValueError(f"chunk_path must be relative: {chunk_path}")

        chunk_key = posixpath.join(prefix, chunk_path) if prefix else chunk_path
        chunk_suffix = os.path.splitext(posixpath.basename(chunk_path))[1] or ".tar"

        with tempfile.NamedTemporaryFile(delete=False, suffix=chunk_suffix) as tmp_chunk:
            tmp_chunk_path = tmp_chunk.name

        try:
            if not _download_object(s3_client, bucket, chunk_key, tmp_chunk_path):
                return False

            if not tarfile.is_tarfile(tmp_chunk_path):
                raise ValueError(f"Chunk is not a tar file: {chunk_path}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                with tarfile.open(tmp_chunk_path, "r:*") as tar:
                    _safe_extract_tar(tar, tmp_dir)

                chunk_manifest_path = os.path.join(tmp_dir, "manifest.jsonl")
                if not os.path.exists(chunk_manifest_path):
                    raise ValueError(f"Chunk missing manifest.jsonl: {chunk_path}")

                with open(merged_manifest_path, "a", encoding="utf-8") as merged:
                    with open(chunk_manifest_path, "r", encoding="utf-8") as chunk_manifest:
                        for line in chunk_manifest:
                            if line.endswith("\n"):
                                merged.write(line)
                            else:
                                merged.write(f"{line}\n")

                for root, _, files in os.walk(tmp_dir):
                    for filename in files:
                        src_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(src_path, start=tmp_dir)
                        if rel_path == "manifest.jsonl":
                            continue
                        dest_path = os.path.join(local_dir, rel_path)
                        if os.path.exists(dest_path):
                            raise FileExistsError(f"Conflict while merging: {dest_path}")
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.move(src_path, dest_path)
        finally:
            try:
                os.remove(tmp_chunk_path)
            except OSError:
                pass

    return True
