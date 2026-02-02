from __future__ import annotations

import json
import os
import posixpath
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

_PROPERTIES_ENCODING = "utf-8"
_SNAPSHOT_STATE_VERSION = 1
_SNAPSHOT_CACHE_DIR = ".snapshot_cache"


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


def _head_object(s3_client, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError):
        return None


def _get_object_size(s3_client, bucket: str, key: str) -> Optional[int]:
    head = _head_object(s3_client, bucket, key)
    if head is None:
        return None
    return head.get("ContentLength")


def _download_object(
    s3_client,
    bucket: str,
    key: str,
    dest_path: str,
    *,
    desc: Optional[str] = None,
    total_size: Optional[int] = None,
) -> bool:
    if total_size is None:
        total_size = _get_object_size(s3_client, bucket, key)
    progress = _ProgressBar(desc=desc or dest_path, total=total_size)
    try:
        s3_client.download_file(bucket, key, dest_path, Callback=progress)
        return True
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] Failed to download s3://{bucket}/{key} -> {dest_path}: {exc}")
        return False
    finally:
        progress.close()


def _normalize_archive_path(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized in ("", ".", "/"):
        raise ValueError(f"Unsafe path in tar: {path}")
    if normalized.startswith("/") or normalized.startswith("../"):
        raise ValueError(f"Unsafe path in tar: {path}")
    return normalized


def _normalize_manifest_blob(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    lines = text.splitlines()
    if not lines:
        return b""
    normalized = "".join(f"{line}\n" for line in lines)
    return normalized.encode("utf-8")


def _save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=True, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"State file must be a JSON object: {path}")
    return data


def _ensure_manifest_offset(path: str, expected_size: int) -> None:
    if expected_size < 0:
        raise ValueError(f"Invalid manifest offset: {expected_size}")

    if not os.path.exists(path):
        if expected_size != 0:
            raise ValueError(
                f"Merged manifest is missing but state expects {expected_size} bytes: {path}"
            )
        with open(path, "wb"):
            pass
        return

    current_size = os.path.getsize(path)
    if current_size < expected_size:
        raise ValueError(
            f"Merged manifest is shorter than state offset ({current_size} < {expected_size}): {path}"
        )
    if current_size > expected_size:
        with open(path, "r+b") as f:
            f.truncate(expected_size)


def _stream_extract_chunk(
    chunk_tar_path: str,
    local_dir: str,
    merged_manifest_path: str,
    *,
    verify_existing_files: bool,
) -> int:
    local_dir_abs = os.path.abspath(local_dir)
    manifest_blob: Optional[bytes] = None

    with tarfile.open(chunk_tar_path, "r:*") as tar:
        for member in tar:
            if member.islnk() or member.issym():
                raise ValueError(f"Symlink not allowed in tar: {member.name}")

            normalized = _normalize_archive_path(member.name)

            if member.isdir():
                continue

            if not member.isfile():
                raise ValueError(f"Unsupported tar entry type: {member.name}")

            if normalized == "manifest.jsonl":
                if manifest_blob is not None:
                    raise ValueError("Chunk contains duplicate manifest.jsonl files")
                extracted_manifest = tar.extractfile(member)
                if extracted_manifest is None:
                    raise ValueError("Failed to read manifest.jsonl from chunk")
                manifest_blob = _normalize_manifest_blob(extracted_manifest.read())
                continue

            dest_path = os.path.abspath(os.path.join(local_dir_abs, normalized))
            if os.path.commonpath([local_dir_abs, dest_path]) != local_dir_abs:
                raise ValueError(f"Unsafe path in tar: {member.name}")

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            if os.path.exists(dest_path):
                if not os.path.isfile(dest_path):
                    raise FileExistsError(f"Conflict while merging: {dest_path}")
                if verify_existing_files:
                    if os.path.getsize(dest_path) != member.size:
                        raise FileExistsError(f"Conflict while merging: {dest_path}")
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Failed to read tar entry: {member.name}")

            part_path = f"{dest_path}.part"
            try:
                with open(part_path, "wb") as out:
                    shutil.copyfileobj(extracted, out, length=1024 * 1024)
                os.replace(part_path, dest_path)
            finally:
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

    if manifest_blob is None:
        raise ValueError(f"Chunk missing manifest.jsonl: {chunk_tar_path}")

    with open(merged_manifest_path, "ab") as merged:
        merged.write(manifest_blob)
        merged.flush()
        os.fsync(merged.fileno())
    return len(manifest_blob)


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
    *,
    resume: bool = True,
    prefetch_chunks: int = 2,
    verify_existing_files: bool = True,
) -> bool:
    """
    Download snapshot chunks listed in manifest.json and merge them into local_dir.

    The output contains a merged manifest.jsonl and the merged images/tags/scores folders.
    Resuming is chunk-level and stores cache/state under <local_dir>/.snapshot_cache.
    """
    if not manifest_s3_path or manifest_s3_path.endswith("/"):
        raise ValueError("manifest_s3_path must be an object key, not a prefix")

    if prefetch_chunks < 1:
        raise ValueError("prefetch_chunks must be >= 1")

    if os.path.exists(local_dir) and os.listdir(local_dir) and not resume:
        raise ValueError(
            f"local_dir must be empty or non-existent when resume=False: {local_dir}"
        )

    os.makedirs(local_dir, exist_ok=True)
    merged_manifest_path = os.path.join(local_dir, "manifest.jsonl")

    cache_root = os.path.join(local_dir, _SNAPSHOT_CACHE_DIR)
    chunks_cache_dir = os.path.join(cache_root, "chunks")
    state_path = os.path.join(cache_root, "state.json")
    os.makedirs(chunks_cache_dir, exist_ok=True)

    s3_client = create_s3_client(cfg, region)

    manifest_head = _head_object(s3_client, bucket, manifest_s3_path)
    manifest_size = manifest_head.get("ContentLength") if manifest_head else None
    manifest_etag = manifest_head.get("ETag") if manifest_head else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        downloaded_manifest_path = tmp.name
    try:
        if not _download_object(
            s3_client,
            bucket,
            manifest_s3_path,
            downloaded_manifest_path,
            desc="snapshot manifest",
            total_size=manifest_size,
        ):
            return False
        with open(downloaded_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    finally:
        try:
            os.remove(downloaded_manifest_path)
        except OSError:
            pass

    chunk_paths = manifest.get("chunk_paths")
    if not isinstance(chunk_paths, list) or not chunk_paths:
        raise ValueError("manifest.json must include a non-empty chunk_paths list")

    prefix = posixpath.dirname(manifest_s3_path)

    for chunk_path in chunk_paths:
        if not isinstance(chunk_path, str) or not chunk_path:
            raise ValueError("chunk_paths must contain non-empty strings")
        if posixpath.isabs(chunk_path):
            raise ValueError(f"chunk_path must be relative: {chunk_path}")

    current_manifest_state = {
        "bucket": bucket,
        "manifest_s3_path": manifest_s3_path,
        "manifest_etag": manifest_etag,
        "manifest_size": manifest_size,
        "total_chunks": len(chunk_paths),
    }

    existing_state = _load_json(state_path)
    if existing_state is None:
        if resume and os.path.exists(merged_manifest_path):
            if os.path.getsize(merged_manifest_path) > 0:
                raise ValueError(
                    "resume=True but state file is missing while manifest.jsonl already "
                    "contains data; clear local_dir/.snapshot_cache and retry."
                )
        existing_state = {
            "version": _SNAPSHOT_STATE_VERSION,
            "manifest": current_manifest_state,
            "merged_manifest_bytes": 0,
            "completed_chunks": {},
        }
    else:
        if existing_state.get("version") != _SNAPSHOT_STATE_VERSION:
            raise ValueError(
                f"Unsupported snapshot state version: {existing_state.get('version')}"
            )
        if resume and existing_state.get("manifest") != current_manifest_state:
            raise ValueError(
                "Cached state does not match current snapshot manifest; "
                "clear local_dir/.snapshot_cache or use a fresh local_dir."
            )
        if not resume:
            existing_state = {
                "version": _SNAPSHOT_STATE_VERSION,
                "manifest": current_manifest_state,
                "merged_manifest_bytes": 0,
                "completed_chunks": {},
            }

    completed_chunks = existing_state.get("completed_chunks")
    if not isinstance(completed_chunks, dict):
        raise ValueError("Invalid snapshot state: completed_chunks must be an object")

    merged_manifest_bytes = existing_state.get("merged_manifest_bytes", 0)
    if not isinstance(merged_manifest_bytes, int):
        raise ValueError("Invalid snapshot state: merged_manifest_bytes must be int")
    _ensure_manifest_offset(merged_manifest_path, merged_manifest_bytes)
    _save_json_atomic(state_path, existing_state)

    @dataclass
    class _ChunkDownloadResult:
        index: int
        chunk_path: str
        chunk_key: str
        cache_path: str
        content_length: Optional[int]
        etag: Optional[str]
        skip_processing: bool
        error: Optional[str] = None

    queue: "Queue[Optional[_ChunkDownloadResult]]" = Queue(maxsize=prefetch_chunks)

    def _download_worker() -> None:
        for index, chunk_path in enumerate(chunk_paths):
            chunk_key = ""
            try:
                chunk_key = posixpath.join(prefix, chunk_path) if prefix else chunk_path
                chunk_suffix = os.path.splitext(posixpath.basename(chunk_path))[1] or ".tar"
                chunk_cache_path = os.path.join(
                    chunks_cache_dir,
                    f"{index:06d}{chunk_suffix}",
                )

                chunk_head = _head_object(s3_client, bucket, chunk_key)
                content_length = (
                    chunk_head.get("ContentLength") if chunk_head is not None else None
                )
                etag = chunk_head.get("ETag") if chunk_head is not None else None

                cached_chunk_state = completed_chunks.get(str(index))
                if isinstance(cached_chunk_state, dict) and cached_chunk_state.get(
                    "chunk_path"
                ) == chunk_path:
                    size_matches = True
                    if (
                        content_length is not None
                        and cached_chunk_state.get("content_length") is not None
                    ):
                        size_matches = (
                            int(cached_chunk_state["content_length"]) == content_length
                        )
                    etag_matches = True
                    if etag and cached_chunk_state.get("etag"):
                        etag_matches = cached_chunk_state["etag"] == etag
                    if size_matches and etag_matches:
                        queue.put(
                            _ChunkDownloadResult(
                                index=index,
                                chunk_path=chunk_path,
                                chunk_key=chunk_key,
                                cache_path=chunk_cache_path,
                                content_length=content_length,
                                etag=etag,
                                skip_processing=True,
                            )
                        )
                        continue

                cache_is_valid = False
                if os.path.exists(chunk_cache_path) and os.path.isfile(chunk_cache_path):
                    cache_is_valid = True
                    if content_length is not None:
                        cache_is_valid = os.path.getsize(chunk_cache_path) == content_length

                if not cache_is_valid:
                    chunk_tmp_path = f"{chunk_cache_path}.download"
                    try:
                        if not _download_object(
                            s3_client,
                            bucket,
                            chunk_key,
                            chunk_tmp_path,
                            desc=f"chunk {index + 1}/{len(chunk_paths)}",
                            total_size=content_length,
                        ):
                            raise RuntimeError(
                                f"Failed to download chunk: s3://{bucket}/{chunk_key}"
                            )
                        os.replace(chunk_tmp_path, chunk_cache_path)
                    finally:
                        if os.path.exists(chunk_tmp_path):
                            try:
                                os.remove(chunk_tmp_path)
                            except OSError:
                                pass
                    if content_length is not None and os.path.getsize(
                        chunk_cache_path
                    ) != content_length:
                        raise RuntimeError(
                            f"Downloaded chunk size mismatch: s3://{bucket}/{chunk_key}"
                        )

                queue.put(
                    _ChunkDownloadResult(
                        index=index,
                        chunk_path=chunk_path,
                        chunk_key=chunk_key,
                        cache_path=chunk_cache_path,
                        content_length=content_length,
                        etag=etag,
                        skip_processing=False,
                    )
                )
            except Exception as exc:
                queue.put(
                    _ChunkDownloadResult(
                        index=index,
                        chunk_path=chunk_path,
                        chunk_key=chunk_key,
                        cache_path="",
                        content_length=None,
                        etag=None,
                        skip_processing=False,
                        error=str(exc),
                    )
                )
                queue.put(None)
                return
        queue.put(None)

    downloader = Thread(target=_download_worker, daemon=True)
    downloader.start()

    results_by_index: Dict[int, _ChunkDownloadResult] = {}
    chunk_progress = tqdm(total=len(chunk_paths), unit="chunk", desc="chunks")
    stream_error: Optional[str] = None

    try:
        for index, chunk_path in enumerate(chunk_paths):
            while index not in results_by_index:
                item = queue.get()
                if item is None:
                    break
                results_by_index[item.index] = item

            result = results_by_index.pop(index, None)
            if result is None:
                stream_error = f"Missing chunk download result for index {index}"
                break
            if result.error:
                stream_error = result.error
                break

            if result.skip_processing:
                chunk_progress.update(1)
                continue

            manifest_bytes_written = _stream_extract_chunk(
                result.cache_path,
                local_dir,
                merged_manifest_path,
                verify_existing_files=verify_existing_files,
            )

            existing_state["merged_manifest_bytes"] = (
                int(existing_state.get("merged_manifest_bytes", 0))
                + manifest_bytes_written
            )
            completed_chunks[str(index)] = {
                "chunk_path": chunk_path,
                "chunk_key": result.chunk_key,
                "etag": result.etag,
                "content_length": result.content_length,
            }
            _save_json_atomic(state_path, existing_state)
            chunk_progress.update(1)
    finally:
        chunk_progress.close()
        downloader.join()

    if stream_error is not None:
        print(f"[WARN] Snapshot download failed: {stream_error}")
        return False

    return True
