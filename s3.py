from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock
from typing import Any, Callable, Dict, Optional, Union

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

_PROPERTIES_ENCODING = "utf-8"
_SNAPSHOT_STATE_VERSION = 1
_SNAPSHOT_CACHE_DIR = ".snapshot_cache"
_ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
_MANIFEST_ALLOWED_FIELDS = {
    "external_id",
    "image_path",
    "tags_path",
    "scores_path",
    "width",
    "height",
}
_SOURCE_LABEL_RE = re.compile(r"^[a-z0-9._-]{1,64}$")
_SOURCE_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9 ._\-:@()\[\]{}+,=;!$'~]{1,512}$")


@dataclass
class S3Config:
    """S3 connection settings."""

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None


class SnapshotTransferPhase(str, Enum):
    """Source-owned stages emitted by :func:`download_snapshot_from_s3`."""

    MANIFEST_READ = "manifest_read"
    CHUNK_TRANSFER = "chunk_transfer"
    EXTRACTION = "extraction"
    ORDERED_MERGE = "ordered_merge"
    COMPLETED = "completed"


class SnapshotMaterializationResult(str, Enum):
    """Terminal result returned when snapshot cancellation is enabled."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


SnapshotCancellationSignal = Union[Event, Callable[[], bool]]


class _SnapshotMaterializationCancelled(Exception):
    """Private control-flow marker for cooperative snapshot cancellation."""


@dataclass(frozen=True)
class SnapshotTransferProgress:
    """One immutable snapshot-materialization observation.

    ``network_bytes`` counts every byte quantity received from boto's chunk
    transfer callbacks, including a retry of a previously received range.
    ``verified_bytes`` counts a successfully validated cache file only once
    per chunk.  It is ``None`` when resumed state lacks the size of an already
    completed chunk, rather than inventing a partial total.

    Item counters count validated ``manifest.jsonl`` records.  They are
    independent from the terminal ``COMPLETED`` phase: callers must continue
    to use the loader result for success or failure authority.
    """

    phase: SnapshotTransferPhase
    operation_id: Optional[str]
    attempt_id: Optional[str]
    network_bytes: int
    verified_bytes: Optional[int]
    total_bytes: Optional[int]
    completed_chunks: int
    total_chunks: int
    completed_items: int
    total_items: Optional[int]


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
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> bool:
    if total_size is None:
        total_size = _get_object_size(s3_client, bucket, key)
    progress = _ProgressBar(desc=desc or dest_path, total=total_size)

    def _report_progress(bytes_amount: int) -> None:
        progress(bytes_amount)
        if cancellation_requested is not None and cancellation_requested():
            raise _SnapshotMaterializationCancelled()

    try:
        s3_client.download_file(bucket, key, dest_path, Callback=_report_progress)
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


def _validate_external_id(external_id: str) -> None:
    if ":" not in external_id:
        raise ValueError(f"Invalid external_id (missing ':'): {external_id}")
    source_label, source_item_id = external_id.split(":", 1)

    if not _SOURCE_LABEL_RE.fullmatch(source_label):
        raise ValueError(f"Invalid source_label in external_id: {external_id}")
    if not _SOURCE_ITEM_ID_RE.fullmatch(source_item_id):
        raise ValueError(f"Invalid source_item_id in external_id: {external_id}")

    if source_item_id != source_item_id.strip():
        raise ValueError(f"Invalid source_item_id whitespace in external_id: {external_id}")
    if ".." in source_item_id:
        raise ValueError(f"Invalid source_item_id '..' in external_id: {external_id}")
    if "/" in source_item_id or "\\" in source_item_id:
        raise ValueError(f"Invalid source_item_id path separator in external_id: {external_id}")
    if "\0" in source_item_id or "\n" in source_item_id or "\r" in source_item_id:
        raise ValueError(f"Invalid source_item_id control character in external_id: {external_id}")


def _normalize_manifest_entry_path(path: Any, field_name: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field_name} must be a non-empty string")
    return _normalize_archive_path(path)


def _validate_chunk_manifest_entries(
    *,
    chunk_tar_path: str,
    manifest_blob: bytes,
    file_paths: list[str],
) -> None:
    manifest_text = manifest_blob.decode("utf-8")
    lines = manifest_text.splitlines()
    files_set = set(file_paths)
    seen_external_ids: set[str] = set()

    for line_no, line in enumerate(lines, start=1):
        if line == "":
            raise ValueError(
                f"manifest.jsonl in {chunk_tar_path} contains an empty line at {line_no}"
            )

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in manifest.jsonl ({chunk_tar_path}:{line_no}): {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"manifest.jsonl line must be an object ({chunk_tar_path}:{line_no})"
            )

        unknown_fields = set(payload.keys()) - _MANIFEST_ALLOWED_FIELDS
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(
                f"manifest.jsonl has unsupported fields ({chunk_tar_path}:{line_no}): {unknown}"
            )

        if "external_id" not in payload or "image_path" not in payload:
            raise ValueError(
                f"manifest.jsonl line missing required fields ({chunk_tar_path}:{line_no})"
            )

        external_id = payload["external_id"]
        if not isinstance(external_id, str) or not external_id:
            raise ValueError(
                f"external_id must be a non-empty string ({chunk_tar_path}:{line_no})"
            )
        _validate_external_id(external_id)
        if external_id in seen_external_ids:
            raise ValueError(
                f"Duplicate external_id in chunk manifest ({chunk_tar_path}:{line_no}): {external_id}"
            )
        seen_external_ids.add(external_id)

        image_path = _normalize_manifest_entry_path(payload["image_path"], "image_path")
        if image_path not in files_set:
            raise ValueError(
                f"manifest.jsonl references missing image file ({chunk_tar_path}:{line_no}): {image_path}"
            )
        if not image_path.startswith("images/"):
            raise ValueError(
                f"image_path must be under images/ ({chunk_tar_path}:{line_no}): {image_path}"
            )
        image_basename = posixpath.basename(image_path)
        if "." not in image_basename:
            raise ValueError(
                f"image_path must include extension ({chunk_tar_path}:{line_no}): {image_path}"
            )
        image_stem, image_ext = image_basename.rsplit(".", 1)
        if image_stem != external_id:
            raise ValueError(
                f"image_path must match external_id ({chunk_tar_path}:{line_no})"
            )
        if image_ext.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension in image_path ({chunk_tar_path}:{line_no}): {image_ext}"
            )

        if "tags_path" in payload:
            tags_path = _normalize_manifest_entry_path(payload["tags_path"], "tags_path")
            if tags_path not in files_set:
                raise ValueError(
                    f"manifest.jsonl references missing tags file ({chunk_tar_path}:{line_no}): {tags_path}"
                )
            if tags_path != f"tags/{external_id}.txt":
                raise ValueError(
                    f"tags_path must match external_id ({chunk_tar_path}:{line_no})"
                )

        if "scores_path" in payload:
            scores_path = _normalize_manifest_entry_path(payload["scores_path"], "scores_path")
            if scores_path not in files_set:
                raise ValueError(
                    f"manifest.jsonl references missing scores file ({chunk_tar_path}:{line_no}): {scores_path}"
                )
            if scores_path != f"scores/{external_id}.jsonl":
                raise ValueError(
                    f"scores_path must match external_id ({chunk_tar_path}:{line_no})"
                )

        if "width" in payload:
            width = payload["width"]
            if not isinstance(width, int) or width <= 0:
                raise ValueError(
                    f"width must be a positive integer ({chunk_tar_path}:{line_no})"
                )

        if "height" in payload:
            height = payload["height"]
            if not isinstance(height, int) or height <= 0:
                raise ValueError(
                    f"height must be a positive integer ({chunk_tar_path}:{line_no})"
                )


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


@dataclass
class _ChunkInspection:
    manifest_blob: bytes
    file_paths: list[str]
    item_count: int


def _inspect_chunk_tar(
    chunk_tar_path: str,
    *,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> _ChunkInspection:
    manifest_blob: Optional[bytes] = None
    file_paths: list[str] = []
    seen_paths: set[str] = set()

    with tarfile.open(chunk_tar_path, "r:*") as tar:
        for member in tar:
            if cancellation_requested is not None and cancellation_requested():
                raise _SnapshotMaterializationCancelled()
            if member.islnk() or member.issym():
                raise ValueError(f"Symlink not allowed in tar: {member.name}")

            normalized = _normalize_archive_path(member.name)
            if normalized in seen_paths:
                raise ValueError(f"Duplicate tar entry in chunk: {member.name}")
            seen_paths.add(normalized)

            if member.isdir():
                continue

            if not member.isfile():
                raise ValueError(f"Unsupported tar entry type: {member.name}")

            if normalized == "manifest.jsonl":
                extracted_manifest = tar.extractfile(member)
                if extracted_manifest is None:
                    raise ValueError("Failed to read manifest.jsonl from chunk")
                manifest_blob = _normalize_manifest_blob(extracted_manifest.read())
                continue

            file_paths.append(normalized)

    if manifest_blob is None:
        raise ValueError(f"Chunk missing manifest.jsonl: {chunk_tar_path}")

    _validate_chunk_manifest_entries(
        chunk_tar_path=chunk_tar_path,
        manifest_blob=manifest_blob,
        file_paths=file_paths,
    )

    return _ChunkInspection(
        manifest_blob=manifest_blob,
        file_paths=file_paths,
        item_count=len(manifest_blob.splitlines()),
    )


def _extract_chunk_with_python_tar(
    chunk_tar_path: str,
    staging_dir: str,
    *,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> None:
    with tarfile.open(chunk_tar_path, "r:*") as tar:
        for member in tar:
            if cancellation_requested is not None and cancellation_requested():
                raise _SnapshotMaterializationCancelled()
            normalized = _normalize_archive_path(member.name)
            if normalized == "manifest.jsonl" or member.isdir():
                continue
            if not member.isfile():
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Failed to read tar entry: {member.name}")

            dest_path = os.path.abspath(os.path.join(staging_dir, normalized))
            if os.path.commonpath([os.path.abspath(staging_dir), dest_path]) != os.path.abspath(
                staging_dir
            ):
                raise ValueError(f"Unsafe path in tar: {member.name}")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            part_path = f"{dest_path}.part"
            try:
                with open(part_path, "wb") as out:
                    while True:
                        if (
                            cancellation_requested is not None
                            and cancellation_requested()
                        ):
                            raise _SnapshotMaterializationCancelled()
                        block = extracted.read(4 * 1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                os.replace(part_path, dest_path)
            finally:
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass


def _extract_chunk_with_system_tar(
    *,
    system_tar_path: str,
    chunk_tar_path: str,
    staging_dir: str,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> None:
    cmd = [
        system_tar_path,
        "-xf",
        chunk_tar_path,
        "-C",
        staging_dir,
        "--exclude=manifest.jsonl",
        "--no-same-owner",
        "--no-same-permissions",
        "--delay-directory-restore",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    while process.poll() is None:
        if cancellation_requested is not None and cancellation_requested():
            # This is an owned direct child.  We wait for it to exit before
            # returning cancellation, but do not claim to hard-kill an
            # uncooperative third-party process.
            process.terminate()
            process.communicate()
            raise _SnapshotMaterializationCancelled()
        time.sleep(0.05)

    _, stderr_output = process.communicate()
    if process.returncode != 0:
        stderr = (stderr_output or "").strip()
        if stderr:
            raise RuntimeError(f"tar extraction failed: {stderr}")
        raise RuntimeError("tar extraction failed")


def _merge_staged_chunk(
    *,
    staging_dir: str,
    local_dir: str,
    file_paths: list[str],
    verify_existing_files: bool,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> None:
    staging_abs = os.path.abspath(staging_dir)
    local_abs = os.path.abspath(local_dir)

    for rel_path in file_paths:
        if cancellation_requested is not None and cancellation_requested():
            raise _SnapshotMaterializationCancelled()
        src_path = os.path.abspath(os.path.join(staging_abs, rel_path))
        if os.path.commonpath([staging_abs, src_path]) != staging_abs:
            raise ValueError(f"Unsafe staged path: {rel_path}")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Expected extracted file is missing: {src_path}")

        dest_path = os.path.abspath(os.path.join(local_abs, rel_path))
        if os.path.commonpath([local_abs, dest_path]) != local_abs:
            raise ValueError(f"Unsafe destination path: {rel_path}")

        if os.path.exists(dest_path):
            if verify_existing_files:
                if not os.path.isfile(dest_path):
                    raise FileExistsError(f"Conflict while merging: {dest_path}")
                if os.path.getsize(dest_path) != os.path.getsize(src_path):
                    raise FileExistsError(f"Conflict while merging: {dest_path}")
                if not _files_are_identical(
                    src_path,
                    dest_path,
                    cancellation_requested=cancellation_requested,
                ):
                    raise FileExistsError(f"Conflict while merging: {dest_path}")
                # A process can stop after os.replace() moved this file but
                # before the chunk state was committed. Treat only a
                # byte-identical destination as the completed part of that
                # interrupted merge.
                continue
            raise FileExistsError(f"Conflict while merging: {dest_path}")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.replace(src_path, dest_path)
        if cancellation_requested is not None and cancellation_requested():
            raise _SnapshotMaterializationCancelled()


def _files_are_identical(
    left_path: str,
    right_path: str,
    *,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> bool:
    with open(left_path, "rb") as left, open(right_path, "rb") as right:
        while True:
            if cancellation_requested is not None and cancellation_requested():
                raise _SnapshotMaterializationCancelled()
            left_block = left.read(4 * 1024 * 1024)
            right_block = right.read(4 * 1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


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
    download_workers: Optional[int] = None,
    extract_workers: Optional[int] = None,
    prefer_system_tar: bool = True,
    progress_callback: Optional[Callable[[SnapshotTransferProgress], None]] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    cancellation_signal: Optional[SnapshotCancellationSignal] = None,
) -> Union[bool, SnapshotMaterializationResult]:
    """
    Download snapshot chunks listed in manifest.json and merge them into local_dir.

    The output contains a merged manifest.jsonl and the merged images/tags/scores folders.
    Resuming is chunk-level and stores cache/state under <local_dir>/.snapshot_cache.

    ``progress_callback`` receives immutable, source-owned observations.  The
    optional operation and attempt IDs are returned unchanged in each
    observation so an owning coordinator can bind it to its durable attempt.
    This loader does not interpret either identifier, create terminal status,
    or infer totals that the manifest and transfer callbacks do not expose.

    ``cancellation_signal`` may be a :class:`threading.Event` or a no-argument
    callable returning ``True`` when cancellation is requested.  With a signal,
    this function returns :class:`SnapshotMaterializationResult` so cancellation
    remains distinct from transfer failure.  Existing callers that omit it keep
    the historical ``True``/``False`` result.
    """
    def _cancellation_requested() -> bool:
        if cancellation_signal is None:
            return False
        if callable(cancellation_signal):
            return bool(cancellation_signal())
        return cancellation_signal.is_set()

    def _result(
        outcome: SnapshotMaterializationResult,
    ) -> Union[bool, SnapshotMaterializationResult]:
        if cancellation_signal is None:
            return outcome is SnapshotMaterializationResult.SUCCESS
        return outcome

    def _raise_if_cancelled() -> None:
        if _cancellation_requested():
            raise _SnapshotMaterializationCancelled()

    if not manifest_s3_path or manifest_s3_path.endswith("/"):
        raise ValueError("manifest_s3_path must be an object key, not a prefix")

    if prefetch_chunks < 1:
        raise ValueError("prefetch_chunks must be >= 1")

    if download_workers is None:
        download_workers = max(1, prefetch_chunks)
    if extract_workers is None:
        cpu_count = os.cpu_count() or 2
        extract_workers = max(1, min(8, cpu_count))

    if download_workers < 1:
        raise ValueError("download_workers must be >= 1")
    if extract_workers < 1:
        raise ValueError("extract_workers must be >= 1")

    if _cancellation_requested():
        return _result(SnapshotMaterializationResult.CANCELLED)

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

    if _cancellation_requested():
        return _result(SnapshotMaterializationResult.CANCELLED)

    manifest_head = _head_object(s3_client, bucket, manifest_s3_path)
    manifest_size = manifest_head.get("ContentLength") if manifest_head else None
    manifest_etag = manifest_head.get("ETag") if manifest_head else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        downloaded_manifest_path = tmp.name
    try:
        _raise_if_cancelled()
        if not _download_object(
            s3_client,
            bucket,
            manifest_s3_path,
            downloaded_manifest_path,
            desc="snapshot manifest",
            total_size=manifest_size,
            cancellation_requested=_cancellation_requested,
        ):
            return _result(SnapshotMaterializationResult.FAILED)
        _raise_if_cancelled()
        with open(downloaded_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except _SnapshotMaterializationCancelled:
        return _result(SnapshotMaterializationResult.CANCELLED)
    finally:
        try:
            os.remove(downloaded_manifest_path)
        except OSError:
            pass

    chunk_paths = manifest.get("chunk_paths")
    if not isinstance(chunk_paths, list) or not chunk_paths:
        raise ValueError("manifest.json must include a non-empty chunk_paths list")

    # Surface a successful manifest read before the per-chunk head requests.
    # There are no worker callbacks yet, so this direct callback is serialized
    # with every later one by program order.
    if progress_callback is not None:
        progress_callback(
            SnapshotTransferProgress(
                phase=SnapshotTransferPhase.MANIFEST_READ,
                operation_id=operation_id,
                attempt_id=attempt_id,
                network_bytes=0,
                verified_bytes=0,
                total_bytes=None,
                completed_chunks=0,
                total_chunks=len(chunk_paths),
                completed_items=0,
                total_items=None,
            )
        )

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
    class _ChunkPlan:
        index: int
        chunk_path: str
        chunk_key: str
        cache_path: str
        content_length: Optional[int]
        etag: Optional[str]
        skip_processing: bool

    @dataclass
    class _PreparedChunk:
        index: int
        chunk_path: str
        chunk_key: str
        cache_path: str
        content_length: Optional[int]
        etag: Optional[str]
        staging_dir: str
        file_paths: list[str]
        manifest_blob: bytes
        item_count: int

    chunk_plans: list[_ChunkPlan] = []
    for index, chunk_path in enumerate(chunk_paths):
        if _cancellation_requested():
            return _result(SnapshotMaterializationResult.CANCELLED)
        chunk_key = posixpath.join(prefix, chunk_path) if prefix else chunk_path
        chunk_suffix = os.path.splitext(posixpath.basename(chunk_path))[1] or ".tar"
        chunk_cache_path = os.path.join(
            chunks_cache_dir,
            f"{index:06d}{chunk_suffix}",
        )

        chunk_head = _head_object(s3_client, bucket, chunk_key)
        content_length = chunk_head.get("ContentLength") if chunk_head is not None else None
        etag = chunk_head.get("ETag") if chunk_head is not None else None

        cached_chunk_state = completed_chunks.get(str(index))
        skip_processing = False
        if isinstance(cached_chunk_state, dict) and cached_chunk_state.get("chunk_path") == chunk_path:
            size_matches = True
            if content_length is not None and cached_chunk_state.get("content_length") is not None:
                size_matches = int(cached_chunk_state["content_length"]) == content_length
            etag_matches = True
            if etag and cached_chunk_state.get("etag"):
                etag_matches = cached_chunk_state["etag"] == etag
            skip_processing = size_matches and etag_matches

        chunk_plans.append(
            _ChunkPlan(
                index=index,
                chunk_path=chunk_path,
                chunk_key=chunk_key,
                cache_path=chunk_cache_path,
                content_length=content_length,
                etag=etag,
                skip_processing=skip_processing,
            )
        )

    chunks_to_download = [plan for plan in chunk_plans if not plan.skip_processing]

    def _count_existing_manifest_items(path: str) -> int:
        with open(path, "rb") as merged_manifest:
            return sum(1 for line in merged_manifest if line.strip())

    class _SnapshotProgressReporter:
        """Serialize worker callbacks into monotone source observations."""

        def __init__(self) -> None:
            self._lock = Lock()
            self._network_bytes = 0
            self._verified_bytes = 0
            self._verified_sizes_unknown = False
            self._verified_chunks: set[int] = set()
            self._completed_chunks = sum(
                1 for plan in chunk_plans if plan.skip_processing
            )
            self._total_chunks = len(chunk_plans)
            self._completed_items = _count_existing_manifest_items(
                merged_manifest_path
            )
            self._discovered_items = self._completed_items
            self._inspected_chunks: set[int] = set()
            self._chunks_to_inspect = len(chunks_to_download)
            self._total_items: Optional[int] = (
                self._completed_items if self._chunks_to_inspect == 0 else None
            )
            content_lengths = [plan.content_length for plan in chunk_plans]
            self._total_bytes: Optional[int] = (
                sum(content_lengths)
                if all(length is not None for length in content_lengths)
                else None
            )

            for plan in chunk_plans:
                if not plan.skip_processing:
                    continue
                if plan.content_length is not None:
                    self._register_verified_chunk_locked(
                        plan.index,
                        plan.content_length,
                    )
                elif os.path.isfile(plan.cache_path):
                    self._register_verified_chunk_locked(
                        plan.index,
                        os.path.getsize(plan.cache_path),
                    )
                else:
                    self._register_verified_chunk_locked(plan.index, None)

        def _register_verified_chunk_locked(
            self,
            chunk_index: int,
            size: Optional[int],
        ) -> None:
            if chunk_index in self._verified_chunks:
                return
            self._verified_chunks.add(chunk_index)
            if size is None:
                self._verified_sizes_unknown = True
                return
            if size < 0:
                raise ValueError("Verified chunk size must be non-negative")
            self._verified_bytes += size
            if (
                self._total_bytes is not None
                and self._verified_bytes > self._total_bytes
            ):
                raise RuntimeError("Verified snapshot bytes exceed the known total")

        def _snapshot_locked(
            self,
            phase: SnapshotTransferPhase,
        ) -> SnapshotTransferProgress:
            return SnapshotTransferProgress(
                phase=phase,
                operation_id=operation_id,
                attempt_id=attempt_id,
                network_bytes=self._network_bytes,
                verified_bytes=(
                    None if self._verified_sizes_unknown else self._verified_bytes
                ),
                total_bytes=self._total_bytes,
                completed_chunks=self._completed_chunks,
                total_chunks=self._total_chunks,
                completed_items=self._completed_items,
                total_items=self._total_items,
            )

        def emit(self, phase: SnapshotTransferPhase) -> None:
            if progress_callback is None:
                return
            with self._lock:
                progress_callback(self._snapshot_locked(phase))

        def record_network_bytes(self, byte_count: int) -> None:
            if byte_count < 0:
                raise ValueError("Chunk transfer callback reported negative bytes")
            with self._lock:
                self._network_bytes += byte_count
                if progress_callback is not None:
                    progress_callback(
                        self._snapshot_locked(SnapshotTransferPhase.CHUNK_TRANSFER)
                    )

        def record_verified_chunk(self, chunk_index: int, size: int) -> None:
            with self._lock:
                self._register_verified_chunk_locked(chunk_index, size)
                if progress_callback is not None:
                    progress_callback(
                        self._snapshot_locked(SnapshotTransferPhase.CHUNK_TRANSFER)
                    )

        def record_extracted_items(self, chunk_index: int, item_count: int) -> None:
            if item_count < 0:
                raise ValueError("Extracted item count must be non-negative")
            with self._lock:
                if chunk_index in self._inspected_chunks:
                    return
                self._inspected_chunks.add(chunk_index)
                self._discovered_items += item_count
                if len(self._inspected_chunks) == self._chunks_to_inspect:
                    self._total_items = self._discovered_items
                if progress_callback is not None:
                    progress_callback(
                        self._snapshot_locked(SnapshotTransferPhase.EXTRACTION)
                    )

        def record_merged_chunk(self, item_count: int) -> None:
            if item_count < 0:
                raise ValueError("Merged item count must be non-negative")
            with self._lock:
                self._completed_chunks += 1
                self._completed_items += item_count
                if self._completed_chunks > self._total_chunks:
                    raise RuntimeError("Completed chunks exceed the manifest total")
                if (
                    self._total_items is not None
                    and self._completed_items > self._total_items
                ):
                    raise RuntimeError("Completed items exceed the manifest total")
                if progress_callback is not None:
                    progress_callback(
                        self._snapshot_locked(SnapshotTransferPhase.ORDERED_MERGE)
                    )

        def record_completed(self) -> None:
            with self._lock:
                # A successful merge reveals the exact item total even when a
                # prior resume used an older cache state without item metadata.
                self._total_items = self._completed_items
                if progress_callback is not None:
                    progress_callback(
                        self._snapshot_locked(SnapshotTransferPhase.COMPLETED)
                    )

    progress = _SnapshotProgressReporter()
    progress.emit(SnapshotTransferPhase.CHUNK_TRANSFER)

    def _ensure_chunk_cache(plan: _ChunkPlan) -> None:
        _raise_if_cancelled()
        cache_is_valid = False
        if os.path.exists(plan.cache_path) and os.path.isfile(plan.cache_path):
            cache_is_valid = True
            if plan.content_length is not None:
                cache_is_valid = os.path.getsize(plan.cache_path) == plan.content_length

        if cache_is_valid:
            progress.record_verified_chunk(plan.index, os.path.getsize(plan.cache_path))
            return

        chunk_tmp_path = f"{plan.cache_path}.download"

        def _record_network_bytes(bytes_amount: int) -> None:
            progress.record_network_bytes(bytes_amount)
            _raise_if_cancelled()

        try:
            _raise_if_cancelled()
            s3_client.download_file(
                bucket,
                plan.chunk_key,
                chunk_tmp_path,
                Callback=_record_network_bytes,
            )
            _raise_if_cancelled()
            os.replace(chunk_tmp_path, plan.cache_path)
        finally:
            if os.path.exists(chunk_tmp_path):
                try:
                    os.remove(chunk_tmp_path)
                except OSError:
                    pass

        if plan.content_length is not None and os.path.getsize(plan.cache_path) != plan.content_length:
            raise RuntimeError(f"Downloaded chunk size mismatch: s3://{bucket}/{plan.chunk_key}")
        progress.record_verified_chunk(plan.index, os.path.getsize(plan.cache_path))

    download_progress = tqdm(total=len(chunks_to_download), unit="chunk", desc="chunk cache")
    download_cancelled = False
    download_error: Optional[str] = None
    try:
        if chunks_to_download:
            with ThreadPoolExecutor(max_workers=download_workers) as download_pool:
                future_map: Dict[Future[None], int] = {}
                next_download_plan = 0

                def _fill_download_queue() -> None:
                    nonlocal next_download_plan
                    while (
                        next_download_plan < len(chunks_to_download)
                        and len(future_map) < prefetch_chunks
                        and not _cancellation_requested()
                    ):
                        plan = chunks_to_download[next_download_plan]
                        next_download_plan += 1
                        future_map[download_pool.submit(_ensure_chunk_cache, plan)] = plan.index

                _fill_download_queue()
                while future_map:
                    if _cancellation_requested():
                        download_cancelled = True
                        for pending in future_map:
                            pending.cancel()

                    future = next(as_completed(future_map))
                    index = future_map.pop(future)
                    try:
                        future.result()
                    except _SnapshotMaterializationCancelled:
                        download_cancelled = True
                    except CancelledError:
                        download_cancelled = True
                    except Exception as exc:
                        download_error = (
                            "failed to cache chunk "
                            f"{index + 1}/{len(chunk_paths)}: {exc}"
                        )
                    else:
                        download_progress.update(1)

                    if download_cancelled or download_error is not None:
                        for pending in future_map:
                            pending.cancel()
                    else:
                        _fill_download_queue()
    finally:
        download_progress.close()

    if download_cancelled or _cancellation_requested():
        return _result(SnapshotMaterializationResult.CANCELLED)
    if download_error is not None:
        print(f"[WARN] Snapshot download failed: {download_error}")
        return _result(SnapshotMaterializationResult.FAILED)

    system_tar_path: Optional[str] = None
    use_system_tar = False
    if prefer_system_tar:
        if not sys.platform.startswith("linux"):
            print(
                "[WARN] Fast extractor is optimized for Linux; falling back to Python tarfile extraction."
            )
        else:
            system_tar_path = shutil.which("tar")
            if system_tar_path is None:
                print(
                    "[WARN] 'tar' not found; falling back to Python tarfile extraction (slower)."
                )
            else:
                use_system_tar = True

    staging_root = os.path.join(cache_root, "staging")
    os.makedirs(staging_root, exist_ok=True)

    fallback_warned = False
    fallback_warn_lock = Lock()

    def _prepare_chunk(plan: _ChunkPlan) -> _PreparedChunk:
        nonlocal fallback_warned
        _raise_if_cancelled()
        inspection = _inspect_chunk_tar(
            plan.cache_path,
            cancellation_requested=_cancellation_requested,
        )
        _raise_if_cancelled()
        staging_dir = os.path.join(staging_root, f"{plan.index:06d}")

        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

        try:
            _raise_if_cancelled()
            if use_system_tar and system_tar_path is not None:
                try:
                    _extract_chunk_with_system_tar(
                        system_tar_path=system_tar_path,
                        chunk_tar_path=plan.cache_path,
                        staging_dir=staging_dir,
                        cancellation_requested=_cancellation_requested,
                    )
                except _SnapshotMaterializationCancelled:
                    raise
                except Exception:
                    with fallback_warn_lock:
                        if not fallback_warned:
                            print(
                                "[WARN] Fast extraction via system tar failed; falling back to Python tarfile extraction."
                        )
                            fallback_warned = True
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    os.makedirs(staging_dir, exist_ok=True)
                    _extract_chunk_with_python_tar(
                        plan.cache_path,
                        staging_dir,
                        cancellation_requested=_cancellation_requested,
                    )
            else:
                _extract_chunk_with_python_tar(
                    plan.cache_path,
                    staging_dir,
                    cancellation_requested=_cancellation_requested,
                )
            _raise_if_cancelled()
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        return _PreparedChunk(
            index=plan.index,
            chunk_path=plan.chunk_path,
            chunk_key=plan.chunk_key,
            cache_path=plan.cache_path,
            content_length=plan.content_length,
            etag=plan.etag,
            staging_dir=staging_dir,
            file_paths=inspection.file_paths,
            manifest_blob=inspection.manifest_blob,
            item_count=inspection.item_count,
        )

    max_pending_prepared = max(prefetch_chunks, extract_workers * 2)
    pending_futures: Dict[int, Future[_PreparedChunk]] = {}
    next_plan_to_submit = 0

    def _fill_pending(extract_pool: ThreadPoolExecutor) -> None:
        nonlocal next_plan_to_submit
        while (
            next_plan_to_submit < len(chunk_plans)
            and len(pending_futures) < max_pending_prepared
            and not _cancellation_requested()
        ):
            plan = chunk_plans[next_plan_to_submit]
            next_plan_to_submit += 1
            if plan.skip_processing:
                continue
            pending_futures[plan.index] = extract_pool.submit(_prepare_chunk, plan)

    stream_error: Optional[str] = None
    stream_cancelled = False
    chunk_progress = tqdm(total=len(chunk_paths), unit="chunk", desc="chunks")
    try:
        with ThreadPoolExecutor(max_workers=extract_workers) as extract_pool:
            progress.emit(SnapshotTransferPhase.EXTRACTION)
            _fill_pending(extract_pool)
            progress.emit(SnapshotTransferPhase.ORDERED_MERGE)
            with open(merged_manifest_path, "ab") as merged_manifest:
                for plan in chunk_plans:
                    if _cancellation_requested():
                        stream_cancelled = True
                        break
                    if plan.skip_processing:
                        chunk_progress.update(1)
                        _fill_pending(extract_pool)
                        continue

                    future = pending_futures.pop(plan.index, None)
                    if future is None:
                        stream_error = f"Missing chunk prepare result for index {plan.index}"
                        break

                    try:
                        prepared = future.result()
                        progress.record_extracted_items(
                            prepared.index,
                            prepared.item_count,
                        )
                    except _SnapshotMaterializationCancelled:
                        stream_cancelled = True
                        break
                    except CancelledError:
                        stream_cancelled = True
                        break
                    except Exception as exc:
                        stream_error = (
                            f"Failed to extract chunk {plan.index + 1}/{len(chunk_paths)}: {exc}"
                        )
                        break

                    try:
                        _merge_staged_chunk(
                            staging_dir=prepared.staging_dir,
                            local_dir=local_dir,
                            file_paths=prepared.file_paths,
                        verify_existing_files=verify_existing_files,
                        cancellation_requested=_cancellation_requested,
                    )
                        _raise_if_cancelled()
                        merged_manifest.write(prepared.manifest_blob)
                        merged_manifest.flush()
                        os.fsync(merged_manifest.fileno())
                        _raise_if_cancelled()
                    except _SnapshotMaterializationCancelled:
                        stream_cancelled = True
                        break
                    finally:
                        shutil.rmtree(prepared.staging_dir, ignore_errors=True)

                    existing_state["merged_manifest_bytes"] = (
                        int(existing_state.get("merged_manifest_bytes", 0))
                        + len(prepared.manifest_blob)
                    )
                    completed_chunks[str(plan.index)] = {
                        "chunk_path": plan.chunk_path,
                        "chunk_key": plan.chunk_key,
                        "etag": plan.etag,
                        "content_length": plan.content_length,
                    }
                    _save_json_atomic(state_path, existing_state)
                    progress.record_merged_chunk(prepared.item_count)
                    chunk_progress.update(1)
                    _fill_pending(extract_pool)

            if stream_error is not None or stream_cancelled:
                for future in pending_futures.values():
                    future.cancel()
    finally:
        chunk_progress.close()
        shutil.rmtree(staging_root, ignore_errors=True)

    if stream_error is not None:
        print(f"[WARN] Snapshot download failed: {stream_error}")
        return _result(SnapshotMaterializationResult.FAILED)

    if stream_cancelled or _cancellation_requested():
        return _result(SnapshotMaterializationResult.CANCELLED)

    progress.record_completed()
    return _result(SnapshotMaterializationResult.SUCCESS)
