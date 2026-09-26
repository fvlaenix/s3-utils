import hashlib
import importlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


def _import_s3_module():
    try:
        import s3 as module  # type: ignore

        return module
    except ModuleNotFoundError as exc:
        if exc.name not in {"boto3", "botocore", "tqdm"}:
            raise

    sys.modules.pop("s3", None)

    if "boto3" not in sys.modules:
        boto3_mod = types.ModuleType("boto3")

        class _Session:
            def __init__(self, **_kwargs):
                pass

            def client(self, *_args, **_kwargs):
                raise RuntimeError("client() should be patched in tests")

        boto3_mod.Session = _Session
        sys.modules["boto3"] = boto3_mod

    if "botocore.client" not in sys.modules:
        botocore_mod = types.ModuleType("botocore")
        botocore_client_mod = types.ModuleType("botocore.client")
        botocore_exceptions_mod = types.ModuleType("botocore.exceptions")

        class _Config:
            def __init__(self, **_kwargs):
                pass

        class _BotoCoreError(Exception):
            pass

        class _ClientError(Exception):
            pass

        botocore_client_mod.Config = _Config
        botocore_exceptions_mod.BotoCoreError = _BotoCoreError
        botocore_exceptions_mod.ClientError = _ClientError
        sys.modules["botocore"] = botocore_mod
        sys.modules["botocore.client"] = botocore_client_mod
        sys.modules["botocore.exceptions"] = botocore_exceptions_mod

    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")

        class _Tqdm:
            def __init__(self, *_args, **_kwargs):
                pass

            def update(self, *_args, **_kwargs):
                pass

            def close(self):
                pass

        tqdm_mod.tqdm = _Tqdm
        sys.modules["tqdm"] = tqdm_mod

    return importlib.import_module("s3")


s3 = _import_s3_module()


class _DummyTqdm:
    def __init__(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


class _FakeS3Client:
    def __init__(self, s3_root: Path):
        self._s3_root = s3_root
        self._lock = threading.Lock()
        self.download_calls: list[tuple[str, str, str]] = []
        self.callback_parts_by_key: dict[str, list[int]] = {}
        self.callback_barrier_by_key: dict[str, threading.Barrier] = {}

    def _object_path(self, bucket: str, key: str) -> Path:
        return self._s3_root / bucket / key

    def head_object(self, Bucket: str, Key: str):
        path = self._object_path(Bucket, Key)
        if not path.exists():
            raise FileNotFoundError(f"Missing object: s3://{Bucket}/{Key}")
        content = path.read_bytes()
        etag = f"\"{hashlib.md5(content).hexdigest()}\""
        return {"ContentLength": len(content), "ETag": etag}

    def download_file(self, bucket: str, key: str, dest_path: str, Callback=None):
        src = self._object_path(bucket, key)
        if not src.exists():
            raise FileNotFoundError(f"Missing object: s3://{bucket}/{key}")

        os_dest = Path(dest_path)
        os_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, os_dest)

        with self._lock:
            self.download_calls.append((bucket, key, dest_path))

        if Callback is not None:
            barrier = self.callback_barrier_by_key.get(key)
            if barrier is not None:
                barrier.wait(timeout=2)
            for byte_count in self.callback_parts_by_key.get(
                key,
                [src.stat().st_size],
            ):
                Callback(byte_count)


def _write_chunk_tar(path: Path, *, manifest_lines: list[str] | None, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        if manifest_lines is not None:
            data = "".join(f"{line.rstrip()}\n" for line in manifest_lines).encode("utf-8")
            manifest_info = tarfile.TarInfo("manifest.jsonl")
            manifest_info.size = len(data)
            tar.addfile(manifest_info, io.BytesIO(data))

        for rel_path, payload in files.items():
            info = tarfile.TarInfo(rel_path)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


class DownloadSnapshotFromS3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bucket = "bucket"
        self.s3_root = self.root / "fake_s3"
        (self.s3_root / self.bucket).mkdir(parents=True, exist_ok=True)
        self.cfg = s3.S3Config()
        self.client = _FakeS3Client(self.s3_root)

        self._patches = [
            mock.patch.object(s3, "create_s3_client", return_value=self.client),
            mock.patch.object(s3, "tqdm", _DummyTqdm),
        ]
        for patcher in self._patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _s3_object_path(self, key: str) -> Path:
        path = self.s3_root / self.bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_snapshot_manifest(self, manifest_key: str, chunk_paths: list[str]) -> None:
        payload = {"snapshot_name": "test_snapshot", "chunk_paths": chunk_paths}
        self._s3_object_path(manifest_key).write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def test_merges_images_tags_scores_and_manifest_in_chunk_order(self):
        manifest_key = "snapshots/case1/manifest.json"
        chunk1_rel = "chunks/000000.tar"
        chunk2_rel = "chunks/000001.tar"

        _write_chunk_tar(
            self._s3_object_path("snapshots/case1/chunks/000000.tar"),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png","tags_path":"tags/rule34:1.txt","scores_path":"scores/rule34:1.jsonl"}'
            ],
            files={
                "images/rule34:1.png": b"img-1",
                "tags/rule34:1.txt": b"tag-one\nlong hair\n",
                "scores/rule34:1.jsonl": b'{"tag":"tag-one","score":0.9}\n',
            },
        )
        _write_chunk_tar(
            self._s3_object_path("snapshots/case1/chunks/000001.tar"),
            manifest_lines=[
                '{"external_id":"rule34:2","image_path":"images/rule34:2.png"}',
            ],
            files={"images/rule34:2.png": b"img-2"},
        )
        self._write_snapshot_manifest(manifest_key, [chunk1_rel, chunk2_rel])

        out_dir = self.root / "out_case1"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=2,
            extract_workers=2,
        )
        self.assertTrue(result)

        self.assertEqual((out_dir / "images/rule34:1.png").read_bytes(), b"img-1")
        self.assertEqual((out_dir / "images/rule34:2.png").read_bytes(), b"img-2")
        self.assertEqual((out_dir / "tags/rule34:1.txt").read_text(encoding="utf-8"), "tag-one\nlong hair\n")
        self.assertEqual(
            (out_dir / "scores/rule34:1.jsonl").read_text(encoding="utf-8"),
            '{"tag":"tag-one","score":0.9}\n',
        )

        merged_manifest = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")
        self.assertEqual(
            merged_manifest,
            (
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png","tags_path":"tags/rule34:1.txt","scores_path":"scores/rule34:1.jsonl"}\n'
                '{"external_id":"rule34:2","image_path":"images/rule34:2.png"}\n'
            ),
        )

    def test_progress_reports_typed_phases_retry_bytes_and_pre_completion_state(self):
        manifest_key = "snapshots/progress/manifest.json"
        chunk_key = "snapshots/progress/chunks/000000.tar"
        _write_chunk_tar(
            self._s3_object_path(chunk_key),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.png"}'],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])
        chunk_size = self._s3_object_path(chunk_key).stat().st_size
        # boto callbacks are deltas. Repeating a delta models retried network
        # traffic without allowing verified snapshot progress to double count.
        self.client.callback_parts_by_key[chunk_key] = [chunk_size, chunk_size]

        events: list[s3.SnapshotTransferProgress] = []
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(self.root / "out_progress"),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
            operation_id="operation-479",
            attempt_id="attempt-1",
            progress_callback=events.append,
        )

        self.assertTrue(result)
        self.assertTrue(
            {
                s3.SnapshotTransferPhase.MANIFEST_READ,
                s3.SnapshotTransferPhase.CHUNK_TRANSFER,
                s3.SnapshotTransferPhase.EXTRACTION,
                s3.SnapshotTransferPhase.ORDERED_MERGE,
                s3.SnapshotTransferPhase.COMPLETED,
            }.issubset({event.phase for event in events})
        )
        self.assertEqual(events[-1].phase, s3.SnapshotTransferPhase.COMPLETED)
        self.assertEqual(events[-1].operation_id, "operation-479")
        self.assertEqual(events[-1].attempt_id, "attempt-1")
        self.assertEqual(events[-1].network_bytes, chunk_size * 2)
        self.assertEqual(events[-1].verified_bytes, chunk_size)
        self.assertEqual(events[-1].total_bytes, chunk_size)
        self.assertEqual(events[-1].completed_chunks, 1)
        self.assertEqual(events[-1].total_chunks, 1)
        self.assertEqual(events[-1].completed_items, 1)
        self.assertEqual(events[-1].total_items, 1)
        self.assertTrue(
            any(
                event.phase == s3.SnapshotTransferPhase.CHUNK_TRANSFER
                and event.network_bytes > 0
                and event.completed_chunks == 0
                for event in events
            ),
            "transfer progress must be observable before terminal completion",
        )
        for event in events:
            self.assertLessEqual(event.completed_chunks, event.total_chunks)
            if event.verified_bytes is not None and event.total_bytes is not None:
                self.assertLessEqual(event.verified_bytes, event.total_bytes)
            if event.total_items is not None:
                self.assertLessEqual(event.completed_items, event.total_items)

    def test_progress_callbacks_are_serialized_and_monotone_for_concurrent_chunks(self):
        manifest_key = "snapshots/concurrent_progress/manifest.json"
        chunk_keys = [
            "snapshots/concurrent_progress/chunks/000000.tar",
            "snapshots/concurrent_progress/chunks/000001.tar",
        ]
        for index, chunk_key in enumerate(chunk_keys, start=1):
            _write_chunk_tar(
                self._s3_object_path(chunk_key),
                manifest_lines=[
                    f'{{"external_id":"rule34:{index}","image_path":"images/rule34:{index}.png"}}'
                ],
                files={f"images/rule34:{index}.png": f"img-{index}".encode()},
            )
        self._write_snapshot_manifest(
            manifest_key,
            ["chunks/000000.tar", "chunks/000001.tar"],
        )
        barrier = threading.Barrier(2)
        for chunk_key in chunk_keys:
            self.client.callback_barrier_by_key[chunk_key] = barrier

        events: list[s3.SnapshotTransferProgress] = []
        callback_lock = threading.Lock()
        active_callbacks = 0
        max_active_callbacks = 0

        def capture(event: s3.SnapshotTransferProgress) -> None:
            nonlocal active_callbacks, max_active_callbacks
            with callback_lock:
                active_callbacks += 1
                max_active_callbacks = max(max_active_callbacks, active_callbacks)
            try:
                time.sleep(0.01)
                events.append(event)
            finally:
                with callback_lock:
                    active_callbacks -= 1

        self.assertTrue(
            s3.download_snapshot_from_s3(
                self.cfg,
                self.bucket,
                None,
                manifest_key,
                str(self.root / "out_concurrent_progress"),
                prefer_system_tar=False,
                download_workers=2,
                extract_workers=2,
                progress_callback=capture,
            )
        )

        self.assertEqual(max_active_callbacks, 1)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[-1].phase, s3.SnapshotTransferPhase.COMPLETED)
        for previous, current in zip(events, events[1:]):
            self.assertLessEqual(previous.network_bytes, current.network_bytes)
            self.assertLessEqual(previous.completed_chunks, current.completed_chunks)
            self.assertLessEqual(previous.completed_items, current.completed_items)
            if previous.verified_bytes is not None and current.verified_bytes is not None:
                self.assertLessEqual(previous.verified_bytes, current.verified_bytes)
            self.assertEqual(previous.total_chunks, current.total_chunks)
            if previous.total_bytes is not None and current.total_bytes is not None:
                self.assertEqual(previous.total_bytes, current.total_bytes)
            if previous.total_items is not None and current.total_items is not None:
                self.assertEqual(previous.total_items, current.total_items)

    def test_progress_keeps_byte_total_unknown_when_chunk_heads_are_unavailable(self):
        manifest_key = "snapshots/unknown_total/manifest.json"
        chunk_key = "snapshots/unknown_total/chunks/000000.tar"
        _write_chunk_tar(
            self._s3_object_path(chunk_key),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.png"}'],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])
        real_head_object = s3._head_object

        def head_without_chunk_size(client, bucket, key):
            if key == chunk_key:
                return None
            return real_head_object(client, bucket, key)

        events: list[s3.SnapshotTransferProgress] = []
        with mock.patch.object(s3, "_head_object", side_effect=head_without_chunk_size):
            self.assertTrue(
                s3.download_snapshot_from_s3(
                    self.cfg,
                    self.bucket,
                    None,
                    manifest_key,
                    str(self.root / "out_unknown_total"),
                    prefer_system_tar=False,
                    download_workers=1,
                    extract_workers=1,
                    progress_callback=events.append,
                )
            )

        self.assertTrue(events)
        self.assertTrue(all(event.total_bytes is None for event in events))
        self.assertEqual(
            events[-1].verified_bytes,
            self._s3_object_path(chunk_key).stat().st_size,
        )

    def test_returns_false_when_chunk_has_no_manifest_jsonl(self):
        manifest_key = "snapshots/missing_manifest/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/missing_manifest/chunks/000000.tar"),
            manifest_lines=None,
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_missing_manifest"
        events: list[s3.SnapshotTransferProgress] = []
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
            progress_callback=events.append,
        )
        self.assertFalse(result)
        self.assertNotIn(
            s3.SnapshotTransferPhase.COMPLETED,
            [event.phase for event in events],
        )
        self.assertTrue(events)
        self.assertTrue(all(event.completed_chunks == 0 for event in events))

    def test_fails_hard_on_cross_chunk_path_conflict(self):
        manifest_key = "snapshots/conflict/manifest.json"
        shared_path = "images/rule34:dup.png"

        _write_chunk_tar(
            self._s3_object_path("snapshots/conflict/chunks/000000.tar"),
            manifest_lines=['{"external_id":"rule34:dup","image_path":"images/rule34:dup.png"}'],
            files={shared_path: b"first"},
        )
        _write_chunk_tar(
            self._s3_object_path("snapshots/conflict/chunks/000001.tar"),
            manifest_lines=['{"external_id":"rule34:dup","image_path":"images/rule34:dup.png"}'],
            files={shared_path: b"second"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar", "chunks/000001.tar"])

        out_dir = self.root / "out_conflict"
        with self.assertRaises(FileExistsError):
            s3.download_snapshot_from_s3(
                self.cfg,
                self.bucket,
                None,
                manifest_key,
                str(out_dir),
                prefer_system_tar=False,
                download_workers=2,
                extract_workers=2,
            )

    def test_resume_skips_completed_chunk_and_does_not_append_manifest_twice(self):
        manifest_key = "snapshots/resume/manifest.json"
        chunk_key = "snapshots/resume/chunks/000000.tar"
        chunk_rel = "chunks/000000.tar"

        _write_chunk_tar(
            self._s3_object_path(chunk_key),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.png"}'],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, [chunk_rel])

        out_dir = self.root / "out_resume"
        first = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertTrue(first)
        merged_once = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")

        resume_events: list[s3.SnapshotTransferProgress] = []
        second = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            resume=True,
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
            progress_callback=resume_events.append,
        )
        self.assertTrue(second)
        merged_twice = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")
        self.assertEqual(merged_once, merged_twice)

        chunk_downloads = [key for _, key, _ in self.client.download_calls if key == chunk_key]
        self.assertEqual(len(chunk_downloads), 1)
        self.assertEqual(resume_events[-1].phase, s3.SnapshotTransferPhase.COMPLETED)
        self.assertTrue(all(event.network_bytes == 0 for event in resume_events))
        self.assertEqual(resume_events[-1].completed_chunks, 1)
        self.assertEqual(resume_events[-1].completed_items, 1)
        self.assertEqual(
            resume_events[-1].verified_bytes,
            self._s3_object_path(chunk_key).stat().st_size,
        )

    def test_resume_after_merge_before_state_commit_reuses_identical_files(self):
        manifest_key = "snapshots/interrupted_merge/manifest.json"
        chunk_key = "snapshots/interrupted_merge/chunks/000000.tar"
        _write_chunk_tar(
            self._s3_object_path(chunk_key),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png","tags_path":"tags/rule34:1.txt"}'
            ],
            files={
                "images/rule34:1.png": b"img-1",
                "tags/rule34:1.txt": b"tag-one\n",
            },
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])
        out_dir = self.root / "out_interrupted_merge"

        real_save = s3._save_json_atomic
        save_calls = 0

        def interrupt_after_merge(path, payload):
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                raise RuntimeError("simulated interruption before state commit")
            return real_save(path, payload)

        with mock.patch.object(s3, "_save_json_atomic", side_effect=interrupt_after_merge):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                s3.download_snapshot_from_s3(
                    self.cfg,
                    self.bucket,
                    None,
                    manifest_key,
                    str(out_dir),
                    prefer_system_tar=False,
                    download_workers=1,
                    extract_workers=1,
                )

        self.assertEqual((out_dir / "images/rule34:1.png").read_bytes(), b"img-1")
        self.assertEqual((out_dir / "tags/rule34:1.txt").read_bytes(), b"tag-one\n")

        self.assertTrue(
            s3.download_snapshot_from_s3(
                self.cfg,
                self.bucket,
                None,
                manifest_key,
                str(out_dir),
                resume=True,
                prefer_system_tar=False,
                download_workers=1,
                extract_workers=1,
            )
        )
        self.assertEqual(
            (out_dir / "manifest.jsonl").read_text(encoding="utf-8"),
            '{"external_id":"rule34:1","image_path":"images/rule34:1.png","tags_path":"tags/rule34:1.txt"}\n',
        )

    def test_resume_rejects_existing_file_with_same_size_but_different_bytes(self):
        manifest_key = "snapshots/interrupted_conflict/manifest.json"
        chunk_key = "snapshots/interrupted_conflict/chunks/000000.tar"
        _write_chunk_tar(
            self._s3_object_path(chunk_key),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.png"}'],
            files={"images/rule34:1.png": b"good"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])
        out_dir = self.root / "out_interrupted_conflict"
        (out_dir / "images").mkdir(parents=True)
        (out_dir / "images/rule34:1.png").write_bytes(b"evil")

        with self.assertRaises(FileExistsError):
            s3.download_snapshot_from_s3(
                self.cfg,
                self.bucket,
                None,
                manifest_key,
                str(out_dir),
                resume=True,
                prefer_system_tar=False,
                download_workers=1,
                extract_workers=1,
            )

    def test_rejects_absolute_chunk_paths_from_snapshot_manifest(self):
        manifest_key = "snapshots/abs/manifest.json"
        self._write_snapshot_manifest(manifest_key, ["/chunks/000000.tar"])

        out_dir = self.root / "out_abs"
        with self.assertRaises(ValueError):
            s3.download_snapshot_from_s3(
                self.cfg,
                self.bucket,
                None,
                manifest_key,
                str(out_dir),
                prefer_system_tar=False,
                download_workers=1,
                extract_workers=1,
            )

    def test_manifest_line_with_extra_field_is_rejected(self):
        manifest_key = "snapshots/extra_field/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/extra_field/chunks/000000.tar"),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png","extra_field":"bad"}'
            ],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_extra_field"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)

    def test_manifest_empty_line_is_rejected(self):
        manifest_key = "snapshots/empty_line/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/empty_line/chunks/000000.tar"),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png"}',
                "",
            ],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_empty_line"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)

    def test_manifest_missing_referenced_image_file_is_rejected(self):
        manifest_key = "snapshots/missing_image_ref/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/missing_image_ref/chunks/000000.tar"),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.png"}'],
            files={"images/rule34:other.png": b"img-other"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_missing_image_ref"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)

    def test_manifest_missing_referenced_tags_or_scores_file_is_rejected(self):
        manifest_key = "snapshots/missing_tags_scores_ref/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/missing_tags_scores_ref/chunks/000000.tar"),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png","tags_path":"tags/rule34:1.txt","scores_path":"scores/rule34:1.jsonl"}'
            ],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_missing_tags_scores_ref"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)

    def test_duplicate_external_id_in_single_chunk_is_rejected(self):
        manifest_key = "snapshots/dup_external_id/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/dup_external_id/chunks/000000.tar"),
            manifest_lines=[
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png"}',
                '{"external_id":"rule34:1","image_path":"images/rule34:1.png"}',
            ],
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_dup_external_id"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)

    def test_unsupported_image_extension_is_rejected(self):
        manifest_key = "snapshots/bad_ext/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/bad_ext/chunks/000000.tar"),
            manifest_lines=['{"external_id":"rule34:1","image_path":"images/rule34:1.gif"}'],
            files={"images/rule34:1.gif": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_bad_ext"
        result = s3.download_snapshot_from_s3(
            self.cfg,
            self.bucket,
            None,
            manifest_key,
            str(out_dir),
            prefer_system_tar=False,
            download_workers=1,
            extract_workers=1,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
