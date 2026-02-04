import hashlib
import importlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import threading
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
            Callback(src.stat().st_size)


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

    def test_returns_false_when_chunk_has_no_manifest_jsonl(self):
        manifest_key = "snapshots/missing_manifest/manifest.json"
        _write_chunk_tar(
            self._s3_object_path("snapshots/missing_manifest/chunks/000000.tar"),
            manifest_lines=None,
            files={"images/rule34:1.png": b"img-1"},
        )
        self._write_snapshot_manifest(manifest_key, ["chunks/000000.tar"])

        out_dir = self.root / "out_missing_manifest"
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
        )
        self.assertTrue(second)
        merged_twice = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")
        self.assertEqual(merged_once, merged_twice)

        chunk_downloads = [key for _, key, _ in self.client.download_calls if key == chunk_key]
        self.assertEqual(len(chunk_downloads), 1)

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
