from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloud_index


class CloudIndexTests(unittest.TestCase):
    def test_concurrent_sessions_build_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def ingest(progress):
                progress("Building")
                (root / "active.json").write_text('{}')
                (root / "chroma.sqlite3").touch()

            with patch.object(cloud_index, "STORE", root), patch(
                    "scripts.ingest_documents.ingest", side_effect=ingest) as build:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: cloud_index.ensure_index(lambda message: None), range(2)))
                self.assertEqual(sorted(results), [False, True])
                build.assert_called_once()

    def test_failed_build_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def success(progress):
                (root / "active.json").write_text('{}')
                (root / "chroma.sqlite3").touch()

            with patch.object(cloud_index, "STORE", root), patch(
                    "scripts.ingest_documents.ingest", side_effect=RuntimeError("Download failed")):
                with self.assertRaises(RuntimeError):
                    cloud_index.ensure_index(lambda message: None)
            with patch.object(cloud_index, "STORE", root), patch(
                    "scripts.ingest_documents.ingest", side_effect=success):
                self.assertTrue(cloud_index.ensure_index(lambda message: None))
