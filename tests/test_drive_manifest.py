import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import sync_drive_manifest as drive


class DriveManifestTests(unittest.TestCase):
    def test_pdf_filter_and_experiment_metadata(self):
        files = [SimpleNamespace(id="one", path="reports/SAME-R_Final.pdf"),
                 SimpleNamespace(id="two", path="BASS-II_Report.PDF"),
                 SimpleNamespace(id="three", path="notes.txt")]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(drive, "ROOT", Path(directory)), patch.object(
                    drive.gdown, "download_folder", return_value=files):
                drive.sync()
                with (Path(directory) / "manifest.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([r["experiment"] for r in rows], ["SAME-R", "BASS-II"])
                self.assertIn("id=one", rows[0]["pdf_url"])

    def test_empty_folder_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("original")
            with patch.object(drive, "ROOT", Path(directory)), patch.object(
                    drive.gdown, "download_folder", return_value=[]):
                with self.assertRaises(ValueError):
                    drive.sync()
            self.assertEqual(manifest.read_text(), "original")
