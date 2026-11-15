"""Offline checks for the small security helpers used by the UI and exports."""
import unittest

import pandas as pd

from security import (MAX_QUESTION_CHARS, csv_bytes, csv_safe_cell,
                      safe_http_url, safe_model_markdown, validate_question)


class SecurityHelperTests(unittest.TestCase):
    def test_safe_http_url_accepts_only_http_and_https(self):
        for url in ("https://drive.google.com/file/d/example/view",
                    "http://example.com/report.pdf?x=1#page=2"):
            with self.subTest(url=url):
                self.assertEqual(safe_http_url(url), url)
        for url in ("javascript:alert(1)", "data:text/html,<script>alert(1)</script>",
                    "file:///etc/passwd", "//example.com/report", "/relative/path",
                    "https://user:password@example.com/report", "https://example.com/a\nb",
                    'https://example.com/" onmouseover="alert(1)',
                    "https://example.com/" + "a" * 2048, "", None, 42):
            with self.subTest(url=url):
                self.assertEqual(safe_http_url(url), "")

    def test_question_length_is_bounded(self):
        self.assertIsNone(validate_question("x" * MAX_QUESTION_CHARS))
        self.assertIn("too long", validate_question("x" * (MAX_QUESTION_CHARS + 1)))
        self.assertIsNone(validate_question(""))

    def test_model_markdown_drops_links_and_images_but_keeps_citations(self):
        self.assertEqual(safe_model_markdown("Supported [S1] and [S3]."),
                         "Supported [S1] and [S3].")
        self.assertEqual(safe_model_markdown("See [the source](javascript:alert(1)) [S1]."),
                         "See the source [S1].")
        self.assertEqual(safe_model_markdown("![tracking](https://example.com/pixel.png) text"),
                         "tracking text")

    def test_csv_text_cells_neutralise_formula_prefixes(self):
        for value in ("=1+1", "+cmd", "-2", "@SUM(A1)", "\tformula", "\rformula"):
            with self.subTest(value=value):
                self.assertEqual(csv_safe_cell(value), "'" + value)
        self.assertEqual(csv_safe_cell("plain text"), "plain text")
        self.assertEqual(csv_safe_cell(-2.0), -2.0)

    def test_csv_export_keeps_numeric_columns_but_escapes_text(self):
        frame = pd.DataFrame({"note": ["=SUM(1,1)", "ok", "-2", "@x"],
                              "value": [1, 2, 3, 4]})
        exported = csv_bytes(frame).decode("utf-8")
        self.assertIn("'=SUM(1,1)", exported)
        self.assertIn("'-2", exported)
        self.assertIn("'@x", exported)
        self.assertIn(",1\n", exported)
        self.assertIn(",4\n", exported)


if __name__ == "__main__":
    unittest.main()
