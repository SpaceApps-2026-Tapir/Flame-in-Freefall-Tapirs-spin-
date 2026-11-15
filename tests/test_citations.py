import unittest

from citations import ieee_citations


class CitationTests(unittest.TestCase):
    def test_first_appearance_document_numbering(self):
        passages = [
            {"document_id": "a", "title": "Report A", "page": 2, "pdf_url": "https://example.com/a"},
            {"document_id": "b", "title": "Report B", "page": 5, "pdf_url": "https://example.com/b"},
            {"document_id": "a", "title": "Report A", "page": 9, "pdf_url": "https://example.com/a"}]
        text, references, labels = ieee_citations("First [S2]. Next [S3]. Also [S1].", passages)
        self.assertEqual(text, "First [1, p. 5]. Next [2, p. 9]. Also [2, p. 2].")
        self.assertEqual([r["title"] for r in references], ["Report B", "Report A"])
        self.assertEqual(labels, {1: 1, 2: 2, 0: 2})

    def test_passages_without_answer_still_get_labels(self):
        text, references, labels = ieee_citations(None, [{"title": "Report", "page": 1}])
        self.assertEqual(text, "")
        self.assertEqual(labels, {0: 1})
        self.assertFalse(references[0]["cited"])
