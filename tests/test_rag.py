"""Offline checks for retrieval and citation safeguards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from rag import answer, retrieve


class RetrievalTests(unittest.TestCase):
    def test_persistent_retrieval_and_reindex_replaces_old_documents(self):
        from vector_store import build_index, search
        from chromadb.api.types import EmbeddingFunction

        class TestEmbedding(EmbeddingFunction):
            def __init__(self):
                pass

            def __call__(self, input):
                return [[1., 0.] if "smoke" in text.lower() else [0., 1.] for text in input]

            @staticmethod
            def name():
                return "offline-test"

        embedding = TestEmbedding()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            chunks = [{"text": "Smoke detectors", "page": 7, "title": "SAME", "experiment": "SAME"},
                      {"text": "Droplets burn", "page": 2, "title": "FLEX", "experiment": "FLEX"}]
            build_index(chunks, path, embedding)
            matches = search("smoke", path=path, embedding_function=embedding, max_distance=0.5)
            self.assertEqual([m["page"] for m in matches], [7])
            explicit = search("FLEX smoke", path=path, embedding_function=embedding)
            self.assertEqual([m["experiment"] for m in explicit], ["FLEX"])
            build_index(chunks[1:], path, embedding)
            self.assertEqual(search("smoke", path=path, embedding_function=embedding,
                                    max_distance=0.5), [])

    def test_missing_index_and_empty_evidence_do_not_call_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(retrieve("smoke", index_path=Path(directory) / "absent"), [])
        with patch("rag.urlopen") as network:
            self.assertIn("Insufficient", answer("question", [], {}, "", ""))
            network.assert_not_called()

    def test_unknown_citation_is_rejected(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Claim [S9]"}}]})
        with patch("rag.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "citations"):
                answer("smoke", [{"text": "Smoke"}], {}, "test-key", "test-model")

    def test_groq_request_and_valid_answer(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Smoke evidence [S1]"}}]})
        with patch("rag.urlopen", return_value=response) as network:
            self.assertEqual(answer("smoke", [{"text": "Smoke"}], {}, "test-key",
                                    "provider/model"), "Smoke evidence [S1]")
            request = network.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "provider/model")
            self.assertEqual(payload["messages"][0]["role"], "system")

    def test_provider_error_does_not_expose_credentials(self):
        from urllib.error import HTTPError
        with patch("rag.urlopen", side_effect=HTTPError("url", 401, "secret", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "Groq request failed.*401"):
                answer("smoke", [{"text": "Smoke"}], {}, "test-key", "model")

    def test_forbidden_error_includes_redacted_provider_message(self):
        import io
        from urllib.error import HTTPError
        body = io.BytesIO(json.dumps({"error": {"message": "Model denied: gsk_secret test-key"}}).encode())
        with patch("rag.urlopen", side_effect=HTTPError("url", 403, "Forbidden", {}, body)):
            with self.assertRaises(RuntimeError) as caught:
                answer("smoke", [{"text": "Smoke"}], {}, "test-key", "model")
        message = str(caught.exception)
        self.assertIn("Model denied", message)
        self.assertNotIn("gsk_secret", message)
        self.assertNotIn("test-key", message)


if __name__ == "__main__":
    unittest.main()
