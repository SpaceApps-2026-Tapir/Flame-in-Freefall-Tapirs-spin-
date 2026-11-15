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
            chunks = [{"text": "Smoke detectors " * 15, "page": 7, "title": "SAME", "experiment": "SAME"},
                      {"text": "Droplets burn " * 15, "page": 2, "title": "FLEX", "experiment": "FLEX"}]
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

    def test_index_inventory_counts_passages_by_pdf(self):
        from vector_store import index_inventory
        collection = MagicMock()
        collection.get.return_value = {"metadatas": [
            {"document_id": "bass", "title": "BASS report", "experiment": "BASS",
             "source_page_url": "https://example.com/bass"},
            {"document_id": "flex", "title": "FLEX report", "experiment": "FLEX",
             "pdf_url": "https://example.com/flex.pdf"},
            {"document_id": "bass", "title": "BASS report", "experiment": "BASS"},
            {"title": "Unassigned", "experiment": ""},
            {"title": "Unassigned", "experiment": ""},
        ]}
        client = MagicMock()
        client.get_collection.return_value = collection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "active.json").write_text(json.dumps({"collection": "test"}))
            with patch("chromadb.PersistentClient", return_value=client):
                rows = index_inventory(path)
        self.assertEqual([(r["title"], r["experiment"], r["url"], r["passages"]) for r in rows],
                         [("BASS report", "BASS", "https://example.com/bass", 2),
                          ("FLEX report", "FLEX", "https://example.com/flex.pdf", 1),
                          ("Unassigned", "", "", 2)])

    def test_index_inventory_without_active_index_is_empty(self):
        from vector_store import index_inventory
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(index_inventory(Path(directory) / "absent"), [])

    def test_retrieval_skips_title_slides_and_contents(self):
        from vector_store import search
        collection = MagicMock()
        collection.count.return_value = 3
        metadata = [{"title": "Report", "document_id": "doc", "page": p} for p in (1, 2, 3)]
        documents = ["Science overview", "Contents ............ variables " * 15,
                     "Investigated variables include oxygen concentration and ventilation velocity. " * 5]
        collection.query.return_value = {"ids": [["0", "1", "2"]], "distances": [[0.1, 0.2, 0.3]]}
        collection.get.return_value = {"ids": ["0", "1", "2"], "documents": documents, "metadatas": metadata}
        client = MagicMock()
        client.get_collection.return_value = collection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "active.json").write_text(json.dumps({"collection": "test"}))
            with patch("vector_store.client_and_embedding", return_value=(client, None)):
                matches = search("What variables were investigated?", path=path)
        self.assertEqual([p["page"] for p in matches], [3])

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

    def test_invalid_answer_is_retried_with_allowed_sources(self):
        with patch("rag._generate", side_effect=["Uncited answer", "Supported [S1]"]) as generate:
            self.assertEqual(answer("q", [{"text": "evidence"}], {}, "key", "model"), "Supported [S1]")
            self.assertEqual(generate.call_count, 2)

    def test_grouped_citations_and_insufficient_evidence(self):
        with patch("rag._generate", return_value="Supported [S1, S2]"):
            self.assertEqual(answer("q", [{}, {}], {}, "key", "model"), "Supported [S1] [S2]")
        with patch("rag._generate", return_value="INSUFFICIENT_EVIDENCE") as generate:
            self.assertIn("not contain enough evidence", answer("q", [{}], {}, "key", "model"))
            self.assertEqual(generate.call_count, 1)

    def test_request_uses_temperature_zero(self):
        with patch("rag._generate", return_value="Supported [S1]") as generate:
            answer("q", [{"text": "evidence"}], {}, "key", "model")
        self.assertEqual(generate.call_args.args[0]["temperature"], 0)

    def test_answer_citing_only_a_context_test_is_accepted(self):
        context = {"ranked_tests": [{"experiment": "FLEX", "test_id": "219F001"}], "gaps": []}
        with patch("rag._generate", return_value="The closest test is FLEX 219F001.") as generate:
            self.assertEqual(answer("q", [{}], context, "key", "model"), "The closest test is FLEX 219F001.")
            self.assertEqual(generate.call_count, 1)

    def test_word_same_is_not_the_same_experiment(self):
        from vector_store import experiments_in
        self.assertEqual(experiments_in("Is extinction the same at high oxygen?")[0], set())
        self.assertEqual(experiments_in("What did SAME and SAME-R measure?")[0], {"SAME", "SAME-R"})
        self.assertEqual(experiments_in("bass-ii vs Flex")[0], {"BASS-II", "FLEX"})

    def test_reference_lists_are_detected(self):
        from vector_store import is_reference_list
        self.assertTrue(is_reference_list("1Bukowski, R.W. and G.W. Mulholland, 1978. 2Bukowski, R. W, R. D. Peacock"))
        self.assertTrue(is_reference_list("12. Olson, Sandra L.; Ferkul, Paul V. 13. Ito, Akihiko; and Kashiwagi"))
        self.assertFalse(is_reference_list("The flame spread at 21% O2 and went out when the flow was reduced."))

    def test_short_last_chunk_of_a_page_is_kept_with_its_neighbour(self):
        from vector_store import search
        collection = MagicMock()
        collection.count.return_value = 2
        metadata = [{"title": "Report", "document_id": "doc", "page": 4}] * 2
        documents = ["Tests were performed aboard the ISS in the wind tunnel. " * 4, "tests varying oxygen below 21%"]
        collection.query.return_value = {"ids": [["1", "0"]], "distances": [[0.1, 0.3]]}
        collection.get.return_value = {"ids": ["0", "1"], "documents": documents, "metadatas": metadata}
        client = MagicMock()
        client.get_collection.return_value = collection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "active.json").write_text(json.dumps({"collection": "test"}))
            with patch("vector_store.client_and_embedding", return_value=(client, None)):
                matches = search("oxygen", path=path)
        self.assertEqual(len(matches), 1)
        self.assertIn("varying oxygen", matches[0]["text"])

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


class ReferenceMetadataTests(unittest.TestCase):
    def test_reference_metadata_is_complete_and_ieee(self):
        from references import load_references, validate

        self.assertEqual(validate(), [])
        refs = {ref["ref_id"]: ref for ref in load_references()}
        self.assertIn("Bhattacharjee", refs["ref-psi-25"]["authors"])
        self.assertNotIn("Bhattachjee", refs["ref-psi-25"]["authors"])
        self.assertIn("Fumiaki Takahashi", refs["ref-psi-26"]["authors"])
        self.assertEqual(len(refs["ref-tm-20210011385"]["authors"].split(";")), 8)
        self.assertEqual(refs["ref-psi-25"]["license"], "CC0-1.0")
        self.assertEqual(refs["ref-tm-20210011385"]["license"], "NASA public use permitted")

        for ref in refs.values():
            self.assertTrue(ref["authors"])
            self.assertTrue(ref["license"])
            self.assertIn('"', ref["citation"])
            self.assertIn("Available:", ref["citation"])
            self.assertNotIn("(20", ref["citation"])


if __name__ == "__main__":
    unittest.main()
