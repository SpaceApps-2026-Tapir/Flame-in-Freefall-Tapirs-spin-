"""Offline regression checks for consecutive questions in the Ask form."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from security import MAX_QUESTION_CHARS


class AskPageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        self.retrieve = self.enterContext(patch("rag.retrieve", return_value=[{
            "title": "Offline report", "page": 1, "text": "Offline evidence",
            "pdf_url": "https://example.com/report.pdf",
        }]))
        self.answer = self.enterContext(patch(
            "rag.answer", side_effect=lambda question, *_: f"Answer for: {question} [S1]"))
        self.enterContext(patch.dict(os.environ, {
            "GROQ_API_KEY": "offline-test-key", "GROQ_MODEL": "offline-test-model",
        }))
        self.app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "streamlit_template.py"), default_timeout=15)
        self.app.query_params["page"] = "ask"

    def submit(self, question):
        self.app.text_input[0].input(question)
        self.app.button[0].click().run()
        self.assertEqual(list(self.app.exception), [])

    def assert_answer(self, question):
        self.assertEqual(self.app.text_input[0].value, question)
        self.assertEqual(self.app.query_params["q"], question)
        self.assertEqual(self.app.session_state["ask_result"]["question"], question)
        self.assertIn(f"Answer for: {question} [S1]", [m.value for m in self.app.markdown])

    def test_consecutive_submissions_use_the_latest_question(self):
        self.app.run()
        questions = ["First question about FLEX", "Second question about BASS", "Third question about SAME"]
        for question in questions:
            with self.subTest(question=question):
                self.submit(question)
                self.assert_answer(question)
        self.app.run()
        self.assert_answer(questions[-1])
        self.assertEqual([call.args[0] for call in self.retrieve.call_args_list], questions)
        self.assertEqual([call.args[0] for call in self.answer.call_args_list], questions)

    def test_shared_question_can_be_replaced_repeatedly(self):
        self.app.query_params["q"] = "Shared question about FLEX"
        self.app.run()
        self.assert_answer("Shared question about FLEX")
        for question in ("Replacement question about BASS", "Another question about SAME"):
            self.submit(question)
            self.assert_answer(question)

    def test_example_after_clearing_can_be_replaced(self):
        self.app.query_params["q"] = "Question to clear"
        self.app.run()
        self.submit("")
        self.assertEqual(self.app.text_input[0].value, "")
        self.assertNotIn("q", self.app.query_params)
        example = self.app.button[1].label
        self.app.button[1].click().run()
        self.assertEqual(list(self.app.exception), [])
        self.assert_answer(example)
        self.submit("New question after the example")
        self.assert_answer("New question after the example")

    def test_overlong_question_is_rejected_before_retrieval(self):
        self.app.run()
        self.submit("x" * (MAX_QUESTION_CHARS + 1))
        rendered = " ".join(markdown.value for markdown in self.app.markdown)
        self.assertIn("Question is too long", rendered)
        self.retrieve.assert_not_called()
        self.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
