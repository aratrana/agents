import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch


MAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "financial_researcher" / "main.py"


class MainInputTests(unittest.TestCase):
    def setUp(self):
        self.factory = MagicMock()
        crew_module = ModuleType("financial_researcher.crew")
        crew_module.FinancialResearcher = self.factory
        spec = importlib.util.spec_from_file_location("financial_main_under_test", MAIN_PATH)
        self.main = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"financial_researcher.crew": crew_module}):
            spec.loader.exec_module(self.main)
        self.clock = patch.object(self.main, "datetime")
        self.clock.start().now.return_value = datetime(2026, 11, 1, 12, 0)
        self.addCleanup(self.clock.stop)
        self.crew = self.factory.return_value.crew.return_value

    def assert_research_inputs(self, inputs):
        self.assertEqual(inputs["company"], "Example Company")
        self.assertEqual(inputs["current_date"], "2026-11-01")
        self.assertEqual(inputs["current_year"], "2026")
        self.assertNotIn("topic", inputs)

    def test_run_inputs(self):
        with patch("builtins.input", return_value="  Example Company  "):
            self.main.run()
        self.assert_research_inputs(self.crew.kickoff.call_args.kwargs["inputs"])

    def test_train_inputs_and_arguments(self):
        with patch("builtins.input", return_value="Example Company"), patch.object(
            self.main.sys, "argv", ["train", "2", "training.pkl"]
        ):
            self.main.train()
        kwargs = self.crew.train.call_args.kwargs
        self.assert_research_inputs(kwargs["inputs"])
        self.assertEqual(kwargs["n_iterations"], 2)
        self.assertEqual(kwargs["filename"], "training.pkl")

    def test_evaluation_inputs_and_arguments(self):
        with patch("builtins.input", return_value="Example Company"), patch.object(
            self.main.sys, "argv", ["test", "2", "ollama/nemotron-3-ultra"]
        ):
            self.main.test()
        kwargs = self.crew.test.call_args.kwargs
        self.assert_research_inputs(kwargs["inputs"])
        self.assertEqual(kwargs["n_iterations"], 2)
        self.assertEqual(kwargs["eval_llm"], "ollama/nemotron-3-ultra")

    def test_trigger_inputs_preserve_payload(self):
        payload = {"company": "Example Company", "event": "research"}
        with patch.object(self.main.sys, "argv", ["trigger", json.dumps(payload)]):
            self.main.run_with_trigger()
        inputs = self.crew.kickoff.call_args.kwargs["inputs"]
        self.assert_research_inputs(inputs)
        self.assertEqual(inputs["crewai_trigger_payload"], payload)

    def test_invalid_trigger_company_fails_before_kickoff(self):
        for payload in ([], None, {}, {"company": 123}, {"company": " "}):
            with self.subTest(payload=payload), patch.object(
                self.main.sys, "argv", ["trigger", json.dumps(payload)]
            ):
                with self.assertRaisesRegex(ValueError, "company"):
                    self.main.run_with_trigger()
        self.factory.assert_not_called()

    def test_blank_interactive_company_fails_before_kickoff(self):
        for entry_point in (self.main.run, self.main.train, self.main.test):
            with self.subTest(entry_point=entry_point.__name__), patch(
                "builtins.input", return_value=" "
            ):
                with self.assertRaisesRegex(ValueError, "company"):
                    entry_point()
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
