import json
import os
import sys
import socket
import unittest
import copy
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any
from unittest.mock import patch


class StockPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contexts = ExitStack()
        cls.addClassCleanup(contexts.close)
        storage = contexts.enter_context(TemporaryDirectory())
        contexts.enter_context(patch.dict(os.environ, {
            "CREWAI_TRACING_ENABLED": "false",
            "CREWAI_DISABLE_TELEMETRY": "true",
            "OTEL_SDK_DISABLED": "true",
            "OLLAMA_API_KEY": "offline-test-key",
            "CREWAI_STORAGE_DIR": storage,
        }))
        contexts.enter_context(patch("dotenv.load_dotenv", return_value=False))
        original_connect = socket.socket.connect

        def local_ipc_only(sock, address):
            if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
                raise AssertionError("External network prohibited")
            return original_connect(sock, address)

        # Windows asyncio creates loopback socket pairs for internal communication.
        contexts.enter_context(patch("socket.socket.connect", new=local_ipc_only))
        from crewai.llms.providers.openai import completion
        from crewai.tools import tool
        from crewai.events import crewai_event_bus

        contexts.enter_context(patch.object(crewai_event_bus, "emit"))
        contexts.enter_context(patch(
            "crewai.utilities.task_output_storage_handler.KickoffTaskOutputsSQLiteStorage"
        ))

        @tool("Offline search")
        def search(query: str) -> str:
            """Search stand-in for construction-only tests."""
            raise AssertionError("Search must not execute during these tests")

        tools = ModuleType("crewai_tools")
        tools.SerperDevTool = lambda: search
        contexts.enter_context(patch.dict(sys.modules, {"crewai_tools": tools}))
        contexts.enter_context(patch.object(completion, "OpenAI"))
        contexts.enter_context(patch.object(completion, "AsyncOpenAI"))

        from stock_picker import main
        from stock_picker.crew import StockPicker, TrendingCompanyList, TrendingCompanyResearchList
        from crewai import BaseLLM, Crew
        from pydantic import Field

        def offline_crew(**kwargs):
            kwargs["tracing"] = False
            return Crew(**kwargs)

        contexts.enter_context(patch("stock_picker.crew.Crew", side_effect=offline_crew))

        class TextOnlyLLM(BaseLLM):
            responses: list[str]
            calls: list[Any] = Field(default_factory=list)

            def supports_function_calling(self) -> bool:
                return False

            def call(self, messages, tools=None, callbacks=None, available_functions=None,
                     from_task=None, from_agent=None, response_model=None, **kwargs):
                if response_model is not None:
                    raise AssertionError("Provider-enforced structured output must not be used")
                if not self.responses:
                    raise AssertionError("Unexpected extra LLM call")
                self.calls.append(copy.deepcopy(messages))
                return "Final Answer: " + self.responses.pop(0)

        cls.main = main
        cls.crew_class = StockPicker
        cls.llm_class = TextOnlyLLM
        cls.schemas = (TrendingCompanyList, TrendingCompanyResearchList)

    def setUp(self):
        self.inputs = {"sector": "Technology", "current_date": "2026-11-01"}
        clock = patch.object(self.main, "datetime")
        self.addCleanup(clock.stop)
        clock.start().now.return_value = datetime(
            2026, 11, 1, 21, 0
        )

    def test_sequential_crew_has_no_manager(self):
        from crewai import Process

        crew = self.crew_class().crew()
        self.assertEqual(crew.process, Process.sequential)
        self.assertIsNone(crew.manager_agent)
        self.assertIsNone(crew.manager_llm)
        self.assertEqual(len(crew.agents), 3)
        self.assertEqual(len(crew.tasks), 3)
        self.assertEqual(
            [task.name for task in crew.tasks],
            ["find_trending_companies", "research_trending_companies", "pick_best_company"],
        )
        for task, agent in zip(crew.tasks, crew.agents):
            self.assertIs(task.agent, agent)
            self.assertEqual(agent.llm.base_url, "https://ollama.com/v1")
        self.assertEqual(crew.tasks[1].context, [crew.tasks[0]])
        self.assertEqual(crew.tasks[2].context, [crew.tasks[1]])
        self.assertEqual(
            [task.output_file for task in crew.tasks],
            ["output/trending_companies.json", "output/research_report.json", "output/decision.md"],
        )

    def test_inputs_interpolate_all_agents_and_tasks(self):
        crew = self.crew_class().crew()
        inputs = self.inputs
        self.assertEqual(inputs, {"sector": "Technology", "current_date": "2026-11-01"})
        crew._interpolate_inputs(inputs)
        self.assertIn("Technology", crew.agents[0].role)
        for agent in crew.agents:
            self.assertIn("2026-11-01", agent.goal)
        for task in crew.tasks:
            self.assertIn("2026-11-01", task.description)
            self.assertNotIn("{sector}", task.description)
            self.assertNotIn("{current_date}", task.description)

    @unittest.skip("Entry-point input fix is absent; user deferred restoring main.py.")
    def test_run_train_and_test_forward_required_inputs(self):
        for entry_point, method, argv in (
            ("run", "kickoff", ["run"]),
            ("train", "train", ["train", "2", "training.pkl"]),
            ("test", "test", ["test", "2", "ollama/nemotron-3-ultra"]),
        ):
            with self.subTest(entry_point=entry_point), patch.object(
                self.main, "StockPicker"
            ) as factory, patch.object(sys, "argv", argv):
                getattr(self.main, entry_point)()
                kwargs = getattr(factory.return_value.crew.return_value, method).call_args.kwargs
                self.assertEqual(kwargs["inputs"], self.main._stock_inputs())
                if method in ("train", "test"):
                    self.assertEqual(kwargs["n_iterations"], 2)

    @unittest.skip("Trigger input fix is absent; user deferred restoring main.py.")
    def test_trigger_preserves_payload_and_uses_sector(self):
        payload = {"sector": "Healthcare", "event": "research"}
        with patch.object(self.main, "StockPicker") as factory, patch.object(
            sys, "argv", ["trigger", json.dumps(payload)]
        ):
            self.main.run_with_trigger()
        inputs = factory.return_value.crew.return_value.kickoff.call_args.kwargs["inputs"]
        self.assertEqual(inputs["sector"], "Healthcare")
        self.assertEqual(inputs["current_date"], "2026-11-01")
        self.assertEqual(inputs["crewai_trigger_payload"], payload)

    @unittest.skip("Trigger validation is absent; user deferred restoring main.py.")
    def test_invalid_trigger_sector_fails_before_construction(self):
        for payload in (None, [], {}, {"sector": 1}, {"sector": " "}):
            with self.subTest(payload=payload), patch.object(
                self.main, "StockPicker"
            ) as factory, patch.object(sys, "argv", ["trigger", json.dumps(payload)]):
                with self.assertRaisesRegex(ValueError, "sector"):
                    self.main.run_with_trigger()
                factory.assert_not_called()

    def test_json_tasks_use_local_validation_only(self):
        crew = self.crew_class().crew()
        for task, schema in zip(crew.tasks[:2], self.schemas):
            self.assertIsNone(task.output_pydantic)
            self.assertIsNone(task.output_json)
            self.assertIsNone(task.response_model)
            self.assertEqual(task.guardrail_max_retries, 2)
            self.assertIn(json.dumps(schema.model_json_schema()), task.expected_output)
        self.assertIsNone(crew.tasks[2].guardrail)

    def test_guardrails_reject_invalid_outputs(self):
        from crewai import TaskOutput

        crew = self.crew_class().crew()
        for task in crew.tasks[:2]:
            invalid_outputs = [
                "Not JSON", "```json\n{}\n```", "{}", "[]",
                '{"companies": [{"name": "Acme", "reason": "News"}]}',
                '{"companies": [{"name": "Acme", "ticker": 123, "reason": "News"}]}',
                '{"research_list": [{"name": "Acme", "market_position": null}]}',
            ]
            for raw in invalid_outputs:
                with self.subTest(task=task.name, raw=raw):
                    success, feedback = task.guardrail(TaskOutput(
                        description=task.description, raw=raw, agent=task.agent.role
                    ))
                    self.assertFalse(success)
                    self.assertIn("Fix these errors", feedback)

    def prepare_offline_run(self, responses):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(os.chdir, previous)
        crew = self.crew_class().crew()
        crew.tracing = False
        crew.verbose = False
        llm = self.llm_class(model="offline/text-only", responses=responses)
        for agent in crew.agents:
            agent.llm = llm
            agent.verbose = False
        return crew, llm

    def test_retry_normalizes_files_and_passes_valid_json_downstream(self):
        companies = {"companies": [
            {"name": "Acme", "ticker": "ACME", "reason": "New product"},
            {"name": "Example", "ticker": "EX", "reason": "Earnings"},
        ]}
        research = {"research_list": [
            {"name": "Acme", "market_position": "Established", "future_outlook": "Uncertain",
             "investment_potential": "Requires further review"},
            {"name": "Example", "market_position": "Growing", "future_outlook": "Uncertain",
             "investment_potential": "Requires further review"},
        ]}
        crew, llm = self.prepare_offline_run([
            '{"companies": [{"name": "Acme"}]}',
            json.dumps(companies, indent=2),
            "invalid research JSON",
            json.dumps(research, indent=2),
            "# Decision\nAcme was selected for this synthetic test.",
        ])
        result = crew.kickoff(inputs=self.inputs)
        self.assertEqual(len(llm.calls), 5)
        self.assertIn("ticker", str(llm.calls[1]))
        self.assertIn("Fix these errors", str(llm.calls[3]))
        for index, schema in enumerate(self.schemas):
            output = result.tasks_output[index]
            parsed = schema.model_validate_json(output.raw)
            self.assertIsNone(output.pydantic)
            self.assertEqual(output.raw, parsed.model_dump_json())
            self.assertEqual(Path(crew.tasks[index].output_file).read_text(), output.raw)
        self.assertIn(result.tasks_output[0].raw, str(llm.calls[2]))
        self.assertIn(result.tasks_output[1].raw, str(llm.calls[4]))
        self.assertEqual(Path("output/decision.md").read_text(), result.raw)

    def test_retry_exhaustion_stops_without_writing_or_running_downstream(self):
        crew, llm = self.prepare_offline_run(["{}", "{}", "{}"])
        with self.assertRaisesRegex(Exception, "validation after 2 retries"):
            crew.kickoff(inputs=self.inputs)
        self.assertEqual(len(llm.calls), 3)
        for task in crew.tasks:
            self.assertFalse(Path(task.output_file).exists())
        self.assertIsNone(crew.tasks[1].output)
        self.assertIsNone(crew.tasks[2].output)


if __name__ == "__main__":
    unittest.main()
