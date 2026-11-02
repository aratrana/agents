# StockPicker Crew

Welcome to the StockPicker Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Set `OLLAMA_API_KEY`, `SERPER_API_KEY`, `PUSHOVER_USER`, and `PUSHOVER_TOKEN`
in your project's `.env` file. Keep credentials out of source control.**

- Modify `src/stock_picker/config/agents.yaml` to define your agents
- Modify `src/stock_picker/config/tasks.yaml` to define your tasks
- Modify `src/stock_picker/crew.py` to add your own logic, tools and specific args
- Modify `src/stock_picker/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the stock_picker Crew, assembling the agents and assigning them tasks as defined in your configuration.

The crew runs sequentially without a manager agent:

1. Find trending companies.
2. Research those companies using the first task's output.
3. Pick a company using the research, send a push notification, and write the decision.

Outputs are written to `output/trending_companies.json`,
`output/research_report.json`, and `output/decision.md`.

Templates require `sector` and `current_date`. The current `run()` uses
`Technology` and supplies the year as `current_date`. Training, evaluation, and
trigger entry points still use scaffold inputs (`topic`/`current_year`) and need
updating before use. Those entry-point changes are intentionally deferred.

### Offline regression tests

Run `uv run python -m unittest discover -s tests -v`. The tests verify manager-free
construction, task order, context dependencies, and template inputs. Provider clients
and search are mocked; no API keys, searches, or push notifications are used.
The three regression tests for the deferred entry-point fixes are explicitly skipped.

### Structured JSON with Ollama Cloud

The two research tasks keep using `ollama/nemotron-3-ultra` at
`https://ollama.com/v1`. Their JSON schemas are included in the prompts, not sent
as provider-enforced response formats. Function guardrails validate responses
locally with Pydantic and return normalized JSON.

Invalid JSON, missing fields, or incorrect field types produce validation
feedback and up to two correction attempts. If validation still fails, the task
raises an error before writing its output or running downstream tasks. Retries
consume additional model requests and remain subject to account usage limits.

Validated JSON is available in each research task's `TaskOutput.raw`, not
`TaskOutput.pydantic` or `TaskOutput.json_dict`. Parse it with
`TrendingCompanyList.model_validate_json(output.raw)` or
`TrendingCompanyResearchList.model_validate_json(output.raw)` when a typed object
is needed. The final decision remains a Markdown report.

`ValidatedJSONTask` ensures files contain the accepted, normalized result;
CrewAI 1.14.4 otherwise writes the original response after a guardrail correction.
Failed runs do not overwrite existing output files, so files from an earlier run
may still be present.

## Understanding Your Crew

The stock_picker Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the StockPicker Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
