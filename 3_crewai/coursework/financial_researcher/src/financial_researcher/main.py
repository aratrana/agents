#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from financial_researcher.crew import FinancialResearcher

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def _research_inputs(company: str) -> dict[str, str]:
    company = company.strip()
    if not company:
        raise ValueError("Please provide a non-empty company name.")
    now = datetime.now()
    return {
        'company': company,
        'current_year': str(now.year),
        'current_date': now.strftime('%Y-%m-%d'),
    }


def run():
    """
    Run the crew.
    """
    company: str = input("Enter the company name for research: ")
    inputs = _research_inputs(company)

    try:
        FinancialResearcher().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = _research_inputs(input("Enter the company name for research: "))
    try:
        FinancialResearcher().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        FinancialResearcher().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = _research_inputs(input("Enter the company name for research: "))

    try:
        FinancialResearcher().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    if not isinstance(trigger_payload, dict) or not isinstance(trigger_payload.get("company"), str):
        raise ValueError("Trigger payload must be a JSON object with a string 'company' field.")

    inputs = {
        **_research_inputs(trigger_payload["company"]),
        "crewai_trigger_payload": trigger_payload,
    }

    try:
        result = FinancialResearcher().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
