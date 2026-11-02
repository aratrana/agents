import json
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field, ValidationError
from crewai_tools import SerperDevTool
from .tools.push_tool import send_push_notification

# Pydantic models(class) for trending companies and their research
class TrendingCompany(BaseModel):
    """ A company that is in the news and attracting attention """
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")

class TrendingCompanyList(BaseModel):
    """ List of multiple trending companies that are in the news """
    companies: list[TrendingCompany] = Field(description="List of companies trending in the news")

class TrendingCompanyResearch(BaseModel):
    """ Detailed research on a company """
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """ A list of detailed research on all the companies """
    research_list: list[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")


class ValidatedJSONTask(Task):
    def _save_file(self, result: Any) -> None:
        # CrewAI 1.14.4 otherwise saves the original response, even after a retry.
        if self.output is None:
            raise RuntimeError("Cannot save JSON before task validation completes.")
        super()._save_file(self.output.raw)


def validated_json_task(config: dict[str, Any], schema: type[BaseModel]) -> Task:
    def validate_json(output: TaskOutput) -> tuple[bool, Any]:
        try:
            parsed = schema.model_validate_json(output.raw, strict=True)
        except ValidationError as error:
            return False, (
                "Return only a JSON object matching the required schema. "
                "Do not include Markdown fences or commentary. Fix these errors: "
                + error.json(include_input=False, include_url=False)
            )
        return True, parsed.model_dump_json()

    task_config = dict(config)
    task_config["expected_output"] += (
        "\nReturn only a JSON object matching this JSON Schema, without Markdown "
        "fences or commentary:\n" + json.dumps(schema.model_json_schema())
    )
    return ValidatedJSONTask(
        config=task_config,
        guardrail=validate_json,
        guardrail_max_retries=2,
    )


@CrewBase
class StockPicker():
    """StockPicker crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
                config=self.agents_config['trending_company_finder'],
                llm=LLM(
                        model=self.agents_config['trending_company_finder']['llm'], # type: ignore[index]
                        base_url='https://ollama.com/v1',
                ),
                tools=[SerperDevTool()]
            )
    
    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
                config=self.agents_config['financial_researcher'],
                llm=LLM(
                        model=self.agents_config['financial_researcher']['llm'], # type: ignore[index]
                        base_url='https://ollama.com/v1',
                ),
                tools=[SerperDevTool()]
            )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
                config=self.agents_config['stock_picker'],
                llm=LLM(
                        model=self.agents_config['stock_picker']['llm'], # type: ignore[index]
                        base_url='https://ollama.com/v1',
                ),
                tools=[send_push_notification]
            )
    
    @task
    def find_trending_companies(self) -> Task:
        return validated_json_task(
            self.tasks_config['find_trending_companies'],
            TrendingCompanyList,
        )

    @task
    def research_trending_companies(self) -> Task:
        return validated_json_task(
            self.tasks_config['research_trending_companies'],
            TrendingCompanyResearchList,
        )

    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            tracing=True
        )