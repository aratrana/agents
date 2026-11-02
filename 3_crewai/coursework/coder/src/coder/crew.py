from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from .tools.sandbox_tools import sandbox_tools
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Coder():
    """Coder crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'], # type: ignore[index],
             llm=LLM(
                model=self.agents_config['coder']['llm'], # type: ignore[index]
                base_url='https://ollama.com/v1',
            ),
            verbose=True,
            tools=sandbox_tools
        )
        
    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Coder crew"""
        
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )
