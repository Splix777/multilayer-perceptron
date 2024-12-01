from typing import Any, Dict
import yaml

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.rag.tools.custom_tool import directory_read_tool, file_read_tool, website_search_tool




@CrewBase
class Rag:
    """Rag crew"""

    agents_config = yaml.safe_load("config/agents.yaml")
    tasks_config = yaml.safe_load("config/tasks.yaml")
    llm = LLM(model="ollama/llama3:latest", base_url="http://ollama:11434")

    @agent
    def researcher(self) -> Agent:
        return Agent(
            llm=self.llm,
            config=self.agents_config["researcher"],
            role=self.agents_config["researcher"]["role"],
            goal=self.agents_config["researcher"]["goal"],
            backstory=self.agents_config["researcher"]["backstory"],
            verbose=True,
        )
    
    @agent
    def engineer(self) -> Agent:
        return Agent(
            llm=self.llm,
            config=self.agents_config["engineer"],
            role=self.agents_config["engineer"]["role"],
            goal=self.agents_config["engineer"]["goal"],
            backstory=self.agents_config["engineer"]["backstory"],
            verbose=True,
        )

    @task
    def researcher_task(self) -> Task:
        return Task(
            name="Find a file",
            config=self.tasks_config["research_task"],
            description=self.tasks_config["research_task"]["description"],
            expected_output=self.tasks_config["research_task"]["expected_output"],
            tools=[website_search_tool],
        )
    
    @task
    def engineer_task(self) -> Task:
        return Task(
            name="Analyze code",
            config=self.tasks_config["engineer_task"],
            description=self.tasks_config["engineer_task"]["description"],
            expected_output=self.tasks_config["engineer_task"]["expected_output"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
            agents=[self.researcher(), self.engineer()],
            tasks=[self.researcher_task(), self.engineer_task()],
            process=Process.sequential,
            verbose=True
        )
