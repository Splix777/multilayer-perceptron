from typing import Any, Dict

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
# from src.rag.tools.custom_tool import github_tool, directory_tool, python_docs_tool


@CrewBase
class Rag():
	"""Rag crew"""

	agents_config: Dict[str, Any] | None = 'config/agents.yaml'
	tasks_config: Dict[str, Any] | None = 'config/tasks.yaml'
	llm = LLM(
		model="ollama/llama3:latest",
		base_url="http://ollama:11434"
	)

	@agent
	def researcher(self) -> Agent:
		return Agent(
			llm=self.llm,
			config=self.agents_config['researcher'],
			verbose=True
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			llm=self.llm,
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			# tools=[github_tool]
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical,
			# manager_llm=self.llm,
			# memory=True,
			# manager_agent=None,
			# planning=True
		)