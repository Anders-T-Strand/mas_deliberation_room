from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MasDeliberationRoom():
    """MasDeliberationRoom crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def marketing_openai_analyst(self):
        return Agent(config=self.agents_config['marketing_openai_analyst'])

    # @agent
    # def marketing_claude_analyst(self):
    #     return Agent(config=self.agents_config['marketing_claude_analyst'])

    @agent
    def marketing_gemini_analyst(self):
        return Agent(config=self.agents_config['marketing_gemini_analyst'])

  
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def marketing_openai_task(self) -> Task:
        return Task(config=self.tasks_config['marketing_openai_task'])

    # @task
    # def marketing_claude_task(self) -> Task:
    #     return Task(config=self.tasks_config['marketing_claude_task'])

    @task
    def marketing_gemini_task(self) -> Task:
        return Task(config=self.tasks_config['marketing_gemini_task'])

    @crew
    def crew(self, agent_mode: str = "multi") -> Crew:
        """Creates the MasDeliberationRoom crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        selected_agents = self.agents
        selected_tasks = self.tasks

        # Allow running with only the initial OpenAI analyst for quicker single-agent runs.
        if agent_mode and agent_mode.lower() == "single":
            selected_agents = [self.marketing_openai_analyst()]
            selected_tasks = [self.marketing_openai_task()]

        return Crew(
            agents=selected_agents, # Automatically created by the @agent decorator
            tasks=selected_tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
