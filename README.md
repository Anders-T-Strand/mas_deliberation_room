# MasDeliberationRoom Crew

Welcome to the MasDeliberationRoom Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

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

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/mas_deliberation_room/config/agents.yaml` to define your agents
- Modify `src/mas_deliberation_room/config/tasks.yaml` to define your tasks
- Modify `src/mas_deliberation_room/crew.py` to add your own logic, tools and specific args
- Modify `src/mas_deliberation_room/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the MAS_Deliberation_Room Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The MAS_Deliberation_Room Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the MasDeliberationRoom Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.


# Notes by Anders Strand

To install and setup CrewAI you need to install uv. All the documentation can be found at this link: https://docs.crewai.com/en/installation
But otherwise here a are the steps (this will all be for windows, go to the prvoide link to see other os installation steps).

1. Install uv by running this in your terminal: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

2. Install CrewAI: uv tool install crewai

To run the agent, go to terminal and type: crewai run
In the output folder you will get the result of what the agent have come up with. 

## Agent

The current agent runs on gemini-2.5-flash.
The reason for the use of gemini right now is because it's free for me right now.
Next steps will be to add another agent until I hit the goal of four agents working together.
Currently the tasks are running in sequential order, this can be changed but requires some adjustments. Read CerwAI documentation for this.

### Current agents (3)
- OpenAI analyst (data-driven diagnostics)
- Claude creative architect (competitor intel + campaign design)
- Gemini operations director (execution, budgets, KPIs)

> Note: The Claude agent defaults to `anthropic/claude-3-haiku-20240307` to avoid model not-found issues. Set `ANTHROPIC_API_KEY` and adjust the model id in `config/agents.yaml` if you have access to Sonnet-tier models.

Run modes:
- Single agent: `MAS_RUN_MODE=single crewai run`
- Multi agent (all three): `MAS_RUN_MODE=multi crewai run` (default)
- Back-to-back comparison: `MAS_RUN_MODE=both crewai run` (writes both runs)

Outputs and visualizations:
- Per-mode metrics are stored in `output/agent_mode_results.json` (rubric %, schema validity, timing, refinement deltas).
- Charts in `output/` compare single vs multi (`quality_comparison.png`, `execution_time_comparison.png`, `diversity_lift.png`).


