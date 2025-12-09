"""
Enhanced CrewAI Crew Definition
Supports: Ablation studies, pipeline ordering experiments, flexible agent configurations
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Optional


# Agent ordering configurations
AGENT_ORDERINGS = {
    "default": ["openai", "claude", "gemini"],
    "reverse": ["gemini", "claude", "openai"],
    "claude_first": ["claude", "openai", "gemini"],
    "claude_last": ["openai", "gemini", "claude"],
    "gemini_first": ["gemini", "openai", "claude"],
    "openai_last": ["claude", "gemini", "openai"],
}

# Ablation configurations
ABLATION_CONFIGS = {
    "full_pipeline": ["openai", "claude", "gemini"],
    "two_agent_no_claude": ["openai", "gemini"],
    "two_agent_no_gemini": ["openai", "claude"],
    "homogeneous_gpt": ["openai", "openai", "openai"],
    "single_agent": ["openai"],
}


@CrewBase
class MasDeliberationRoom():
    """MasDeliberationRoom crew with ablation and ordering support"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # =========================================================================
    # AGENT DEFINITIONS
    # =========================================================================
    
    @agent
    def marketing_openai_analyst(self) -> Agent:
        """Primary data-driven analyst using GPT-4o-mini"""
        return Agent(config=self.agents_config['marketing_openai_analyst'])

    @agent
    def marketing_claude_analyst(self) -> Agent:
        """Creative architect using Claude Haiku"""
        return Agent(config=self.agents_config['marketing_claude_analyst'])

    @agent
    def marketing_gemini_analyst(self) -> Agent:
        """Operations director using Gemini Flash"""
        return Agent(config=self.agents_config['marketing_gemini_analyst'])
    
    # Additional agent variants for ablation studies
    def _create_openai_creative(self) -> Agent:
        """GPT variant for creative role (homogeneous ablation)"""
        config = dict(self.agents_config['marketing_openai_analyst'])
        config['role'] = "Creative Marketing Architect"
        config['goal'] = """Enrich the strategy with creative elements, emotional resonance, 
        and brand voice while preserving structure and compliance."""
        config['backstory'] = """You enhance narrative and emotional appeal while maintaining 
        schema integrity and all required elements."""
        return Agent(config=config)
    
    def _create_openai_operations(self) -> Agent:
        """GPT variant for operations role (homogeneous ablation)"""
        config = dict(self.agents_config['marketing_openai_analyst'])
        config['role'] = "Marketing Operations Director"
        config['goal'] = """Validate, finalize, and add operational details to the strategy. 
        Ensure rubric perfection and JSON integrity."""
        config['backstory'] = """You are the final authority ensuring the strategy is 
        operational, consistent, and maximally aligned with scoring criteria."""
        return Agent(config=config)

    # =========================================================================
    # TASK DEFINITIONS
    # =========================================================================
    
    @task
    def marketing_openai_task(self) -> Task:
        return Task(config=self.tasks_config['marketing_openai_task'])

    @task
    def marketing_claude_task(self) -> Task:
        return Task(config=self.tasks_config['marketing_claude_task'])

    @task
    def marketing_gemini_task(self) -> Task:
        return Task(config=self.tasks_config['marketing_gemini_task'])
    
    # Additional task variants
    def _create_openai_creative_task(self) -> Task:
        """Creative task for GPT (homogeneous ablation)"""
        config = dict(self.tasks_config['marketing_claude_task'])
        config['agent'] = 'marketing_openai_analyst'  # Use OpenAI agent
        return Task(config=config)
    
    def _create_openai_operations_task(self) -> Task:
        """Operations task for GPT (homogeneous ablation)"""
        config = dict(self.tasks_config['marketing_gemini_task'])
        config['agent'] = 'marketing_openai_analyst'  # Use OpenAI agent
        return Task(config=config)

    # =========================================================================
    # AGENT/TASK SELECTION HELPERS
    # =========================================================================
    
    def _get_agent_by_name(self, name: str) -> Agent:
        """Get agent instance by name."""
        agent_map = {
            "openai": self.marketing_openai_analyst,
            "claude": self.marketing_claude_analyst,
            "gemini": self.marketing_gemini_analyst,
        }
        if name not in agent_map:
            raise ValueError(f"Unknown agent: {name}. Valid: {list(agent_map.keys())}")
        return agent_map[name]()
    
    def _get_task_by_name(self, name: str) -> Task:
        """Get task instance by name."""
        task_map = {
            "openai": self.marketing_openai_task,
            "claude": self.marketing_claude_task,
            "gemini": self.marketing_gemini_task,
        }
        if name not in task_map:
            raise ValueError(f"Unknown task: {name}. Valid: {list(task_map.keys())}")
        return task_map[name]()

    # =========================================================================
    # CREW BUILDERS
    # =========================================================================

    @crew
    def crew(self, agent_mode: str = "multi", 
             ablation: str = None, 
             ordering: str = None) -> Crew:
        """
        Creates the MasDeliberationRoom crew with flexible configuration.
        
        Args:
            agent_mode: "single" or "multi"
            ablation: Ablation config name (e.g., "homogeneous_gpt", "two_agent_no_claude")
            ordering: Agent ordering name (e.g., "default", "reverse", "claude_first")
        
        Returns:
            Configured Crew instance
        """
        mode = (agent_mode or "multi").lower()
        
        # Handle single agent mode
        if mode == "single":
            return self.single_agent_crew()
        
        # Handle ablation studies
        if ablation and ablation in ABLATION_CONFIGS:
            return self._build_ablation_crew(ablation)
        
        # Handle ordering experiments
        if ordering and ordering in AGENT_ORDERINGS:
            return self._build_ordered_crew(ordering)
        
        # Default multi-agent configuration
        return self._build_default_multi_crew()
    
    def _build_default_multi_crew(self) -> Crew:
        """Build standard 3-agent crew: OpenAI → Claude → Gemini"""
        return Crew(
            agents=[
                self.marketing_openai_analyst(),
                self.marketing_claude_analyst(),
                self.marketing_gemini_analyst(),
            ],
            tasks=[
                self.marketing_openai_task(),
                self.marketing_claude_task(),
                self.marketing_gemini_task(),
            ],
            process=Process.sequential,
            verbose=False,  # Reduced verbosity for cleaner output
        )
    
    def _build_ordered_crew(self, ordering: str) -> Crew:
        """
        Build crew with specified agent ordering.
        
        This addresses Feedback Item #10: Pipeline ordering effects
        
        Creates tasks dynamically based on POSITION in pipeline, not agent type.
        Each task only depends on the previous task(s), avoiding future dependencies.
        """
        order = AGENT_ORDERINGS.get(ordering, AGENT_ORDERINGS["default"])
        
        agents = [self._get_agent_by_name(name) for name in order]
        
        # Create tasks dynamically - each depends only on previous tasks
        tasks = []
        for i, agent_name in enumerate(order):
            agent = agents[i]
            
            if i == 0:
                # First agent: Create initial strategy
                task = Task(
                    description=self._get_initial_task_desc(),
                    expected_output="Valid JSON marketing strategy",
                    agent=agent,
                    output_file=f"output/{agent_name}_strategy.json"
                )
            elif i == len(order) - 1:
                # Last agent: Finalize strategy
                task = Task(
                    description=self._get_final_task_desc(),
                    expected_output="Final JSON marketing strategy",
                    agent=agent,
                    context=tasks.copy(),  # Copy to avoid mutation issues
                    output_file="output/final_strategy.json"
                )
            else:
                # Middle agent: Enhance strategy
                task = Task(
                    description=self._get_enhance_task_desc(),
                    expected_output="Enhanced JSON marketing strategy",
                    agent=agent,
                    context=tasks.copy(),  # Copy to avoid mutation issues
                    output_file=f"output/{agent_name}_strategy.json"
                )
            tasks.append(task)
        
        print(f"[Crew] Ordering: {ordering} ({' → '.join(order)})")
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
    
    def _get_initial_task_desc(self) -> str:
        """Task description for first agent - create initial strategy."""
        return """Create a marketing strategy for {industry} targeting {target_audience}.
Context: {current_strategy}
Data: {sales_data}

Output JSON with: strategy_text (mention industry/audience), weaknesses (3-5), 
kpis (with name/current/target/timeframe_days), competitors (2+), 
tactics (with roi_rank), budget (total_usd + items array), benchmarks, assumptions.
Output ONLY valid JSON, no markdown fences."""

    def _get_enhance_task_desc(self) -> str:
        """Task description for middle agents - enhance strategy."""
        return """Enhance the previous strategy while preserving all required fields.
Add: creative details, measurement_methods to KPIs, 1 new tactic with budget.
Preserve: industry/audience mentions, all numeric data, budget structure.
Output ONLY valid JSON, no markdown fences."""

    def _get_final_task_desc(self) -> str:
        """Task description for final agent - finalize with operational details."""
        return """Finalize the strategy with operational details.
Add: implementation_phases (2-4), team_requirements, risks with mitigations.
Verify: budget total_usd matches sum of items, all KPIs have measurement_method.
Output ONLY valid JSON, no markdown fences."""
    
    def _build_ablation_crew(self, ablation: str) -> Crew:
        """
        Build crew for ablation study.
        
        This addresses Feedback Item #2: No ablation studies
        
        Uses dynamic task creation to avoid context dependency errors.
        """
        config = ABLATION_CONFIGS.get(ablation)
        if not config:
            raise ValueError(f"Unknown ablation: {ablation}")
        
        print(f"[Crew] Ablation: {ablation} ({config})")
        
        # Special handling for homogeneous configurations
        if ablation == "homogeneous_gpt":
            return self._build_homogeneous_gpt_crew()
        
        # Build crew with dynamic tasks
        agents = [self._get_agent_by_name(name) for name in config]
        
        tasks = []
        for i, agent_name in enumerate(config):
            agent = agents[i]
            
            if i == 0:
                task = Task(
                    description=self._get_initial_task_desc(),
                    expected_output="Valid JSON marketing strategy",
                    agent=agent,
                    output_file=f"output/{agent_name}_strategy.json"
                )
            elif i == len(config) - 1:
                task = Task(
                    description=self._get_final_task_desc(),
                    expected_output="Final JSON marketing strategy",
                    agent=agent,
                    context=tasks.copy(),  # Copy to avoid mutation issues
                    output_file="output/final_strategy.json"
                )
            else:
                task = Task(
                    description=self._get_enhance_task_desc(),
                    expected_output="Enhanced JSON marketing strategy",
                    agent=agent,
                    context=tasks.copy(),  # Copy to avoid mutation issues
                    output_file=f"output/{agent_name}_strategy.json"
                )
            tasks.append(task)
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
    
    def _build_homogeneous_gpt_crew(self) -> Crew:
        """
        Build homogeneous GPT crew: GPT → GPT → GPT
        Uses same model but different role prompts.
        """
        # Create three GPT agents with different roles
        analyst = self.marketing_openai_analyst()
        creative = self._create_openai_creative()
        operations = self._create_openai_operations()
        
        # Create tasks dynamically (not from tasks.yaml)
        task1 = Task(
            description=self._get_initial_task_desc(),
            expected_output="Valid JSON marketing strategy",
            agent=analyst,
            output_file="output/openai_strategy.json"
        )
        
        task2 = Task(
            description=self._get_enhance_task_desc(),
            expected_output="Enhanced JSON marketing strategy",
            agent=creative,
            context=[task1],
            output_file="output/openai_creative_strategy.json"
        )
        
        task3 = Task(
            description=self._get_final_task_desc(),
            expected_output="Final JSON marketing strategy",
            agent=operations,
            context=[task1, task2],
            output_file="output/final_strategy.json"
        )
        
        print("[Crew] Homogeneous GPT: GPT-Analyst → GPT-Creative → GPT-Operations")
        
        return Crew(
            agents=[analyst, creative, operations],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=False,
        )

    def single_agent_crew(self) -> Crew:
        """Creates a single-agent crew (OpenAI analyst only)"""
        return Crew(
            agents=[self.marketing_openai_analyst()],
            tasks=[self.marketing_openai_task()],
            process=Process.sequential,
            verbose=False,  # Reduced verbosity
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @classmethod
    def list_configurations(cls) -> dict:
        """List all available configurations."""
        return {
            "orderings": list(AGENT_ORDERINGS.keys()),
            "ablations": list(ABLATION_CONFIGS.keys()),
            "ordering_details": AGENT_ORDERINGS,
            "ablation_details": {k: {"agents": v} for k, v in ABLATION_CONFIGS.items()},
        }
    
    @classmethod
    def print_configurations(cls):
        """Print available configurations."""
        print("\n" + "="*60)
        print("AVAILABLE CREW CONFIGURATIONS")
        print("="*60)
        
        print("\nAgent Orderings:")
        for name, order in AGENT_ORDERINGS.items():
            print(f"  {name:<20} {' → '.join(order)}")
        
        print("\nAblation Configurations:")
        for name, agents in ABLATION_CONFIGS.items():
            print(f"  {name:<25} {agents}")
        
        print("="*60 + "\n")