import json
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class EventGrade(BaseModel):
    grade: int


class EpisodicMemory(Memory):
    def __init__(
        self,
        agent: "LLMAgent",
        api_key: str | None = None,
        llm_model: str | None = None,
        display: bool = True,
        max_memory: int = 10,
    ):
        """
        Initialize the EpisodicMemory
        """
        if not api_key or not llm_model:
            raise ValueError(
                "Both api_key and llm_model must be provided for the usage of episodic memory"
            )

        super().__init__(agent, api_key=api_key, llm_model=llm_model, display=display)

        self.max_memory = max_memory
        self.memory = deque(maxlen=self.max_memory)

        self.system_prompt = """
            You are an assistant that evaluates memory entries on a scale from 1 to 5, based on their importance to a specific problem or task. Your goal is to assign a score that reflects how much each entry contributes to understanding, solving, or advancing the task. Use the following grading scale:

            5 - Critical: Introduces essential, novel information that significantly impacts problem-solving or decision-making.

            4 - High: Provides important context or clarification that meaningfully improves understanding or direction.

            3 - Moderate: Adds somewhat useful information that may assist but is not essential.

            2 - Low: Offers minimal relevance or slight redundancy; impact is marginal.

            1 - Irrelevant: Contains no useful or applicable information for the current problem.

            Only assess based on the entry's content and its value to the task at hand. Ignore style, grammar, or tone.
            """

    def grade_event_importance(self, type: str, content: dict) -> float:
        """
        Grade this event based on the content respect to the previous memory entries
        """
        if len(self.memory) in range(5):
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in self.memory]
            )
        elif len(self.memory) > 5:
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in self.memory[-5:]]
            )
        else:
            previous_entries = "No previous memory entries"

        prompt = f"""
            grade the importance of the following event on a scale from 1 to 5:
            {type}: {content}
            ------------------------------
            {previous_entries}
            """

        self.llm.system_prompt = self.system_prompt

        rsp = self.agent.llm.generate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    def retrieve_top_k_entries(self, k: int) -> list[MemoryEntry]:
        """
        Retrieve the top k entries based on the importance and recency
        """
        return []

    def add_to_memory(self, type: str, content: dict):
        """
        Add a new memory entry to the memory
        """
        self.grade_event_importance(type, content)
        content["importance"] = self.grade_event_importance(type, content)

        super().add_to_memory(type, content)
