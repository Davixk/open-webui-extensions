"""
title: nokodo ai
authors: @nokodo
author_url: https://nokodo.net
repository_url: https://nokodo.net/github/open-webui-extensions
description: proxy pipe for nokodo ai models with data injection.
version: 0.1.0
required_open_webui_version: >= 0.5.0
funding_url: https://ko-fi.com/nokodo
"""

from typing import Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message, add_or_update_system_message

# TODO: do not append system message if one already exists in the body, only append context info
# TODO: actually hook it up to underlying model. as it is now, it doesnt work. find out how to do this.


class Pipe:
    class Valves(BaseModel):
        nokodo_ai_model: str = Field(
            default="openai/gpt-4o",
            description="nokodo ai model to use. this is the standard model for nokodo ai.",
        )
        nokodo_ai_reasoner_model: str = Field(
            default="openai/gpt-4o",
            description="nokodo ai reasoner model to use. this is the reasoner model for nokodo ai.",
        )
        nokodo_ai_mini_model: str = Field(
            default="openai/gpt-4o",
            description="nokodo ai mini model to use. this is the mini model for nokodo ai.",
        )
        nokodo_ai_system_prompt: str = Field(
            default="You are Nokodo AI, a helpful assistant created to provide accurate and thoughtful responses.",
            description="system prompt for nokodo ai model.",
        )
        nokodo_ai_reasoner_system_prompt: str = Field(
            default="You are Nokodo AI Reasoner, an assistant that carefully reasons through problems step by step, showing your work and explaining your thought process thoroughly.",
            description="system prompt for nokodo ai reasoner model.",
        )
        nokodo_ai_mini_system_prompt: str = Field(
            default="You are Nokodo AI Mini, a concise and efficient assistant focused on providing direct answers.",
            description="system prompt for nokodo ai mini model.",
        )
        global_context: str = Field(
            default="",
            description="global context to add to all system prompts. this is a suffix to the system prompt, to inject context info into the model.",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "nokodo"
        self.name = "nokodo/"
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        return [
            {"id": "nokodo-ai", "name": "nokodo ai"},
            {"id": "nokodo-ai-reasoner", "name": "nokodo ai reasoner"},
            {"id": "nokodo-ai-mini", "name": "nokodo ai mini"},
        ]

    def pipe(self, body: dict) -> str | Generator | Iterator:
        # Extract model ID
        full_model_name: str = body[
            "model"
        ]  # will be "nokodo.nokodo-ai" or "nokodo.nokodo-ai-reasoner" or "nokodo.nokodo-ai-mini"
        if "." in full_model_name:
            model_parts = full_model_name.split(".")
            if "nokodo" in model_parts[0]:
                model_id = model_parts[1]
            else:
                raise ValueError(
                    f"Unknown model prefix: {model_parts[0]}. Expected 'nokodo'."
                )
        else:
            model_id = full_model_name
        # Debugging output
        print(f"nokodo_ai:DEBUG: full model name: {full_model_name}")
        print(f"nokodo_ai:DEBUG: Model ID: {model_id}")

        # Determine which real model to use based on Nokodo model name
        target_model = None
        custom_system_prompt = None

        if model_id == "nokodo-ai":
            target_model = self.valves.nokodo_ai_model
            custom_system_prompt = self.valves.nokodo_ai_system_prompt
        elif model_id == "nokodo-ai-reasoner":
            target_model = self.valves.nokodo_ai_reasoner_model
            custom_system_prompt = self.valves.nokodo_ai_reasoner_system_prompt
        elif model_id == "nokodo-ai-mini":
            target_model = self.valves.nokodo_ai_mini_model
            custom_system_prompt = self.valves.nokodo_ai_mini_system_prompt
        else:
            raise ValueError(f"Unknown model: {model_id}")

        print(f"nokodo_ai:DEBUG: Target Model: {target_model}")
        print(f"nokodo_ai:DEBUG: Custom System Prompt: {custom_system_prompt}")

        # Get system message and regular messages
        system_message, messages = pop_system_message(body["messages"])

        if system_message:
            print(f"nokodo_ai:DEBUG: Original System Message: {system_message}")
            # add context info to system message
            if self.valves.global_context:
                system_message = f"{system_message}\n\n{self.valves.global_context}"
        else:
            print("nokodo_ai:DEBUG: No system message found in the body.")
            # create a new system message with context info
            system_message = custom_system_prompt
            if self.valves.global_context:
                system_message = f"{system_message}\n\n{self.valves.global_context}"

        # Create a new message list with the system message added back
        messages = add_or_update_system_message(
            content=system_message, messages=messages
        )

        print(f"nokodo_ai:DEBUG: system_message: {system_message}")

        # Create a new request body with the target model and updated messages
        new_body = body.copy()
        new_body["model"] = target_model
        new_body["messages"] = messages

        raise NotImplementedError(
            "This is a stub. The actual model call should be implemented here."
        )
