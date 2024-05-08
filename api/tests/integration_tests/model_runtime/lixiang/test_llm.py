import os
from collections.abc import Generator

import pytest

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.lixiang.llm.llm import LixiangLargeLanguageModel

"""
Using Together.ai's OpenAI-compatible API as testing endpoint
"""


def test_validate_credentials():
    model = LixiangLargeLanguageModel()

    with pytest.raises(CredentialsValidateFailedError):
        model.validate_credentials(
            model='gpt4-turbo',
            credentials={
                'api_base_url': 'hahahaha',
                'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
            }
        )

    model.validate_credentials(
        model='gpt4-turbo',
        credentials={
            'api_base_url': 'http://api-hub.inner.chj.cloud/bcs-apihub-ai-proxy-service/apihub/openai/chat/completions',
            'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
        }
    )


def test_invoke_model():
    model = LixiangLargeLanguageModel()

    response = model.invoke(
        model='gpt4-turbo',
        credentials={
            'api_base_url': 'http://api-hub.inner.chj.cloud/bcs-apihub-ai-proxy-service/apihub/openai/chat/completions',
            'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
        },
        prompt_messages=[
            SystemPromptMessage(
                content='You are a helpful AI assistant.',
            ),
            UserPromptMessage(
                content='Who are you?'
            )
        ],
        model_parameters={
            'temperature': 1.0,
            'top_k': 2,
            'top_p': 0.5,
        },
        stop=['How'],
        stream=False,
        user="abc-123"
    )

    assert isinstance(response, LLMResult)
    assert len(response.message.content) > 0


def test_invoke_stream_model():
    model = LixiangLargeLanguageModel()

    response = model.invoke(
        model='gpt4-turbo',
        credentials={
            'api_base_url': 'http://api-hub.inner.chj.cloud/bcs-apihub-ai-proxy-service/apihub/openai/chat/completions',
            'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
        },
        prompt_messages=[
            SystemPromptMessage(
                content='You are a helpful AI assistant.',
            ),
            UserPromptMessage(
                content='Who are you?'
            )
        ],
        model_parameters={
            'temperature': 1.0,
            'top_k': 2,
            'top_p': 0.5,
        },
        stop=['How'],
        stream=True,
        user="abc-123"
    )

    assert isinstance(response, Generator)

    for chunk in response:
        assert isinstance(chunk, LLMResultChunk)
        assert isinstance(chunk.delta, LLMResultChunkDelta)
        assert isinstance(chunk.delta.message, AssistantPromptMessage)


# using OpenAI's ChatGPT-3.5 as testing endpoint
def test_invoke_chat_model_with_tools():
    model = LixiangLargeLanguageModel()

    result = model.invoke(
        model='gpt4-turbo',
        credentials={
            'api_base_url': 'http://api-hub.inner.chj.cloud/bcs-apihub-ai-proxy-service/apihub/openai/chat/completions',
            'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
            'function_calling_type': 'function_call',
        },
        prompt_messages=[
            SystemPromptMessage(
                content='You are a helpful AI assistant.',
            ),
            UserPromptMessage(
                content="what's the weather today in London?",
            )
        ],
        tools=[
            PromptMessageTool(
                name='get_weather',
                description='Determine weather in my location',
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            ),
        ],
        model_parameters={
            'temperature': 0.0,
            'max_tokens': 1024
        },
        stream=False,
        user="abc-123"
    )

    assert isinstance(result, LLMResult)
    assert isinstance(result.message, AssistantPromptMessage)
    assert len(result.message.tool_calls) > 0


def test_get_num_tokens():
    model = LixiangLargeLanguageModel()

    num_tokens = model.get_num_tokens(
        model='gpt4-turbo',
        credentials={
            'api_base_url': 'http://api-hub.inner.chj.cloud/bcs-apihub-ai-proxy-service/apihub/openai/chat/completions',
            'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2RXp0NWdOU0ZVem54ZUlOWXVTUTg4cmNHU05uYU9JTiJ9.ecBJSGWxEtplXhbigaTurAvz6T7LCu-gObgj6hIZ-Q0',
        },
        prompt_messages=[
            SystemPromptMessage(
                content='You are a helpful AI assistant.',
            ),
            UserPromptMessage(
                content='Hello World!'
            )
        ]
    )

    assert isinstance(num_tokens, int)
    assert num_tokens == 21
