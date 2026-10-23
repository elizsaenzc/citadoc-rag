"""LLM providers behind a single small interface.

Calls go straight to the HTTP APIs with httpx (no vendor SDKs), so adding a
provider is one class. ``demo`` mode has no LLM at all: the RAG service then
builds an extractive answer from the retrieved text.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.config import Settings


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str: ...


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("CITADOC_ANTHROPIC_API_KEY is required")
        self.model = model
        self._client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=90,
        )

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self._client.post(
            "/messages",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        if response.status_code >= 400:
            raise LLMError(f"Anthropic API error {response.status_code}: {response.text[:300]}")
        blocks = response.json()["content"]
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


class OpenAICompatibleLLM:
    """OpenAI, OpenRouter, Ollama or any server with the OpenAI chat API."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            raise ValueError("CITADOC_OPENAI_API_KEY is required")
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
        )

    def _chat_request(self, body: dict) -> httpx.Response:
        return self._client.post("/chat/completions", json={"model": self.model, **body})

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self._chat_request(
            {
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        )
        if response.status_code >= 400:
            raise LLMError(f"LLM API error {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"].strip()


class AzureOpenAILLM(OpenAICompatibleLLM):
    """Azure OpenAI: the model is chosen by the *deployment* name, auth uses the api-key header."""

    name = "azure"

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str):
        if not (endpoint and api_key and deployment):
            raise ValueError(
                "CITADOC_AZURE_OPENAI_ENDPOINT, CITADOC_AZURE_OPENAI_API_KEY and "
                "CITADOC_AZURE_OPENAI_CHAT_DEPLOYMENT are required"
            )
        self.deployment = deployment
        self.api_version = api_version
        self._client = httpx.Client(
            base_url=endpoint.rstrip("/"), headers={"api-key": api_key}, timeout=90
        )

    def _chat_request(self, body: dict) -> httpx.Response:
        return self._client.post(
            f"/openai/deployments/{self.deployment}/chat/completions",
            params={"api-version": self.api_version},
            json=body,
        )


def build_llm(settings: Settings) -> LLM | None:
    """Return the configured LLM, or None in demo mode."""
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(
            settings.openai_api_key, settings.llm_model, settings.openai_base_url
        )
    if settings.llm_provider == "azure":
        return AzureOpenAILLM(
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_chat_deployment,
            settings.azure_openai_api_version,
        )
    return None
