import json

import httpx
import pytest

from app.llm.providers import AnthropicLLM, LLMError, OpenAICompatibleLLM


def _mock(llm, handler):
    llm._client = httpx.Client(base_url=str(llm._client.base_url), transport=httpx.MockTransport(handler))
    return llm


def test_anthropic_request_and_response_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Son 23 días [1]."}]})

    llm = _mock(AnthropicLLM("key", "claude-sonnet-4-5"), handler)
    assert llm.complete("sistema", "pregunta") == "Son 23 días [1]."
    assert seen["system"] == "sistema"
    assert seen["messages"] == [{"role": "user", "content": "pregunta"}]
    assert seen["temperature"] == 0


def test_openai_compatible_request_and_response_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert body["messages"][0] == {"role": "system", "content": "sistema"}
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hola "}}]})

    llm = _mock(OpenAICompatibleLLM("key", "gpt-4o-mini", "https://openrouter.ai/api/v1"), handler)
    assert llm.complete("sistema", "pregunta") == "Hola"


def test_api_errors_are_reported():
    llm = _mock(AnthropicLLM("key", "m"), lambda request: httpx.Response(401, text="invalid key"))
    with pytest.raises(LLMError, match="401"):
        llm.complete("s", "u")


def test_missing_key_fails_fast():
    with pytest.raises(ValueError):
        AnthropicLLM("", "m")


def test_azure_openai_uses_deployment_url_api_version_and_api_key_header():
    from app.llm.providers import AzureOpenAILLM

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/deployments/gpt-4o-mini/chat/completions"
        assert request.url.params["api-version"] == "2024-10-21"
        assert request.headers["api-key"] == "secret"
        assert "model" not in json.loads(request.content)  # Azure picks the model by deployment
        return httpx.Response(200, json={"choices": [{"message": {"content": "Respuesta [1]"}}]})

    llm = AzureOpenAILLM("https://demo.openai.azure.com", "secret", "gpt-4o-mini", "2024-10-21")
    llm._client = httpx.Client(
        base_url="https://demo.openai.azure.com",
        headers={"api-key": "secret"},
        transport=httpx.MockTransport(handler),
    )
    assert llm.complete("s", "u") == "Respuesta [1]"


def test_azure_openai_embeddings(monkeypatch):
    from app.retrieval.embeddings import AzureOpenAIEmbedder

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/deployments/emb/embeddings"
        texts = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [1.0, float(i), 0.0]} for i in range(len(texts))]
        return httpx.Response(200, json={"data": data})

    real_client = httpx.Client

    def fake_client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", fake_client)
    embedder = AzureOpenAIEmbedder("https://demo.openai.azure.com", "secret", "emb", "2024-10-21")
    vectors = embedder.embed(["a", "b"])
    assert embedder.dim == 3 and vectors.shape == (2, 3)
