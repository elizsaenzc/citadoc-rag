from app.rag.service import NOT_FOUND


def test_sample_documents_are_loaded(container):
    names = {d.name for d in container.rag.documents.values()}
    assert "manual_seguridad_obra.md" in names


def test_answer_is_grounded_and_cited(container):
    result = container.rag.ask("¿Cuántos días de vacaciones hay al año?")
    assert result.grounded
    assert "23 días" in result.answer
    assert "[1]" in result.answer
    assert result.sources[0].doc_name == "politica_vacaciones_teletrabajo.md"


def test_out_of_scope_question_is_refused(container):
    result = container.rag.ask("¿Cuál es la capital de Francia?")
    assert not result.grounded
    assert result.answer == NOT_FOUND
    assert result.sources == []


def test_search_can_be_restricted_to_one_document(container):
    compras = next(d for d in container.rag.documents.values() if "compras" in d.name)
    result = container.rag.ask("¿Quién aprueba los pedidos?", doc_ids=[compras.id])
    assert all(s.doc_id == compras.id for s in result.sources)


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        self.prompts.append(user)
        return self.reply


def test_llm_answer_drops_citations_to_missing_sources(container):
    container.rag.llm = FakeLLM("Son 23 días [1], ampliables [9].")
    result = container.rag.ask("¿Cuántos días de vacaciones hay?")
    assert result.answer == "Son 23 días [1], ampliables ."
    assert [s.ref for s in result.sources] == [1]
    # the retrieved fragments reach the model, numbered
    assert "[1] (politica_vacaciones_teletrabajo.md)" in container.rag.llm.prompts[0]


def test_llm_saying_not_found_is_not_grounded(container):
    container.rag.llm = FakeLLM(NOT_FOUND)
    result = container.rag.ask("¿Cuántos días de vacaciones hay?")
    assert not result.grounded and result.sources == []


def test_agent_routes_to_the_right_tool_in_demo_mode(container):
    assert container.agent.run("¿qué documentos hay?").steps[0].tool == "list_documents"
    summary = container.agent.run("resume el manual de seguridad")
    assert summary.steps[0].tool == "summarize_document"
    assert "manual_seguridad_obra.md" in summary.answer
    comparison = container.agent.run("compara la política de vacaciones con el procedimiento de compras")
    assert comparison.steps[0].tool == "compare_documents"
    assert container.agent.run("¿Cuándo es obligatorio el arnés?").steps[0].tool == "search_documents"


def test_agent_with_llm_uses_tools_then_answers(container):
    # 1) the agent picks a tool, 2) the search tool asks the LLM for a cited answer,
    # 3) the agent gives the final answer
    replies = iter(
        [
            '{"tool": "search_documents", "input": {"query": "plazo para comunicar un accidente"}}',
            "En un plazo máximo de 24 horas [1].",
            '{"final": "Hay que comunicarlo en 24 horas [1]."}',
        ]
    )

    class ScriptedLLM:
        name = "scripted"

        def complete(self, system, user, max_tokens=800):
            return next(replies)

    container.rag.llm = ScriptedLLM()
    result = container.agent.run("¿En cuánto tiempo hay que avisar de un accidente?")
    assert [s.tool for s in result.steps] == ["search_documents"]
    assert result.answer == "Hay que comunicarlo en 24 horas [1]."
    assert result.sources


def test_llm_rewrites_follow_up_questions_before_searching(container):
    class RecordingLLM:
        name = "recording"

        def __init__(self):
            self.calls = []

        def complete(self, system, user, max_tokens=800):
            self.calls.append(user)
            if len(self.calls) == 1:  # condense step
                return "¿Cuántos días de vacaciones tienen los empleados a jornada parcial?"
            return "En proporción a su jornada [1]."

    container.rag.llm = RecordingLLM()
    history = [("user", "¿Cuántos días de vacaciones hay?"), ("assistant", "23 días [1].")]
    result = container.rag.ask("¿y a jornada parcial?", history=history)
    assert result.search_query.startswith("¿Cuántos días de vacaciones tienen")
    assert "Última pregunta: ¿y a jornada parcial?" in container.rag.llm.calls[0]
    assert result.grounded
