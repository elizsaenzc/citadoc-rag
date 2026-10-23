def test_health_reports_demo_mode(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "demo"
    assert body["documents"] >= 3


def test_upload_ask_and_delete_a_document(client):
    content = b"El horario de verano es de 8:00 a 15:00 durante julio y agosto."
    response = client.post("/api/documents", files={"file": ("horario_verano.txt", content, "text/plain")})
    assert response.status_code == 201
    doc = response.json()
    assert doc["name"] == "horario_verano.txt" and doc["chunks"] == 1

    answer = client.post(
        "/api/ask", json={"question": "¿Cuál es el horario de verano?", "doc_ids": [doc["id"]]}
    ).json()
    assert answer["grounded"]
    assert "8:00 a 15:00" in answer["answer"]
    assert answer["sources"][0]["doc_name"] == "horario_verano.txt"

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_unsupported_file_type(client):
    response = client.post("/api/documents", files={"file": ("datos.xlsx", b"x", "application/octet-stream")})
    assert response.status_code == 415


def test_question_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_agent_endpoint_returns_steps(client):
    body = client.post("/api/agent", json={"question": "¿Qué documentos hay?"}).json()
    assert body["steps"][0]["tool"] == "list_documents"


def test_conversation_is_persisted_and_follow_up_uses_history(client):
    first = client.post("/api/ask", json={"question": "¿Cuántos días de vacaciones hay al año?"}).json()
    conversation_id = first["conversation_id"]
    assert conversation_id and first["grounded"]

    follow_up = client.post(
        "/api/ask", json={"question": "¿y cuántos se pueden trasladar?", "conversation_id": conversation_id}
    ).json()
    assert follow_up["conversation_id"] == conversation_id
    assert follow_up["search_query"]  # the short follow-up was expanded with the previous question
    assert "5 días" in follow_up["answer"]

    listed = client.get("/api/conversations").json()
    assert listed[0]["id"] == conversation_id and listed[0]["messages"] == 4
    assert listed[0]["title"].startswith("¿Cuántos días de vacaciones")

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["sources"][0]["doc_name"] == "politica_vacaciones_teletrabajo.md"

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_unknown_conversation_is_rejected(client):
    response = client.post("/api/ask", json={"question": "hola hola", "conversation_id": "nope"})
    assert response.status_code == 404


def test_swagger_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/ask", "/api/documents", "/api/conversations"} <= set(paths)
