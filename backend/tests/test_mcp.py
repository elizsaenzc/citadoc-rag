import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BACKEND = Path(__file__).resolve().parent.parent


def test_mcp_server_exposes_tools_and_answers(tmp_path):
    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            cwd=str(BACKEND),
            env={**os.environ, "CITADOC_DATA_DIR": str(tmp_path)},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            question = {"question": "¿Cuántas horas dura el curso de PRL?"}
            result = await session.call_tool("ask_documents", question)
            return tools, result.content[0].text

    tools, answer = asyncio.run(run())
    assert {"list_documents", "search_documents", "ask_documents", "summarize_document"} <= tools
    assert "20 horas" in answer
