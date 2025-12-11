import os
import httpx

MCP_FILESYSTEM_URL = os.environ.get("MCP_FILESYSTEM_URL", "http://mcp-filesystem:7001")
MCP_DIARY_URL = os.environ.get("MCP_DIARY_URL", "http://mcp-diary:7002")

async def list_diary_files():
  async with httpx.AsyncClient(timeout=30.0) as client:
    r = await client.get(f"{MCP_FILESYSTEM_URL}/files")
    r.raise_for_status()
    return r.json().get("files", [])

async def summarize_diary(keyword: str | None, start_date: str | None,
                          end_date: str | None, mode: str):
    payload = {
        "keyword": keyword or None,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "mode": mode,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{MCP_DIARY_URL}/summarize", json=payload)
        r.raise_for_status()
        return r.json().get("result", "")

async def get_mood_stats(start_date: str | None = None, end_date: str | None = None):
    payload = {
        "start_date": start_date or None,
        "end_date": end_date or None,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MCP_DIARY_URL}/mood-stats", json=payload)
        r.raise_for_status()
        return r.json().get("stats", [])

async def get_project_timeline(projects: list[str],
                               start_date: str | None = None,
                               end_date: str | None = None):
    payload = {
        "projects": projects,
        "start_date": start_date or None,
        "end_date": end_date or None,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MCP_DIARY_URL}/project-timeline", json=payload)
        r.raise_for_status()
        return r.json().get("projects", {})
