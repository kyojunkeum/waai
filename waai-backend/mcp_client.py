import os
import httpx

MCP_FILESYSTEM_URL = os.environ.get("MCP_FILESYSTEM_URL", "http://mcp-filesystem:7001")
# backward compatibility: fall back to old MCP_DIARY_URL if provided
MCP_BRIDGE_URL = os.environ.get(
    "MCP_BRIDGE_URL",
    os.environ.get("MCP_DIARY_URL", "http://mcp-bridge:7002"),
)

async def list_diary_files():
  async with httpx.AsyncClient(timeout=30.0) as client:
    r = await client.get(f"{MCP_FILESYSTEM_URL}/files")
    r.raise_for_status()
    return r.json().get("files", [])

async def summarize_diary(keyword: str | None, start_date: str | None,
                          end_date: str | None, mode: str,
                          topic: str | None = None, extra_instruction: str | None = None):
    payload = {
        "keyword": keyword or None,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "mode": mode,
        "topic": topic or None,
        "extra_instruction": extra_instruction or None,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{MCP_BRIDGE_URL}/summarize", json=payload)
        r.raise_for_status()
        return r.json().get("result", "")


async def select_and_summarize(include: list[str],
                               start_date: str | None = None,
                               end_date: str | None = None,
                               keyword: str | None = None,
                               extra_instruction: str | None = None,
                               limit_per_type: int = 10,
                               preview_chars: int = 1200):
    payload = {
        "include": include,
        "start_date": start_date,
        "end_date": end_date,
        "keyword": keyword,
        "limit_per_type": limit_per_type,
        "preview_chars": preview_chars,
        "extra_instruction": extra_instruction,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{MCP_BRIDGE_URL}/select-and-summarize", json=payload)
        r.raise_for_status()
        return r.json()

async def get_mood_stats(start_date: str | None = None, end_date: str | None = None):
    payload = {
        "start_date": start_date or None,
        "end_date": end_date or None,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MCP_BRIDGE_URL}/mood-stats", json=payload)
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
        r = await client.post(f"{MCP_BRIDGE_URL}/project-timeline", json=payload)
        r.raise_for_status()
        return r.json().get("projects", {})
