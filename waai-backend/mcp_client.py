import os
import httpx

from utils import monitor_log

MCP_FILESYSTEM_URL = os.environ.get("MCP_FILESYSTEM_URL", "http://mcp-filesystem:7001")
# backward compatibility: fall back to old MCP_DIARY_URL if provided
MCP_BRIDGE_URL = os.environ.get(
    "MCP_BRIDGE_URL",
    os.environ.get("MCP_DIARY_URL", "http://mcp-bridge:7002"),
)

async def list_diary_files():
  url = f"{MCP_FILESYSTEM_URL}/files"
  monitor_log("api_call", "mcp-filesystem list start", {"method": "GET", "url": url})
  try:
    async with httpx.AsyncClient(timeout=30.0) as client:
      r = await client.get(url)
      r.raise_for_status()
      files = r.json().get("files", [])
    monitor_log(
      "api_call",
      "mcp-filesystem list ok",
      {"method": "GET", "url": url, "status_code": r.status_code, "count": len(files)},
    )
    return files
  except Exception as exc:
    monitor_log(
      "api_call",
      "mcp-filesystem list failed",
      {"method": "GET", "url": url, "error": str(exc)},
    )
    raise

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
    url = f"{MCP_BRIDGE_URL}/summarize"
    monitor_log("api_call", "mcp-bridge summarize start", {"method": "POST", "url": url})
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            result = r.json().get("result", "")
        monitor_log(
            "api_call",
            "mcp-bridge summarize ok",
            {"method": "POST", "url": url, "status_code": r.status_code},
        )
        return result
    except Exception as exc:
        monitor_log(
            "api_call",
            "mcp-bridge summarize failed",
            {"method": "POST", "url": url, "error": str(exc)},
        )
        raise


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
    url = f"{MCP_BRIDGE_URL}/select-and-summarize"
    monitor_log("api_call", "mcp-bridge select start", {"method": "POST", "url": url})
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            result = r.json()
        monitor_log(
            "api_call",
            "mcp-bridge select ok",
            {"method": "POST", "url": url, "status_code": r.status_code},
        )
        return result
    except Exception as exc:
        monitor_log(
            "api_call",
            "mcp-bridge select failed",
            {"method": "POST", "url": url, "error": str(exc)},
        )
        raise

async def get_mood_stats(start_date: str | None = None, end_date: str | None = None):
    payload = {
        "start_date": start_date or None,
        "end_date": end_date or None,
    }
    url = f"{MCP_BRIDGE_URL}/mood-stats"
    monitor_log("api_call", "mcp-bridge mood-stats start", {"method": "POST", "url": url})
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            result = r.json().get("stats", [])
        monitor_log(
            "api_call",
            "mcp-bridge mood-stats ok",
            {"method": "POST", "url": url, "status_code": r.status_code},
        )
        return result
    except Exception as exc:
        monitor_log(
            "api_call",
            "mcp-bridge mood-stats failed",
            {"method": "POST", "url": url, "error": str(exc)},
        )
        raise

async def get_project_timeline(projects: list[str],
                               start_date: str | None = None,
                               end_date: str | None = None):
    payload = {
        "projects": projects,
        "start_date": start_date or None,
        "end_date": end_date or None,
    }
    url = f"{MCP_BRIDGE_URL}/project-timeline"
    monitor_log("api_call", "mcp-bridge project-timeline start", {"method": "POST", "url": url})
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            result = r.json().get("projects", {})
        monitor_log(
            "api_call",
            "mcp-bridge project-timeline ok",
            {"method": "POST", "url": url, "status_code": r.status_code},
        )
        return result
    except Exception as exc:
        monitor_log(
            "api_call",
            "mcp-bridge project-timeline failed",
            {"method": "POST", "url": url, "error": str(exc)},
        )
        raise
