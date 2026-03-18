from __future__ import annotations

import json
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    show_agent_status: bool = True
    show_plan: bool = True

    show_tool_start: bool = True
    show_tool_done: bool = True

    show_tool_args: bool = True
    show_tool_result: bool = False
    show_tool_errors: bool = True

    show_saved_path: bool = True
    show_final_text: bool = True

    max_preview_chars: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOG_",
        extra="ignore",
    )


log_settings = LoggingSettings()


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def preview(value: Any, max_chars: int | None = None) -> str:
    text = _to_text(value).strip()
    limit = max_chars or log_settings.max_preview_chars

    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."


def filter_payload(
    data: dict[str, Any],
    *,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    exclude_keys = exclude_keys or set()
    return {k: v for k, v in data.items() if k not in exclude_keys}


def format_tool_call(name: str, args: dict[str, Any]) -> str:
    if log_settings.show_tool_args:
        return f"[plan] {name}({preview(args)})"
    return f"[plan] {name}()"


def format_tool_start(tool: str, payload: dict[str, Any]) -> str:
    if log_settings.show_tool_args and payload:
        return f"[tool:start] {tool} {preview(payload)}"
    return f"[tool:start] {tool}"


def format_tool_done(tool: str, payload: dict[str, Any]) -> str:
    if payload:
        return f"[tool:done]  {tool} {preview(payload)}"
    return f"[tool:done]  {tool}"


def format_tool_result(tool: str, result: Any) -> str:
    return f"[tool:result] {tool} {preview(result)}"


def format_tool_error(tool: str, error: Any) -> str:
    return f"[tool:error] {tool} {preview(error)}"