import json
from collections import deque
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import settings, SYSTEM_PROMPT
from tools import OPENAI_TOOLS, execute_tool_call, set_event_sink


class ResponsesResearchAgent:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        system_prompt: str,
        max_iterations: int = 10,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # проста пам'ять по thread_id
        self._previous_response_ids: Dict[str, str] = {}

    def _make_update(
        self,
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        final_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        if final_text is not None:
            payload["final_text"] = final_text

        return {
            "type": "updates",
            "data": {
                "model": payload
            },
        }

    def _serialize_tool_output(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _extract_user_text(self, inputs: Dict[str, Any]) -> str:
        messages = inputs.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
                return str(msg.get("content", "")).strip()
        return ""

    def _extract_function_calls(self, response) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type != "function_call":
                continue

            raw_args = getattr(item, "arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except Exception:
                args = {}

            calls.append(
                {
                    "call_id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "args": args,
                }
            )

        return calls

    def _extract_final_text(self, response) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue

            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) == "output_text":
                    value = getattr(block, "text", "")
                    if value:
                        parts.append(value)

        return "\n".join(parts).strip()

    def stream(
        self,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        stream_mode: Optional[List[str]] = None,
        version: Optional[str] = None,
    ):
        configurable = (config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id", "default")

        user_text = self._extract_user_text(inputs)
        if not user_text:
            return

        previous_response_id = self._previous_response_ids.get(thread_id)
        event_queue = deque()

        def event_sink(payload: Dict[str, Any]) -> None:
            event_queue.append({"type": "custom", "data": payload})

        response_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "instructions": self.system_prompt,
            "input": [{"role": "user", "content": user_text}],
            "tools": OPENAI_TOOLS,
        }
        if previous_response_id:
            response_kwargs["previous_response_id"] = previous_response_id

        response = self.client.responses.create(**response_kwargs)

        for _ in range(self.max_iterations):
            tool_calls = self._extract_function_calls(response)

            if not tool_calls:
                final_text = self._extract_final_text(response)
                self._previous_response_ids[thread_id] = response.id
                yield self._make_update(final_text=final_text or "Finished.")
                return

            yield self._make_update(tool_calls=tool_calls)

            tool_outputs = []
            set_event_sink(event_sink)

            try:
                for tool_call in tool_calls:
                    result = execute_tool_call(tool_call["name"], tool_call["args"])

                    while event_queue:
                        yield event_queue.popleft()

                    is_error = isinstance(result, str) and result.startswith("Error:")

                    yield {
                        "type": "custom",
                        "data": {
                            "event": "tool_error" if is_error else "tool_result",
                            "tool": tool_call["name"],
                            "error": result if is_error else None,
                            "result": None if is_error else result,
                        },
                    }

                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": self._serialize_tool_output(result),
                        }
                    )
            finally:
                set_event_sink(None)

            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.system_prompt,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=OPENAI_TOOLS,
            )

        yield self._make_update(
            final_text=f"Stopped after {self.max_iterations} iterations without a final answer."
        )


agent = ResponsesResearchAgent(
    model_name=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
    system_prompt=SYSTEM_PROMPT,
    max_iterations=settings.max_iterations,
)