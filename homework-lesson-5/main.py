import json
from agent import agent


def _short_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("[agent] Understood the task. Starting research...")

        saved_path = None

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "user_1"}},
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            # 1) MODEL/TOOLS updates
            if chunk["type"] == "updates":
                for step, data in chunk["data"].items():
                    messages = data.get("messages", [])
                    if not messages:
                        continue

                    msg = messages[-1]
                    blocks = getattr(msg, "content_blocks", []) or []

                    # Показуємо тільки tool calls, а не фінальний текст звіту
                    if step == "model":
                        for block in blocks:
                            if block.get("type") == "tool_call":
                                tool_name = block.get("name")
                                tool_args = block.get("args", {})

            # 2) CUSTOM progress from tools
            elif chunk["type"] == "custom":
                data = chunk["data"]

                if not isinstance(data, dict):
                    print(f"[log] {data}")
                    continue

                event = data.get("event")

                if event == "tool_start":
                    tool = data.get("tool")
                    payload = {k: v for k, v in data.items() if k not in {"event", "tool"}}
                    if payload:
                        print(f"[tool:start] {tool} {_short_json(payload)}")
                    else:
                        print(f"[tool:start] {tool}")

                elif event == "tool_done":
                    tool = data.get("tool")
                    payload = {k: v for k, v in data.items() if k not in {"event", "tool"}}
                    if payload:
                        print(f"[tool:done]  {tool} {_short_json(payload)}")
                    else:
                        print(f"[tool:done]  {tool}")

                elif event == "report_saved":
                    saved_path = data.get("path")
                    print(f"[saved] {saved_path}")

                else:
                    print(f"[log] {_short_json(data)}")

        if saved_path:
            print(f"\nAgent: Report saved to {saved_path}")
        else:
            print("\nAgent: Finished, but no saved report path was detected.")


if __name__ == "__main__":
    main()