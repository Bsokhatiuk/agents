from agent import agent
from config_logging import (
    log_settings,
    filter_payload,
    format_tool_call,
    format_tool_done,
    format_tool_error,
    format_tool_result,
    format_tool_start,
    preview,
)


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

        if log_settings.show_agent_status:
            print("[agent] Understood the task. Starting research...")

        saved_path = None
        final_text = None

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "user_1"}},
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            if chunk["type"] == "updates":
                model_data = chunk["data"].get("model", {})

                tool_calls = model_data.get("tool_calls", [])
                if tool_calls and log_settings.show_plan:
                    for tool_call in tool_calls:
                        print(format_tool_call(tool_call["name"], tool_call.get("args", {})))

                text = model_data.get("final_text")
                if text:
                    final_text = text

            elif chunk["type"] == "custom":
                data = chunk["data"]
                event = data.get("event")

                if event == "tool_start" and log_settings.show_tool_start:
                    tool = data.get("tool", "unknown_tool")
                    payload = filter_payload(data, exclude_keys={"event", "tool"})
                    print(format_tool_start(tool, payload))

                elif event == "tool_done" and log_settings.show_tool_done:
                    tool = data.get("tool", "unknown_tool")
                    payload = filter_payload(data, exclude_keys={"event", "tool"})
                    print(format_tool_done(tool, payload))

                elif event == "tool_result" and log_settings.show_tool_result:
                    tool = data.get("tool", "unknown_tool")
                    result = data.get("result")
                    print(format_tool_result(tool, result))

                elif event == "tool_error" and log_settings.show_tool_errors:
                    tool = data.get("tool", "unknown_tool")
                    error = data.get("error")
                    print(format_tool_error(tool, error))

                elif event == "report_saved":
                    saved_path = data.get("path")
                    if log_settings.show_saved_path and saved_path:
                        print(f"[saved] {saved_path}")

                # інші події можна тихо ігнорувати
                else:
                    pass

        if saved_path:
            print(f"\nAgent: Report saved to {saved_path}")
        elif final_text and log_settings.show_final_text:
            print(f"\nAgent: {preview(final_text, 1500)}")
        else:
            print("\nAgent: Finished, but no final text or saved report path was detected.")


if __name__ == "__main__":
    main()