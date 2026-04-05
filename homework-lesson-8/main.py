import json
from supervisor import supervisor
from langgraph.types import Command
from config import settings


def _short_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def _content_preview(content: str) -> str:
    lines = content.splitlines()
    preview_lines = lines[: settings.max_preview_lines]
    preview = "\n".join(preview_lines)
    if len(preview) > settings.max_preview_chars:
        preview = preview[: settings.max_preview_chars].rstrip() + "..."
    if len(lines) > settings.max_preview_lines:
        preview += f"\n  ... ({len(lines) - settings.max_preview_lines} more lines)"
    return preview


def _process_chunk(chunk) -> str | None:
    """Print streaming chunk info, return saved_path if report was saved."""
    saved_path = None

    if chunk["type"] == "updates":
        for step, data in chunk["data"].items():
            if step == "__interrupt__":
                continue
            if not isinstance(data, dict):
                continue
            messages = data.get("messages", [])
            if not messages:
                continue
            msg = messages[-1]
            blocks = getattr(msg, "content_blocks", []) or []
            if step == "model":
                for block in blocks:
                    if block.get("type") == "tool_call":
                        tool_name = block.get("name", "")
                        tool_args = block.get("args", {})
                        args_str = _short_json(tool_args)
                        if len(args_str) > settings.max_inline_chars:
                            args_str = args_str[: settings.max_inline_chars] + "..."
                        if tool_name in settings.top_level_tools:
                            agent_label = settings.display_agent_name.get(tool_name, tool_name)
                            print(f"\n[Supervisor → {agent_label}]")
                        print(f"  🔧 {tool_name}({args_str})")

    elif chunk["type"] == "custom":
        data = chunk["data"]
        if not isinstance(data, dict):
            print(f"[log] {data}")
            return None

        event = data.get("event")
        if event == "tool_start":
            tool = data.get("tool")
            payload = {k: v for k, v in data.items() if k not in {"event", "tool"}}
            if payload:
                print(f"  🔧 {tool} {_short_json(payload)}")
            else:
                print(f"  🔧 {tool}")
        elif event == "tool_done":
            tool = data.get("tool")
            payload = {k: v for k, v in data.items() if k not in {"event", "tool"}}
            if payload:
                print(f"  📎 {tool} {_short_json(payload)}")
        elif event == "report_saved":
            saved_path = data.get("path")
        else:
            print(f"[log] {_short_json(data)}")

    return saved_path


def _find_interrupt(chunk) -> list | None:
    """Extract interrupt list from a chunk, if any."""
    if chunk.get("type") == "interrupt":
        data = chunk.get("data", [])
        return data if data else None
    if chunk.get("type") == "updates":
        interrupts = chunk["data"].get("__interrupt__")
        if interrupts:
            return interrupts
    return None


def _extract_tool_args(interrupt_data) -> tuple[str, str]:
    """Extract filename and content from interrupt data.

    interrupt_data is a list/tuple of Interrupt dataclass objects (or dicts).
    Each Interrupt has .value = ActionRequest(name, args, description).
    """
    items = list(interrupt_data) if isinstance(interrupt_data, (list, tuple)) else [interrupt_data]
    for item in items:
        # Interrupt dataclass → .value; plain dict → ["value"]
        if isinstance(item, dict):
            value = item.get("value", item)
        else:
            value = getattr(item, "value", item)

        if not isinstance(value, dict):
            continue

        # ActionRequest: {"name": "write_report", "args": {"filename": ..., "content": ...}}
        args = value.get("args")
        if isinstance(args, dict) and "filename" in args:
            return args["filename"], args.get("content", "")

        # Flat fallback: value itself has filename/content
        if "filename" in value:
            return value["filename"], value.get("content", "")

    return "report.md", ""


def _stream_graph(input_or_command, thread_id: str) -> tuple[str | None, list | None]:
    """Stream supervisor until completion or interrupt.

    Returns (saved_path, interrupt_data).
    """
    saved_path = None
    interrupt_data = None

    for chunk in supervisor.stream(
        input_or_command,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        result = _process_chunk(chunk)
        if result:
            saved_path = result

        interrupt = _find_interrupt(chunk)
        if interrupt:
            interrupt_data = interrupt
            break

    return saved_path, interrupt_data


def _show_approval_ui(interrupt_data: list) -> None:
    filename, content = _extract_tool_args(interrupt_data)
    args_preview = f'{{"filename": "{filename}", "content": "{content[:50]}..."}}'
    print()
    print("  " + "=" * 58)
    print("  ⏸️  ACTION REQUIRES APPROVAL")
    print("  " + "=" * 58)
    print(f"    Tool:  write_report")
    print(f"    Args:  {args_preview}")
    print()
    print(f"  Preview of '{filename}':")
    print("  " + "-" * 40)
    for line in _content_preview(content).splitlines():
        print(f"    {line}")
    print()


def _handle_hitl_loop(thread_id: str, interrupt_data: list) -> str | None:
    """Interactive HITL loop. Returns saved_path or None."""
    while True:
        _show_approval_ui(interrupt_data)

        try:
            action = input("  👉 approve / edit / reject: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            action = "reject"

        if action == "approve":
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
            saved_path, new_interrupt = _stream_graph(cmd, thread_id)
            if new_interrupt:
                interrupt_data = new_interrupt
                continue
            if saved_path:
                print(f"\n  ✅ Approved! Report saved to {saved_path}")
            return saved_path

        elif action == "edit":
            try:
                feedback = input("  ✏️  Your feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""

            # Reject with feedback message so Supervisor sees it as a ToolMessage
            # and revises the report accordingly before calling write_report again.
            print("\n[Supervisor revises report based on feedback]")
            cmd = Command(
                resume={"decisions": [{"type": "reject", "message": f"Please revise the report based on this feedback: {feedback}"}]}
            )
            saved_path, new_interrupt = _stream_graph(cmd, thread_id)
            if new_interrupt:
                interrupt_data = new_interrupt
                continue
            if saved_path:
                print(f"\n  ✅ Approved! Report saved to {saved_path}")
            return saved_path

        elif action == "reject":
            print("\n  ❌ Rejected! Report will not be saved.")
            cmd = Command(
                resume={"decisions": [{"type": "reject", "message": "User rejected the report"}]}
            )
            _stream_graph(cmd, thread_id)
            return None

        else:
            print("  Please enter 'approve', 'edit', or 'reject'")


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    thread_id = "user_1"

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

        saved_path, interrupt_data = _stream_graph(
            {"messages": [{"role": "user", "content": user_input}]},
            thread_id,
        )

        if interrupt_data:
            saved_path = _handle_hitl_loop(thread_id, interrupt_data)

        if saved_path:
            print(f"\nAgent: Report saved to {saved_path}")
        else:
            print("\nAgent: Finished.")


if __name__ == "__main__":
    main()
