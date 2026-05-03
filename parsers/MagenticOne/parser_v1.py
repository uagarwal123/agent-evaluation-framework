import re
from pathlib import Path


TRACE_START = "SCENARIO.PY STARTING !#!#"
SEPARATOR = re.compile(r"^---------- (.+?) ----------$")
FINAL_ANSWER = re.compile(r"FINAL ANSWER:\s*(.+)")
RUNTIME = re.compile(r"SCENARIO\.PY RUNTIME:\s*(\d+)")
NOISE_LINE = re.compile(r"^(SCENARIO\.PY|RUN\.SH) (COMPLETE|RUNTIME:|STARTING)")


def _parse_messages(console_log: str) -> tuple[list[dict], str | None, str | None, int | None]:
    """Split console_log text into messages after the trace start marker.

    Returns (messages, task, final_answer, runtime_seconds).
    The task is the content of the first 'user' message and is not included in messages.
    """
    start = console_log.find(TRACE_START)
    if start == -1:
        return [], None, None, None
    trace_text = console_log[start + len(TRACE_START):]

    messages = []
    current_agent = None
    current_lines = []
    task = None
    final_answer = None
    runtime_seconds = None

    for line in trace_text.splitlines():
        sep_match = SEPARATOR.match(line)

        if sep_match:
            if current_agent is not None:
                content = "\n".join(current_lines).strip()
                if current_agent == "user" and task is None:
                    task = content
                else:
                    messages.append({"agent": current_agent, "content": content})
            current_agent = sep_match.group(1)
            current_lines = []
        else:
            rt_match = RUNTIME.match(line)
            if rt_match:
                runtime_seconds = int(rt_match.group(1))

            if NOISE_LINE.match(line):
                continue
            current_lines.append(line)

            fa_match = FINAL_ANSWER.match(line)
            if fa_match:
                final_answer = fa_match.group(1).strip()

    if current_agent is not None:
        content = "\n".join(current_lines).strip()
        if current_agent == "user" and task is None:
            task = content
        else:
            messages.append({"agent": current_agent, "content": content})

    return messages, task, final_answer, runtime_seconds


def _level_from_path(trace_dir: Path) -> int | None:
    """Extract difficulty level (1/2/3) from the parent folder name."""
    for part in trace_dir.parts:
        match = re.search(r"level_(\d)", part)
        if match:
            return int(match.group(1))
    return None


def load_trace(trace_dir: Path) -> dict:
    """Parse one trace folder into a list of dicts (one key per dict)."""
    trace_dir = Path(trace_dir)
    run_dir = trace_dir / "0"

    console_log = (run_dir / "console_log.txt").read_text(encoding="utf-8", errors="replace")
    expected_answer = (run_dir / "expected_answer.txt").read_text(encoding="utf-8", errors="replace").strip()

    messages, task, final_answer, runtime_seconds = _parse_messages(console_log)

    return {
        "trace_id": trace_dir.name,
        "level": _level_from_path(trace_dir),
        "task": task,
        "expected_answer": expected_answer,
        "final_answer": final_answer,
        "runtime_seconds": runtime_seconds,
        "messages": messages,
    }


def load_all_traces(magenticone_dir: Path) -> list[dict]:
    """Load all traces from the MagenticOne_GAIA directory."""

    magenticone_dir = Path(magenticone_dir)
    traces = []

    for level_dir in sorted(magenticone_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        for trace_dir in sorted(level_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            traces.append(load_trace(trace_dir))

    return traces


if __name__ == "__main__":
    import json
    here = Path(__file__).parent
    data_dir = "parsers\MagenticOne\MagenticOne_GAIA"
    output_file = here / "output_v1.json"

    print(f"Parsing traces from: {data_dir}")
    traces = load_all_traces(data_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"Done — wrote {len(traces)} traces to {output_file}")

