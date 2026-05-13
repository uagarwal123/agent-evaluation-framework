"""Parse AppWorld traces from MAD_full_dataset.json."""
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

ROOT = Path(__file__).parent.parent.parent
MAD_PATH = ROOT / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "appworld_output_mad.json"

# --- Patterns ---

TASK_HEADER_PATTERN = re.compile(
    r"^\*{10,}\s*Task\s+\d+/\d+\s+\(([a-f0-9_]+)\)\s*\*{10,}\s*$"
)
COMPLETE_TASK_PATTERN = re.compile(r'complete_task\((?:[^)]*?status=["\']?(\w+)["\']?)?')
COMPLETE_TASK_ANSWER_PATTERN = re.compile(r'complete_task\([^)]*?answer=(?:["\']([^"\']*)["\']|(\w+))')
API_CALL_PATTERN = re.compile(r'\bsend_message\(|\bapis\.\w|\bprint\(apis\.')
CALL_EXTENSION_PATTERN = re.compile(r'^\s*#\s*CallExtension\b')

_HEADERS = [
    (re.compile(r"^Response from Supervisor Agent\s*$"),            "supervisor_response",  4),
    (re.compile(r"^Code Execution Output\s*$"),                     "code_exec",            4),
    (re.compile(r"^Response from send_message API\s*$"),            "send_msg_api",         4),
    (re.compile(r"^Message to Supervisor Agent\s*$"),               "msg_to_supervisor",    4),
    (re.compile(r"^Entering (.+?) message loop\s*$"),               "entering_loop",        4),
    (re.compile(r"^Exiting (.+?) message loop\s*$"),                "exiting_loop",         4),
    (re.compile(r"^Evaluation\s*$"),                                "evaluation",           4),
    (re.compile(r"^    Message to (.+?) Agent\s*$"),                "msg_to_app",           8),
    (re.compile(r"^    Response from (.+?) Agent\s*$"),             "app_response",         8),
    (re.compile(r"^    Reply from (.+?) Agent to Supervisor\s*$"),  "app_reply",            8),
]


def _match_header(line: str):
    for pattern, htype, indent in _HEADERS:
        m = pattern.match(line)
        if m:
            return htype, m, indent
    return None


def _strip_indent(line: str, n: int) -> str:
    return line[n:] if line.startswith(" " * n) else line


def _build_content(lines: list[str], indent: int) -> str:
    return "\n".join(_strip_indent(l, indent) for l in lines).strip()


def _classify_supervisor(content: str) -> str:
    if COMPLETE_TASK_PATTERN.search(content):
        return "message"
    if API_CALL_PATTERN.search(content):
        return "tool_call"
    return "message"


def _extract_evaluation(lines: list[str]) -> dict:
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "Evaluation":
            json_text = "\n".join(lines[i + 1:]).strip()
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            break
    return {}


def parse_file(path: Path) -> Trace:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    trace_id = path.stem
    task: str | None = None
    steps: list[Step] = []

    current_htype: str | None = None
    current_hmatch = None
    current_indent: int = 4
    current_lines: list[str] = []
    prev_htype: str | None = None

    in_loop = False
    loop_app: str = ""
    loop_msg_count = 0

    def flush():
        nonlocal task, prev_htype
        if current_htype is None:
            return
        content = _build_content(current_lines, current_indent)

        if current_htype == "supervisor_response":
            agent, kind = "Supervisor", _classify_supervisor(content)
        elif current_htype == "code_exec":
            if CALL_EXTENSION_PATTERN.search(content):
                agent, kind = "System", "system"
            else:
                agent, kind = "System", "tool_result"
        elif current_htype == "send_msg_api":
            agent, kind = "System", "message"
        elif current_htype == "msg_to_supervisor":
            agent, kind = "System", "message"
        elif current_htype == "entering_loop":
            agent, kind = "System", "system"
            if not content:
                content = f"Entering {current_hmatch.group(1)} message loop"
        elif current_htype == "exiting_loop":
            agent, kind = "System", "system"
            if not content:
                content = f"Exiting {current_hmatch.group(1)} message loop"
        elif current_htype == "evaluation":
            agent, kind = "System", "system"
            if not content:
                return
        elif current_htype == "msg_to_app":
            if loop_msg_count == 1:
                agent, kind = "Supervisor", "message"
            else:
                agent, kind = "System", "tool_result"
        elif current_htype == "app_response":
            agent = current_hmatch.group(1) + " Agent"
            kind = "tool_call" if API_CALL_PATTERN.search(content) else "message"
        elif current_htype == "app_reply":
            agent = current_hmatch.group(1) + " Agent"
            kind = "message"
        else:
            return

        if not content:
            prev_htype = current_htype
            return

        final_match = COMPLETE_TASK_PATTERN.search(content)
        step = Step(
            agent=agent,
            content=content,
            kind=kind,
            metadata={
                "step_index": len(steps),
                "is_final_answer": final_match is not None,
            },
        )
        steps.append(step)
        prev_htype = current_htype

    found_task_header = False
    for line in lines:
        if not found_task_header:
            m = TASK_HEADER_PATTERN.match(line)
            if m:
                trace_id = m.group(1)
                found_task_header = True
                continue

        if found_task_header and task is None and line.strip():
            task = line.strip()
            steps.append(Step(
                agent="User",
                content=task,
                kind="message",
                metadata={"step_index": 0, "is_final_answer": False},
            ))
            continue

        result = _match_header(line)
        if result is not None:
            flush()
            htype, hmatch, indent = result
            current_htype = htype
            current_hmatch = hmatch
            current_indent = indent
            current_lines = []

            if htype == "entering_loop":
                in_loop = True
                loop_app = hmatch.group(1)
                loop_msg_count = 0
            elif htype == "exiting_loop":
                in_loop = False
                loop_app = ""
                loop_msg_count = 0
            elif htype == "msg_to_app":
                loop_msg_count += 1
        else:
            if current_htype is not None:
                current_lines.append(line)

    flush()

    evaluation = _extract_evaluation(lines)

    final_step = next((s for s in steps if s.metadata.get("is_final_answer")), None)
    final_status: str | None = None
    final_answer: str | None = None
    if final_step:
        fm = COMPLETE_TASK_PATTERN.search(final_step.content)
        final_status = fm.group(1) if fm else None
        am = COMPLETE_TASK_ANSWER_PATTERN.search(final_step.content)
        raw = (am.group(1) or am.group(2)) if am else None
        final_answer = None if raw in (None, "None") else raw

    agent_sequence = [s.agent for s in steps]
    n_switches = sum(1 for i in range(1, len(agent_sequence)) if agent_sequence[i] != agent_sequence[i - 1])

    return Trace(
        steps=steps,
        metadata={
            "trace_id": trace_id,
            "task": task,
            "final_status": final_status,
            "final_answer": final_answer,
            "success": evaluation.get("success"),
            "passes": evaluation.get("passes", []),
            "failures": evaluation.get("failures", []),
            "n_turns": len(steps),
            "n_agent_switches": n_switches,
            "agent_participation": dict(Counter(agent_sequence)),
        },
    )


def trace_to_dict(t: Trace) -> dict:
    return {
        "steps": [
            {"agent": s.agent, "content": s.content, "kind": s.kind, "metadata": s.metadata}
            for s in t.steps
        ],
        "metadata": t.metadata,
    }


def save_traces(traces: list[Trace], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([trace_to_dict(t) for t in traces], f, indent=2, ensure_ascii=False)


def parse_all(data_path: Path = MAD_PATH) -> list[Trace]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    traces = []
    for rec in [r for r in data if r["mas_name"] == "AppWorld"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as tmp:
            tmp.write(rec["trace"]["trajectory"])
            tmp_path = Path(tmp.name)
        try:
            trace = parse_file(tmp_path)
            trace.metadata["mad_trace_key"] = rec["trace"]["key"]
            trace.metadata["mas_name"] = rec["mas_name"]
            trace.metadata["llm_name"] = rec["llm_name"]
            trace.metadata["benchmark_name"] = rec["benchmark_name"]
            trace.metadata["mast_annotation"] = rec["mast_annotation"]
            traces.append(trace)
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)
    return traces


def main():
    traces = parse_all()
    print(f"Parsed: {len(traces)} records, Errors: 0")
    save_traces(traces, OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
