import json
import re
from collections import Counter
from pathlib import Path

# --- Regex patterns ---

TRACE_START_MARKER = "SCENARIO.PY STARTING !#!#"

HEADER_PATTERN = re.compile(r"^---------- (.+?) ----------$")
NOISE_LINE_PATTERN = re.compile(r"^(SCENARIO\.PY|RUN\.SH) (COMPLETE|RUNTIME:|STARTING)")
RUNTIME_PATTERN = re.compile(r"SCENARIO\.PY RUNTIME:\s*(\d+)")

FINAL_ANSWER_PATTERN = re.compile(r"FINAL ANSWER:", re.IGNORECASE)
FINAL_ANSWER_TEXT_PATTERN = re.compile(r"FINAL ANSWER:\s*(.+)")

PLANNING_SECTION_PATTERN = re.compile(
    r"^\s*#{0,4}\s*(?:\*{1,2})?\s*(?:\d+\.\s*)?"
    r"(GIVEN OR VERIFIED FACTS|FACTS TO LOOK UP|FACTS TO DERIVE|EDUCATED GUESSES)"
    r"(?:\*{1,2})?\s*\n"
    r"(.*?)"
    r"(?=^\s*#{0,4}\s*(?:\*{1,2})?\s*\d+\.|Here is the plan|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
PLAN_BLOCK_PATTERN = re.compile(r"Here is the plan to follow.*?:\s*\n(.*?)$", re.DOTALL | re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)

URL_PATTERN = re.compile(r"https?://[^\s\)\"'<>]+")
FILE_ATTACHMENT_PATTERN = re.compile(r"\b\w+\.\w{2,4}\b.*current working directory", re.IGNORECASE)
PLANNING_SIGNAL_PATTERN = re.compile(
    r"GIVEN OR VERIFIED FACTS|FACTS TO LOOK UP|Here is the plan|Here is an initial fact sheet",
    re.IGNORECASE,
)

# --- Step-level helpers ---

def classify_step_kind(agent: str, content: str) -> str:
    if agent == "user":
        return "message"
    if agent == "MagenticOneOrchestrator":
        if FINAL_ANSWER_PATTERN.search(content) or PLANNING_SIGNAL_PATTERN.search(content):
            return "message"
        return "tool_call"  
    if agent == "Assistant":
        return "message"
    return "tool_result"


def build_step(agent: str, content: str, index: int) -> dict:
    return {
        "step_index": index,
        "agent": agent,
        "kind": classify_step_kind(agent, content),
        "content": content,
        "is_final_answer": bool(FINAL_ANSWER_PATTERN.search(content)),
        "urls": URL_PATTERN.findall(content),
    }

# --- Planning section helpers ---

def extract_bullets(text: str) -> list[str]:
    return [m.group(1).strip() for m in BULLET_PATTERN.finditer(text)]


def parse_planning_section(step: dict) -> dict | None:
    content = step["content"]
    given_facts: list[str] = []
    facts_to_look_up: list[str] = []
    educated_guesses: list[str] = []

    for match in PLANNING_SECTION_PATTERN.finditer(content):
        heading = match.group(1).upper()
        bullets = extract_bullets(match.group(2))
        if "GIVEN" in heading:
            given_facts = bullets
        elif "LOOK UP" in heading:
            facts_to_look_up = bullets
        elif "EDUCATED" in heading:
            educated_guesses = bullets

    plan_match = PLAN_BLOCK_PATTERN.search(content)
    plan = plan_match.group(1).strip() if plan_match else None

    if not any([given_facts, facts_to_look_up, educated_guesses, plan]):
        return None

    return {
        "step_index": step["step_index"],
        "given_facts": given_facts,
        "facts_to_look_up": facts_to_look_up,
        "educated_guesses": educated_guesses,
        "plan": plan,
    }

# --- Trace parsing ---

def extract_level_from_path(trace_dir: Path) -> int | None:
    for part in trace_dir.parts:
        match = re.search(r"level_(\d)", part)
        if match:
            return int(match.group(1))
    return None


def parse_trace(trace_dir: Path) -> dict:
    trace_dir = Path(trace_dir)
    run_dir = trace_dir / "0"

    console_log = (run_dir / "console_log.txt").read_text(encoding="utf-8", errors="replace")
    expected_answer = (run_dir / "expected_answer.txt").read_text(encoding="utf-8", errors="replace").strip()

    # Discard everything before the official trace-start marker.
    start_pos = console_log.find(TRACE_START_MARKER)
    trace_text = console_log[start_pos + len(TRACE_START_MARKER):] if start_pos != -1 else ""

    steps: list[dict] = []
    task: str | None = None
    final_answer: str | None = None
    runtime_seconds: int | None = None
    current_agent: str | None = None
    current_lines: list[str] = []

    def flush_current_block(agent: str, lines: list[str]) -> None:
        nonlocal task, final_answer
        content = "\n".join(lines).strip()
        if agent == "user" and task is None:
            task = content
        steps.append(build_step(agent, content, len(steps)))
        if not final_answer:
            match = FINAL_ANSWER_TEXT_PATTERN.search(content)
            if match:
                final_answer = match.group(1).strip()

    for line in trace_text.splitlines():
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            if current_agent is not None:
                flush_current_block(current_agent, current_lines)
            current_agent = header_match.group(1)
            current_lines = []
        else:
            runtime_match = RUNTIME_PATTERN.match(line)
            if runtime_match:
                runtime_seconds = int(runtime_match.group(1))
            if not NOISE_LINE_PATTERN.match(line):
                current_lines.append(line)

    if current_agent is not None:
        flush_current_block(current_agent, current_lines)

    for i, step in enumerate(steps):
        if step["agent"] == "MagenticOneOrchestrator" and step["kind"] == "tool_call":
            if i + 1 < len(steps) and steps[i + 1]["agent"] == "Assistant":
                step["kind"] = "message"

    # Derive summary statistics from parsed steps.
    orch_steps = [s for s in steps if s["agent"] == "MagenticOneOrchestrator"]
    planning_events = [p for s in orch_steps if (p := parse_planning_section(s)) is not None]
    n_replans = max(0, len(planning_events) - 1)

    agent_participation: dict[str, int] = dict(Counter(s["agent"] for s in steps))
    agent_sequence = [s["agent"] for s in steps]
    n_switches = sum(1 for i in range(1, len(agent_sequence)) if agent_sequence[i] != agent_sequence[i - 1])

    return {
        "trace_id": trace_dir.name,
        "level": extract_level_from_path(trace_dir),
        "task": task,
        "expected_answer": expected_answer,
        "final_answer": final_answer,
        "runtime_seconds": runtime_seconds,
        "planning_events": planning_events,
        "n_turns": len(steps),
        "n_agent_switches": n_switches,
        "n_replans": n_replans,
        "agent_participation": agent_participation,
        "task_requires_file": bool(FILE_ATTACHMENT_PATTERN.search(task or "")),
        "steps": steps,
    }

# --- Batch parsing and file I/O ---

def parse_all(magenticone_dir: Path) -> list[dict]:
    magenticone_dir = Path(magenticone_dir)
    traces: list[dict] = []
    for level_dir in sorted(magenticone_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        for trace_dir in sorted(level_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            traces.append(parse_trace(trace_dir))
    return traces


def save_traces(traces: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    here = Path(__file__).parent
    data_dir = "parsers\MagenticOne\MagenticOne_GAIA"
    output_file = here / "output_v2.json"

    print(f"Parsing traces from: {data_dir}")
    traces = parse_all(data_dir)
    save_traces(traces, output_file)
    print(f"Done — wrote {len(traces)} traces to {output_file}")
