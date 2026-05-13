"""Parse Magentic (MagenticOne) traces from MAD_full_dataset.json.

MAD stores traces as raw console-log text (same format as the github parser but as a
string, not a file directory). This module contains all needed helpers inline.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

ROOT = Path(__file__).parent.parent.parent
MAD_PATH = ROOT / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "magenticone_output_mad.json"

# --- Patterns ---

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
WHAT_WENT_WRONG_PATTERN = re.compile(
    r"###?\s*What Went Wrong\s*\n(.*?)(?=###?\s*New Plan|\Z)",
    re.DOTALL | re.IGNORECASE,
)
NEW_PLAN_PATTERN = re.compile(r"###?\s*New Plan\s*\n(.*?)$", re.DOTALL | re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://[^\s\)\"'<>\\]+")
FILE_ATTACHMENT_PATTERN = re.compile(r"\b\w+\.\w{2,4}\b.*current working directory", re.IGNORECASE)
PLANNING_SIGNAL_PATTERN = re.compile(
    r"GIVEN OR VERIFIED FACTS|FACTS TO LOOK UP|Here is the plan|Here is an initial fact sheet",
    re.IGNORECASE,
)


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


def build_step(agent: str, content: str, index: int) -> Step:
    content = content.replace("\n<image>", "").rstrip()
    return Step(
        agent=agent,
        content=content,
        kind=classify_step_kind(agent, content),
        metadata={
            "step_index": index,
            "is_final_answer": bool(FINAL_ANSWER_PATTERN.search(content)),
            "urls": list(dict.fromkeys(URL_PATTERN.findall(content))),
        },
    )


def bullets_to_list(text: str) -> list[str]:
    return [m.group(1).strip() for m in BULLET_PATTERN.finditer(text)]


def parse_planning_section(step: Step, step_index: int) -> dict | None:
    content = step.content
    given_facts: list[str] = []
    facts_to_look_up: list[str] = []
    educated_guesses: list[str] = []

    for match in PLANNING_SECTION_PATTERN.finditer(content):
        heading = match.group(1).upper()
        bullets = bullets_to_list(match.group(2))
        if "GIVEN" in heading:
            given_facts = bullets
        elif "LOOK UP" in heading:
            facts_to_look_up = bullets
        elif "EDUCATED" in heading:
            educated_guesses = bullets

    plan_match = PLAN_BLOCK_PATTERN.search(content)
    plan_block = plan_match.group(1).strip() if plan_match else None

    what_went_wrong: str | None = None
    plan: str | None = plan_block
    if plan_block:
        wwm = WHAT_WENT_WRONG_PATTERN.search(plan_block)
        npm = NEW_PLAN_PATTERN.search(plan_block)
        if wwm:
            what_went_wrong = wwm.group(1).strip()
            plan = npm.group(1).strip() if npm else None

    if not any([given_facts, facts_to_look_up, educated_guesses, plan_block]):
        return None

    return {
        "step_index": step_index,
        "given_facts": given_facts,
        "facts_to_look_up": facts_to_look_up,
        "educated_guesses": educated_guesses,
        "plan": plan,
        "what_went_wrong": what_went_wrong,
    }


def trace_to_dict(t: Trace) -> dict:
    return {
        "steps": [
            {"agent": s.agent, "content": s.content, "kind": s.kind, "metadata": s.metadata}
            for s in t.steps
        ],
        "metadata": t.metadata,
    }


def parse_trajectory_text(console_log: str, record: dict) -> Trace | None:
    """Parse a single MAD console-log string into a Trace."""
    start_pos = console_log.find(TRACE_START_MARKER)
    trace_text = (
        console_log[start_pos + len(TRACE_START_MARKER):]
        if start_pos != -1
        else console_log
    )

    steps: list[Step] = []
    task: str | None = None
    final_answer: str | None = None
    runtime_seconds: int | None = None
    current_agent: str | None = None
    current_lines: list[str] = []

    def flush_block(agent: str, lines: list[str]) -> None:
        nonlocal task, final_answer
        content = "\n".join(lines).strip()
        if agent == "user" and task is None:
            task = content
        steps.append(build_step(agent, content, len(steps)))
        if not final_answer:
            m = FINAL_ANSWER_TEXT_PATTERN.search(content)
            if m:
                final_answer = m.group(1).strip()

    for line in trace_text.splitlines():
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            if current_agent is not None:
                flush_block(current_agent, current_lines)
            current_agent = header_match.group(1)
            current_lines = []
        else:
            runtime_match = RUNTIME_PATTERN.match(line)
            if runtime_match:
                runtime_seconds = int(runtime_match.group(1))
            if not NOISE_LINE_PATTERN.match(line):
                current_lines.append(line)

    if current_agent is not None:
        flush_block(current_agent, current_lines)

    if not steps:
        return None

    for i, step in enumerate(steps):
        if step.agent == "MagenticOneOrchestrator" and step.kind == "tool_call":
            if i + 1 < len(steps) and steps[i + 1].agent == "Assistant":
                step.kind = "message"

    planning_events = [
        p
        for i, s in enumerate(steps)
        if s.agent == "MagenticOneOrchestrator"
        if (p := parse_planning_section(s, i)) is not None
    ]

    agent_sequence = [s.agent for s in steps]
    agent_participation = dict(Counter(agent_sequence))
    n_switches = sum(
        1
        for i in range(1, len(agent_sequence))
        if agent_sequence[i] != agent_sequence[i - 1]
    )

    return Trace(
        steps=steps,
        metadata={
            "trace_id": record["trace_id"],
            "mad_trace_key": record["trace"]["key"],
            "mas_name": record["mas_name"],
            "llm_name": record["llm_name"],
            "benchmark_name": record["benchmark_name"],
            "mast_annotation": record["mast_annotation"],
            "task": task,
            "final_answer": final_answer,
            "runtime_seconds": runtime_seconds,
            "planning_events": planning_events,
            "n_turns": len(steps),
            "n_agent_switches": n_switches,
            "n_replans": max(0, len(planning_events) - 1),
            "agent_participation": agent_participation,
        },
    )


def parse_all(data_path: Path = MAD_PATH) -> list[Trace]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    records = [r for r in data if r["mas_name"] == "Magentic"]
    traces = []
    errors = []

    for rec in records:
        try:
            trace = parse_trajectory_text(rec["trace"]["trajectory"], rec)
            if trace is None:
                errors.append((rec["trace"]["key"], "no steps extracted"))
                continue
            traces.append(trace)
        except Exception as e:
            errors.append((rec["trace"]["key"], str(e)))

    if errors:
        print(f"Errors ({len(errors)}):")
        for key, err in errors:
            print(f"  {key}: {err}")

    return traces


def main():
    traces = parse_all()
    print(f"Parsed: {len(traces)} records, Errors: 0")
    output = [trace_to_dict(t) for t in traces]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
