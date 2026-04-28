from pathlib import Path
import json


# Extract task from the top of the trace
def extract_task(lines: list[str]) -> str | None:
    for line in lines:
        line = line.strip()

        if line.startswith("Task "):
            continue

        if line and not line.startswith("*"):
            return line

    return None


# Extract completion status if present
def extract_status(lines: list[str]) -> str | None:
    for line in lines:
        line = line.strip().lower()

        if "complete_task" in line:
            if "status=\"success\"" in line or "status='success'" in line:
                return "success"
            if "status=\"fail\"" in line or "status='fail'" in line:
                return "fail"

    return None


# Single-trace parser
def parse_appworld_trace(file_path: str | Path) -> list[dict]:
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    messages: list[dict] = []
    current_agent: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_agent is not None and current_lines:
            messages.append({
                "agent": current_agent,
                "content": " ".join(current_lines).strip(),
            })

    for raw in lines:
        line = raw.strip()

        if line.startswith("Response from "):
            flush()
            current_agent = line.removeprefix("Response from ").strip()
            current_lines = []

        elif line.startswith("Message to "):
            flush()
            target = line.removeprefix("Message to ").strip()
            current_agent = f"user→{target}"
            current_lines = []

        elif line.startswith("Code Execution Output"):
            continue

        elif line.startswith("{") or line.startswith("["):
            continue

        elif line.startswith("}") or line.startswith("]"):
            continue

        elif "print(" in line or "api_docs" in line:
            continue

        elif line == "":
            continue

        else:
            if current_agent is not None:
                current_lines.append(line)

    flush()
    return messages


# Single structured trace
def parse_appworld_single(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    lines = file_path.read_text(encoding="utf-8").splitlines()

    return {
        "trace_id": file_path.stem,
        "task": extract_task(lines),
        "status": extract_status(lines),
        "expected_answer": None,
        "final_answer": None,
        "level": None,
        "runtime_seconds": None,
        "messages": parse_appworld_trace(file_path),
        "MAST_annotations": {},
        "metadata": {},
    }


# Dataset-level parser
def parse_appworld_dataset(folder_path: str | Path) -> list[dict]:
    folder_path = Path(folder_path)

    dataset = []

    for file_path in sorted(folder_path.glob("*.txt")):
        dataset.append(parse_appworld_single(file_path))

    return dataset