import json
from pathlib import Path


def content_to_text(content):
    if isinstance(content, list):
        return "\n".join(str(x) for x in content)
    if content is None:
        return ""
    return str(content)


# Parse ONE AG2 file
def parse_ag2_trace(file_path):
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    messages = []

    for step in raw.get("trajectory", []):
        messages.append({
            "agent": str(step.get("name") or step.get("role") or ""),
            "content": content_to_text(step.get("content"))
        })

    return {
        "trace_id": str(raw.get("instance_id")),
        "level": None,
        "task": content_to_text(raw.get("problem_statement")),
        "expected_answer": raw.get("other_data", {}).get("answer"),
        "final_answer": raw.get("other_data", {}).get("given"),
        "runtime_seconds": None,
        "messages": messages
    }


# Parse ALL AG2 files
def parse_ag2_dataset(folder_path):
    folder_path = Path(folder_path)

    dataset = []

    for file_path in folder_path.glob("*.json"):
        dataset.append(parse_ag2_trace(file_path))

    return dataset


# Quick test
if __name__ == "__main__":
    folder = "data/raw/MAST/traces/AG2"

    data = parse_ag2_dataset(folder)

    print("Parsed traces:", len(data))

    if data:
        print("\nExample:")
        print(data[0]["trace_id"])
        print(data[0]["task"][:100])
        print(data[0]["messages"][:2])