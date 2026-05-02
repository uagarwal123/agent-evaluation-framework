import re
from pathlib import Path


def parse_chatdev_log(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    messages = []
    current_agent = None
    current_lines = []

    speaker_pattern = re.compile(r"\[.*? INFO\]\s*(.*?):\s*(.*)")

    # agents we want to ignore
    excluded_agents = {"System", "HTTP Request"}

    def flush():
        nonlocal current_agent, current_lines
        content = "\n".join(current_lines).strip()

        if current_agent and content and current_agent not in excluded_agents:
            messages.append({
                "agent": current_agent,
                "content": content
            })

    for line in lines:
        match = speaker_pattern.match(line)

        if match:
            flush()

            agent = match.group(1).strip()
            first_content = match.group(2).strip()

            current_agent = agent
            current_lines = []

            if first_content:
                current_lines.append(first_content)

        else:
            if current_agent:
                current_lines.append(line)

    flush()
    return messages


def extract_task_from_log(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    # Try different patterns for task
    match = re.search(r"\*\*task_prompt\*\*:\s*(.*?)\n", text)
    if match:
        return match.group(1).strip()

    match = re.search(r"\*\*task_prompt\*\*\s*\|\s*(.*?)\s*\|", text)
    if match:
        return match.group(1).strip()

    return None


def parse_chatdev_dataset(folder_path):
    folder_path = Path(folder_path)

    dataset = []

    for project in sorted(folder_path.iterdir()):
        if not project.is_dir():
            continue

        log_files = list(project.glob("*.log"))
        if not log_files:
            continue

        log_file = log_files[0]

        dataset.append({
            "trace_id": project.name,
            "level": None,
            "task": extract_task_from_log(log_file),
            "expected_answer": None,
            "final_answer": None,
            "runtime_seconds": None,
            "messages": parse_chatdev_log(log_file)
        })

    return dataset


# 🔍 Test block
if __name__ == "__main__":
    folder = "/content/MAST/traces/programdev/chatdev"

    data = parse_chatdev_dataset(folder)

    print("Parsed:", len(data))

    if data:
        print("\n=== Example Trace ===")
        example = data[0]

        print("\nTrace ID:")
        print(example["trace_id"])

        print("\nTask:")
        print(example["task"])

        print("\nFirst 3 messages:")
        for msg in example["messages"][:3]:
            print(msg)