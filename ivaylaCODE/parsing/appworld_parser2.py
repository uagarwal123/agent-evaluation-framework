from pathlib import Path
import json

# Single-trace parser
def parse_appworld_trace(file_path: str | Path) -> list[dict]:
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    messages: list[dict] = []
    current_agent: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_agent is not None and current_lines:
            messages.append({
                "agent":   current_agent,
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


# Dataset-level parser
def parse_appworld_dataset(folder_path: str | Path) -> list[dict]:
    folder_path = Path(folder_path)
    dataset = []
    for file_path in sorted(folder_path.glob("*.txt")):
        dataset.append({
            "trace_id":         file_path.stem,
            "messages":         parse_appworld_trace(file_path),
            "MAST_annotations": {},
        })

    return dataset



# Sanity-check / SMOKE TEST  (I CAN RUN IT WITH:  python parse_appworld.py)
def _smoke_test_single(file_path: str | Path) -> None:
    print("=== Single-trace smoke test ===")
    messages = parse_appworld_trace(file_path)
    print(f"Messages parsed : {len(messages)}")
    for msg in messages[:5]:
        print(f"  agent   : {msg['agent']}")
        print(f"  content : {msg['content'][:120]!r}")
        print()


def _smoke_test_dataset(folder_path: str | Path) -> None:
    print("=== Dataset smoke test ===")
    dataset = parse_appworld_dataset(folder_path)
    print(f"Traces parsed : {len(dataset)}")

    if not dataset:
        print("  (no .txt files found – check the folder path)")
        return

    first = dataset[0]

    print("First trace (top-level keys):")
    print(f"  trace_id         : {first['trace_id']!r}")
    print(f"  messages         : list of {len(first['messages'])} dicts")
    print(f"  MAST_annotations : {first['MAST_annotations']}")
    print()
    print("First 2 messages (real dict):")
    for msg in first["messages"][:2]:
        print(f"  {msg}")
    print()

    # Scheme validation
    errors = []
    for trace in dataset:
        assert isinstance(trace["trace_id"],         str),  "trace_id must be str"
        assert isinstance(trace["messages"],          list), "messages must be list"
        assert isinstance(trace["MAST_annotations"],  dict), "MAST_annotations must be dict"
        for i, msg in enumerate(trace["messages"]):
            for key in ("agent", "content"):
                if key not in msg:
                    errors.append(f"{trace['trace_id']} msg[{i}] missing '{key}'")
                elif not isinstance(msg[key], str):
                    errors.append(f"{trace['trace_id']} msg[{i}]['{key}'] not str")

    if errors:
        print("Scheme errors found:")
        for e in errors:
            print(" ", e)
    else:
        print("All traces passed the validation")


if __name__ == "__main__":
    SINGLE_FILE  = "data/raw/MAST/traces/AppWorld/229360a_1.txt"
    FOLDER_PATH  = "data/raw/MAST/traces/AppWorld"

    _smoke_test_single(SINGLE_FILE)
    _smoke_test_dataset(FOLDER_PATH)
