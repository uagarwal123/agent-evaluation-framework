from pathlib import Path
import random


# ---------- Load raw trace ----------
def load_trace(file_path):
    return Path(file_path).read_text(encoding="utf-8")


# ---------- Print first N lines ----------
def show_head(file_path, n=50):
    print(f"\n=== FIRST {n} LINES ===\n")
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines[:n], start=1):
        print(f"{i:03}: {line}")


# ---------- Find important keywords ----------
def find_keywords(file_path, keywords):
    print("\n=== KEYWORD SEARCH ===\n")
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                print(f"[Line {i}] {line}")


# ---------- Show only agent transitions ----------
def show_agent_structure(file_path):
    print("\n=== AGENT STRUCTURE ===\n")
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()

        if line.startswith("Response from"):
            print(f"AGENT RESPONSE → {line}")

        elif line.startswith("Message to"):
            print(f"AGENT MESSAGE  → {line}")


# ---------- Guess task ----------
def extract_task_guess(file_path):
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    print("\n=== TASK CANDIDATE ===\n")

    for line in lines[:20]:
        line = line.strip()
        if len(line) > 20:
            print(line)
            break


# ---------- Random trace ----------
def get_random_trace(folder_path):
    files = list(Path(folder_path).glob("*.txt"))
    return random.choice(files)


# ---------- MAIN ----------
if __name__ == "__main__":
    folder = "data/raw/MAST/traces/AppWorld"

    # pick one trace (random or specific)
    file = get_random_trace(folder)
    print("FILE:", file.name)

    # Explore it
    show_head(file, 40)

    find_keywords(file, [
        "final",
        "answer",
        "task",
        "result",
        "done",
        "completed"
    ])

    show_agent_structure(file)

    extract_task_guess(file)