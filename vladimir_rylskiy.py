from huggingface_hub import hf_hub_download
import json
import re

def parse_hyperagent_trace(trace):
    messages = []

    # extract trajectory if trace is dict
    if isinstance(trace, dict):
        trace = trace.get("trajectory", "")

    if not isinstance(trace, str):
        return messages

    pattern = r"INFO - ([^:\n]+):"
    parts = re.split(pattern, trace)

    for i in range(1, len(parts), 2):
        agent_raw = parts[i].strip()
        content = parts[i + 1].strip()

        agent = agent_raw
        agent = agent.replace("'s Response", "")
        agent = agent.replace("'s Thought", "")
        agent = agent.replace("'s Action", "")
        agent = agent.replace("Response", "")
        agent = agent.replace("Thought", "")
        agent = agent.replace("Action", "")
        agent = agent.strip()

        if len(content) < 5:
            continue

        messages.append({
            "agent": agent,
            "content": content
        })

    return messages

def main():
    path = hf_hub_download(
        repo_id="mcemri/MAST-Data",
        filename="MAD_full_dataset.json",
        repo_type="dataset"
    )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hyperagent_traces = []

    for item in data:
        if item["mas_name"].lower() == "hyperagent":
            messages = parse_hyperagent_trace(item["trace"])

            if messages:
                hyperagent_traces.append(messages)

    print("Number of cleaned HyperAgent traces:", len(hyperagent_traces))
    print(json.dumps(hyperagent_traces[0][:3], indent=2, ensure_ascii=False))

    with open("hyperagent_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(hyperagent_traces, f, indent=2, ensure_ascii=False)

    print("Saved to hyperagent_cleaned.json")


if __name__ == "__main__":
    main()