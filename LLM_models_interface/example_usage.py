"""
Example: detect FM-1.3 (Step Repetition) using two synthetic traces.

Trace A — should be detected (agent repeats identical search call despite receiving a result).
Trace B — should NOT be detected (agent retries after a failed tool call, which is valid behaviour).
"""

from pydantic import BaseModel
from llm_interface import judge


SYSTEM_PROMPT = (
    "You are an expert evaluator of multi-agent system traces.\n\n"
    "Your task is to detect FM-1.3 (Step Repetition): a failure mode where an agent "
    "issues the SAME action or request multiple times without any new information, "
    "changed state, or meaningful variation between attempts.\n\n"
    "CRITICAL rule: if a step failed with an ERROR or timeout, a retry of that same step "
    "is VALID behaviour and must NOT be marked as FM-1.3. The error itself is new state.\n\n"
    "FM-1.3 requires ALL of the following to be true:\n"
    "1. The action is repeated verbatim or near-verbatim.\n"
    "2. The previous attempt returned a successful (non-error) result.\n"
    "3. Nothing changed between attempts (no new input, no error, no user feedback).\n\n"
    "Only mark detected=true when all three conditions are met."
)

# Trace A: agent repeats the same search call after already receiving results — clear FM-1.3
TRACE_A = """
Step 1 [Orchestrator]: search_web(query="IKEA annual revenue 2023")
Step 2 [WebSearchAgent]: Returned results: "IKEA reported €47.6B revenue in FY2023."
Step 3 [Orchestrator]: search_web(query="IKEA annual revenue 2023")
Step 4 [WebSearchAgent]: Returned results: "IKEA reported €47.6B revenue in FY2023."
Step 5 [Orchestrator]: search_web(query="IKEA annual revenue 2023")
Step 6 [WebSearchAgent]: Returned results: "IKEA reported €47.6B revenue in FY2023."
"""

# Trace B: agent retries after a tool error — this is valid, NOT FM-1.3
TRACE_B = """
Step 1 [Orchestrator]: search_web(query="IKEA annual revenue 2023")
Step 2 [WebSearchAgent]: ERROR — connection timeout, no results returned.
Step 3 [Orchestrator]: search_web(query="IKEA annual revenue 2023")
Step 4 [WebSearchAgent]: Returned results: "IKEA reported €47.6B revenue in FY2023."
Step 5 [Orchestrator]: summarise(text="IKEA reported €47.6B revenue in FY2023.")
"""


class FailureDetection(BaseModel):
    failure_mode: str   # e.g. "FM-1.3"
    detected: bool
    evidence: str       # Short quote or reasoning from the trace


for label, trace in [("A (should detect)", TRACE_A), ("B (should NOT detect)", TRACE_B)]:
    response = judge(
        model="qwen2.5:7b",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Does this trace contain FM-1.3 (step repetition)?\n\n{trace}",
        schema=FailureDetection,
    )
    print(f"\n--- Trace {label} ---")
    print(f"Detected : {response.parsed.detected}")
    print(f"Evidence : {response.parsed.evidence}")
    print(f"Cost     : ${response.cost_usd:.6f}  |  Latency: {response.latency_s:.2f}s")
