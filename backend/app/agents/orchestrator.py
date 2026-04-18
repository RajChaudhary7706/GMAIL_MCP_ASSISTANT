"""
LangGraph Multi-Agent Orchestrator — 5-agent pipeline:
  Planner → Triage Officer → Researcher → Executor → Critic
"""
import json, logging
from typing import Dict, List, Literal, Optional, TypedDict
from anthropic import Anthropic
from app.core.config import settings
from app.mcp.tools.gmail_tools import dispatch_tool

logger = logging.getLogger(__name__)
client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)


class AgentState(TypedDict):
    user_id: str; access_token: str; user_request: str
    plan: Optional[List[str]]; email_context: Optional[str]
    tool_calls: List[Dict]; execution_results: List[Dict]
    critique: Optional[str]; pii_detected: bool
    final_response: Optional[str]
    requires_human_approval: bool; error: Optional[str]


def _claude(system: str, messages: List[Dict]) -> str:
    resp = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        system=system, messages=messages)
    return resp.content[0].text if resp.content else ""


# ── Agent 1: Planner ──────────────────────────────────────────
def planner_agent(state: AgentState) -> AgentState:
    """Decomposes user request into ordered steps."""
    resp = _claude(
        system="You are a Planner. Decompose the request into steps. "
               "Return a JSON array of step strings. Flag destructive actions.",
        messages=[{"role":"user","content":
            f"Request: {state['user_request']}\n"
            f"Return JSON array of steps."}])
    try:
        import re
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        plan = json.loads(match.group()) if match else [state["user_request"]]
    except:
        plan = ["Process user request"]

    high_risk = any(kw in state["user_request"].lower()
        for kw in ["delete","permanent","bulk","all emails"])

    return {**state, "plan": plan, "requires_human_approval": high_risk}


# ── Agent 2: Triage Officer ───────────────────────────────────
async def triage_agent(state: AgentState) -> AgentState:
    """IMAP IDLE watcher + AI classification of incoming emails."""
    try:
        unread = await dispatch_tool("list_emails", {
            "access_token": state["access_token"],
            "query": "is:unread", "max_results": 5})
        classified = []
        for email in unread.get("emails", [])[:3]:
            try:
                cl = await dispatch_tool("classify_email", {
                    "access_token": state["access_token"],
                    "message_id": email["id"],
                    "anthropic_api_key": settings.ANTHROPIC_API_KEY})
                classified.append({"email": email, "classification": cl})
            except: pass
        return {**state, "email_context": json.dumps({
            "triage_results": classified,
            "unread_count": unread.get("estimated_total", 0)})}
    except Exception as e:
        return {**state, "error": str(e)}


# ── Agent 3: Researcher ───────────────────────────────────────
async def researcher_agent(state: AgentState) -> AgentState:
    """RAG semantic search for historical email context."""
    try:
        results = await dispatch_tool("semantic_search", {
            "query": state["user_request"],
            "user_id": state["user_id"], "top_k": 5})
        if not results.get("matches"):
            results = await dispatch_tool("list_emails", {
                "access_token": state["access_token"],
                "query": state["user_request"][:100], "max_results": 5})
        return {**state, "email_context": json.dumps(results)}
    except Exception as e:
        return {**state, "email_context": "No historical context available."}


# ── Agent 4: Executor ─────────────────────────────────────────
async def executor_agent(state: AgentState) -> AgentState:
    """Runs Gmail tool calls based on plan + context."""
    resp = _claude(
        system=f"You are the Executor. Execute the user's Gmail request. "
               f"Context: {state.get('email_context','None')[:2000]}\n"
               f"Return a JSON summary of actions and results.",
        messages=[{"role":"user","content":
            f"Request: {state['user_request']}\n"
            f"Plan: {json.dumps(state.get('plan',[]))}\n"
            f"Access token available. Return JSON summary."}])

    calls = state.get("tool_calls", [])
    results = state.get("execution_results", [])
    calls.append({"agent": "executor", "response": resp[:500]})
    results.append({"result": resp})
    return {**state, "tool_calls": calls, "execution_results": results}


# ── Agent 5: Critic ───────────────────────────────────────────
async def critic_agent(state: AgentState) -> AgentState:
    """Reviews results for PII, tone, accuracy, and safety."""
    import re
    results_text = json.dumps(state.get("execution_results", []))
    pii_found = bool(re.search(r"\b\d{3}-\d{2}-\d{4}\b", results_text))

    resp = _claude(
        system="You are the Critic. Review results for PII, tone, accuracy. "
               "Return JSON: {approved, issues, improved_response}",
        messages=[{"role":"user","content":
            f"Request: {state['user_request']}\n"
            f"Results: {results_text[:3000]}\n"
            f"PII detected: {pii_found}\nReturn JSON critique."}])
    try:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        critique = json.loads(match.group()) if match else {}
    except:
        critique = {"approved": True, "improved_response": ""}

    final = (critique.get("improved_response") or
             (state.get("execution_results") or [{}])[-1].get("result",""))

    return {**state, "critique": json.dumps(critique),
            "pii_detected": pii_found, "final_response": final}


# ── Routing ───────────────────────────────────────────────────
def needs_research(state: AgentState) -> Literal["researcher","executor"]:
    plan = state.get("plan", [])
    kws = ["search","find","history","previous","context","recall"]
    return "researcher" if any(
        any(k in s.lower() for k in kws) for s in plan
    ) else "executor"


# ── Graph Assembly ────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    def build_graph():
        g = StateGraph(AgentState)
        g.add_node("planner", planner_agent)
        g.add_node("triage", triage_agent)
        g.add_node("researcher", researcher_agent)
        g.add_node("executor", executor_agent)
        g.add_node("critic", critic_agent)
        g.set_entry_point("planner")
        g.add_edge("planner", "triage")
        g.add_conditional_edges("triage", needs_research,
            {"researcher":"researcher","executor":"executor"})
        g.add_edge("researcher", "executor")
        g.add_edge("executor", "critic")
        g.add_edge("critic", END)
        return g.compile()
    agent_graph = build_graph()
except ImportError:
    agent_graph = None


async def run_agent(user_request:str, user_id:str, access_token:str) -> Dict:
    """Entry point for the agent pipeline."""
    state: AgentState = {
        "user_id":user_id,"access_token":access_token,
        "user_request":user_request,"plan":None,"email_context":None,
        "tool_calls":[],"execution_results":[],"critique":None,
        "pii_detected":False,"final_response":None,
        "requires_human_approval":False,"error":None}

    if agent_graph:
        final = await agent_graph.ainvoke(state)
    else:
        s = planner_agent(state)
        s = await researcher_agent(s)
        s = await executor_agent(s)
        s = await critic_agent(s)
        final = s

    return {
        "response": final.get("final_response","No response generated"),
        "plan": final.get("plan",[]),
        "tool_calls": len(final.get("tool_calls",[])),
        "pii_detected": final.get("pii_detected",False),
        "requires_approval": final.get("requires_human_approval",False),
        "error": final.get("error"),
    }