import logging
import re
from datetime import date
from typing import Literal
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

_CURRENT_STATUS_TERMS = ("latest", "current", "right now", "today", "currently", "ongoing", "active", "status")
_HISTORICAL_TERMS = ("historical", "history", "historically", "when did", "last year", "previous", "past outage", "postmortem", "what happened")
_EVIDENCE_EVENT_TERMS = ("outage", "incident", "vulnerability", "security disclosure", "cve", "postmortem", "service degradation")
_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
    flags=re.IGNORECASE,
)


def is_current_status_question(question: str) -> bool:
    normalized = question.casefold()
    return any(term in normalized for term in _CURRENT_STATUS_TERMS) and not any(
        term in normalized for term in _HISTORICAL_TERMS
    )


def requires_evidence_backed_retrieval(question: str) -> bool:
    """Return True for factual event questions that must not use direct generation."""
    normalized = question.casefold()
    has_event = any(term in normalized for term in _EVIDENCE_EVENT_TERMS)
    has_date = bool(_DATE_PATTERN.search(question)) or any(term in normalized for term in _HISTORICAL_TERMS)
    return has_event and has_date


def build_web_query(question: str, today: date | None = None) -> str:
    """Add freshness and authority hints only for current-status requests."""
    if not is_current_status_question(question):
        return question
    as_of = (today or date.today()).isoformat()
    return (
        f"{question} current incident status as of {as_of}; prioritize official "
        "Microsoft Service Health and Microsoft sources; include publication date and incident date"
    )


def _first_value(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _source_metadata(item: dict) -> dict:
    for key in ("metadata", "source_metadata", "meta"):
        metadata = item.get(key)
        if isinstance(metadata, dict):
            return metadata
    return {}


def _infer_incident_date(content: str) -> str:
    match = re.search(
        r"(?:incident|outage|disruption|service degradation)\s+"
        r"(?:date\s*[:\-]?\s*|on\s+|started(?:\s+on)?\s+|began(?:\s+on)?\s+)"
        r"(\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        content,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _web_source(item: dict) -> dict:
    title = _first_value(item, "title")
    url = _first_value(item, "url")
    content = _first_value(item, "content", "raw_content")
    metadata = _source_metadata(item)
    published_date = _first_value(item, "published_date", "published") or _first_value(
        metadata, "published_date", "published", "date"
    )
    incident_date = _first_value(item, "incident_date") or _first_value(metadata, "incident_date")
    if not incident_date:
        incident_date = _infer_incident_date(f"{title}\n{content}")
    domain = urlparse(url).netloc.casefold()
    return {
        "title": title or url,
        "url": url,
        "content": content,
        "published_date": published_date,
        "incident_date": incident_date,
        "status": _first_value(item, "current_status", "status"),
        "authoritative": domain.endswith(("microsoft.com", "microsoftonline.com", "office.com")),
    }


def _status_basis(sources: list[dict]) -> str:
    if any(source.get("authoritative") and source.get("status") for source in sources):
        return "official"
    if any(source.get("status") for source in sources):
        return "third_party"
    return "unverified"

def llm():
    global _llm
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    route = "kb" if requires_evidence_backed_retrieval(state["question"]) else decision.route
    return {"source_used": route, "trace": add_trace(state, f"Router → {route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    current_status_question = is_current_status_question(state["question"])
    result = web_search_tool().invoke({"query": build_web_query(state["current_query"])})
    lines, citations = [], []
    sources = []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            source = _web_source(item)
            sources.append(source)
            date_lines = []
            if source["published_date"]:
                date_lines.append(f"Publication date: {source['published_date']}")
            if source["incident_date"]:
                date_lines.append(f"Incident date: {source['incident_date']}")
            if source["status"]:
                date_lines.append(f"Reported status: {source['status']}")
            lines.append(
                f"Title: {source['title']}\nURL: {source['url']}\n"
                f"{chr(10).join(date_lines)}\nContent: {source['content']}"
            )
            citations.append({
                "title": source["title"],
                "url": source["url"],
                "type": "web",
                "published_date": source["published_date"],
                "incident_date": source["incident_date"],
            })
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "web_sources": sources,
        "web_current_status_verified": False,
        "web_is_current_status_question": current_status_question,
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    today = date.today().isoformat()
    grade = grader.invoke(f"""
You grade web evidence for an enterprise IT support assistant as of {today}.
Question: {state['question']}
Is this a current-status question? {state['web_is_current_status_question']}
Web evidence:\n{state['web_results']}
Distinguish each source's publication date from its incident date. A publication date
does not prove that an incident is active today. For a current-status question, set
current_status_verified=true only when authoritative evidence explicitly reports the
present status, such as official Microsoft Service Health/status information. An older
article can be good historical context but must not verify a current incident. If
current status cannot be established, set current_status_verified=false. Return good
when the evidence can answer with an appropriate qualification; otherwise weak.
Return valid JSON with grade, current_status_verified, and has_relevant_dates.
""")
    basis = _status_basis(state["web_sources"])
    verified = bool(grade.current_status_verified and basis == "official" and state["web_is_current_status_question"])
    return {
        "web_grade": grade.grade,
        "web_current_status_verified": verified,
        "web_current_status_basis": basis,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()} "
            f"(current status {'verified' if verified else basis})",
        ),
    }

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""").content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}\n\nPrivate KB:\n{context}
""").content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            safe_name = re.split(r"[\\/]", str(src).rstrip("\\/"))[-1] or "Private KB"
            citations.append({"title": safe_name, "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    status_instruction = {
        "official": "Current status is confirmed only by an authoritative source; state it with the source and date.",
        "third_party": "Only third-party reporting is available; do not present it as officially confirmed current status.",
        "unverified": "Current status is unverified; do not describe a historical article as an active incident.",
    }[state["web_current_status_basis"]]
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}
{status_instruction}
For every useful source, distinguish its publication date from its incident date and include those dates when available. Preserve older articles as historical context when the user asks a historical question.
\nWeb evidence:\n{state['web_results']}
""").content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "web_sources": [],
        "web_current_status_verified": False,
        "web_current_status_basis": "unverified",
        "web_is_current_status_question": False,
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)
