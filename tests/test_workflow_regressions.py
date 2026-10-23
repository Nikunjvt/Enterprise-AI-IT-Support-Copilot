import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.rag import workflow
from app.rag.state import EvidenceGrade, RouteDecision


class _Router:
    def __init__(self, route):
        self.route = route

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _prompt):
        return RouteDecision(route=self.route)


class _WebSearch:
    def __init__(self, result):
        self.result = result

    def invoke(self, _payload):
        return self.result


class _Grader:
    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _prompt):
        return EvidenceGrade(grade="good", current_status_verified=True, has_relevant_dates=True)


class _Answerer:
    def invoke(self, _prompt):
        return SimpleNamespace(content="Evidence-backed answer")


def _state(question="What is the latest Microsoft Teams outage guidance?"):
    return {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "web_sources": [],
        "web_current_status_verified": False,
        "web_current_status_basis": "unverified",
        "web_is_current_status_question": workflow.is_current_status_question(question),
        "kb_grade": "weak",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }


class WorkflowRegressionTests(unittest.TestCase):
    def test_hello_stays_direct_and_vpn_stays_kb(self):
        with patch.object(workflow, "llm", return_value=_Router("direct")):
            hello = workflow.route_question(_state("Hello!"))
        self.assertEqual(hello["source_used"], "direct")

        with patch.object(workflow, "llm", return_value=_Router("kb")):
            vpn = workflow.route_question(_state("How do I connect to the company VPN from home?"))
        self.assertEqual(vpn["source_used"], "kb")

    def test_dated_historical_outage_cannot_route_direct(self):
        question = "What happened during the Microsoft Teams outage on January 22, 2026?"
        with patch.object(workflow, "llm", return_value=_Router("direct")):
            result = workflow.route_question(_state(question))
        self.assertEqual(result["source_used"], "kb")
        self.assertTrue(workflow.requires_evidence_backed_retrieval(question))

    def test_source_dates_are_conservative_and_preserved_through_pipeline(self):
        explicit = {
            "title": "Teams incident report",
            "url": "https://status.example.test/teams",
            "content": "The outage began on January 22, 2026.",
            "metadata": {"published_date": "2026-01-23"},
        }
        source = workflow._web_source(explicit)
        self.assertEqual(source["published_date"], "2026-01-23")
        self.assertEqual(source["incident_date"], "January 22, 2026")

        unrelated = {
            "title": "Teams outage report",
            "url": "https://status.example.test/teams",
            "content": "Page updated on January 22, 2026.",
            "date": "2026-01-22",
        }
        unrelated_source = workflow._web_source(unrelated)
        self.assertEqual(unrelated_source["published_date"], "")
        self.assertEqual(unrelated_source["incident_date"], "")

        with patch.object(workflow, "web_search_tool", return_value=_WebSearch({"results": [explicit]})):
            state = _state()
            searched = workflow.search_web(state)
        self.assertEqual(searched["citations"][0]["published_date"], "2026-01-23")
        self.assertEqual(searched["citations"][0]["incident_date"], "January 22, 2026")

        graded_state = {**state, **searched}
        with patch.object(workflow, "llm", return_value=_Grader()):
            graded = workflow.grade_web(graded_state)
        self.assertEqual(graded["web_current_status_basis"], "unverified")
        self.assertFalse(graded["web_current_status_verified"])

        with patch.object(workflow, "llm", return_value=_Answerer()):
            generated = workflow.generate_from_web({**graded_state, **graded})
        complete_state = {**graded_state, **graded, **generated}
        self.assertEqual(complete_state["citations"][0]["published_date"], "2026-01-23")
        self.assertEqual(complete_state["citations"][0]["incident_date"], "January 22, 2026")

    def test_current_status_requires_authoritative_explicit_status(self):
        official_item = {
            "title": "Microsoft incident",
            "url": "https://status.microsoft.com/incident/123",
            "content": "Service is investigating the incident.",
            "current_status": "Investigating",
        }
        state = _state()
        official_source = workflow._web_source(official_item)
        official_state = {
            **state,
            "web_sources": [official_source],
            "web_is_current_status_question": True,
            "web_results": "official status evidence",
        }
        with patch.object(workflow, "llm", return_value=_Grader()):
            official_grade = workflow.grade_web(official_state)
            third_party_grade = workflow.grade_web(
                {
                    **official_state,
                    "web_sources": [
                        workflow._web_source({
                            **official_item,
                            "url": "https://status.example.test/incident/123",
                        })
                    ],
                }
            )
        self.assertEqual(official_grade["web_current_status_basis"], "official")
        self.assertTrue(official_grade["web_current_status_verified"])
        self.assertEqual(third_party_grade["web_current_status_basis"], "third_party")
        self.assertFalse(third_party_grade["web_current_status_verified"])


if __name__ == "__main__":
    unittest.main()
