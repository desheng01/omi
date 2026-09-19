"""Hermetic tests for the ClinicalTrials.gov Omi integration."""

import json
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs

import httpx
from fastapi.testclient import TestClient


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import clinicaltrials
import main
from models import (
    ChatToolResponse,
    RecruitingTrialsRequest,
    SearchTrialsRequest,
    TrialDetailsRequest,
)


SEARCH_PAYLOAD = {
    "totalCount": 2,
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "briefTitle": "A study of treatment A",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                },
                "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                "designModule": {"phases": ["PHASE2"]},
                "contactsLocationsModule": {
                    "locations": [
                        {
                            "facility": "Example Hospital",
                            "city": "Boston",
                            "state": "Massachusetts",
                            "country": "United States",
                        }
                    ]
                },
            }
        },
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000002",
                    "briefTitle": "A study of treatment B",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                },
                "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                "designModule": {"phases": ["PHASE3"]},
                "contactsLocationsModule": {
                    "locations": [
                        {
                            "facility": "Another Hospital",
                            "city": "Chicago",
                            "state": "Illinois",
                            "country": "United States",
                        }
                    ]
                },
            }
        },
    ],
}

DETAIL_PAYLOAD = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "officialTitle": "A randomized study of treatment A",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "startDateStruct": {"date": "2026-01-01"},
            "primaryCompletionDateStruct": {"date": "2027-01-01"},
        },
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example University"}},
        "descriptionModule": {"briefSummary": "A short public study summary."},
        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2"],
            "enrollmentInfo": {"count": 120, "type": "ESTIMATED"},
        },
        "eligibilityModule": {"sex": "ALL", "minimumAge": "18 Years"},
        "contactsLocationsModule": {
            "locations": [
                {
                    "facility": "Example Hospital",
                    "city": "Boston",
                    "country": "United States",
                }
            ]
        },
    }
}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class ModelTests(unittest.TestCase):
    def test_search_requires_a_term(self):
        with self.assertRaises(ValueError):
            SearchTrialsRequest()

    def test_detail_normalizes_nct_id(self):
        request = TrialDetailsRequest(nct_id="nct04280705")
        self.assertEqual(request.nct_id, "NCT04280705")

    def test_chat_response_requires_exactly_one_outcome(self):
        with self.assertRaises(ValueError):
            ChatToolResponse()
        with self.assertRaises(ValueError):
            ChatToolResponse(result="ok", error="bad")


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_builds_provider_query_and_formats_results(self):
        seen = {}

        def handler(request):
            seen.update(parse_qs(request.url.query.decode()))
            return httpx.Response(200, json=SEARCH_PAYLOAD, request=request)

        async with _client(handler) as client:
            result = await clinicaltrials.search_trials(
                client,
                SearchTrialsRequest(
                    condition="type 2 diabetes",
                    location="Boston",
                    phase="PHASE2",
                    page_size=2,
                ),
            )

        self.assertEqual(seen["query.cond"], ["type 2 diabetes"])
        self.assertEqual(seen["query.locn"], ["Boston"])
        self.assertEqual(seen["filter.overallStatus"], ["RECRUITING"])
        self.assertEqual(seen["filter.advanced"], ["AREA[Phase](PHASE2)"])
        self.assertNotIn("uid", seen)
        self.assertIn("Found 2 matching studies", result)
        self.assertIn("NCT00000001", result)
        self.assertIn("Example Hospital, Boston", result)
        self.assertIn("not medical advice", result)

    async def test_search_empty_result_is_successful(self):
        def handler(request):
            return httpx.Response(
                200, json={"totalCount": 0, "studies": []}, request=request
            )

        async with _client(handler) as client:
            result = await clinicaltrials.search_trials(
                client,
                SearchTrialsRequest(condition="a very rare condition"),
            )

        self.assertIn("No ClinicalTrials.gov studies matched", result)

    async def test_get_trial_formats_details(self):
        def handler(request):
            self.assertTrue(request.url.path.endswith("/NCT00000001"))
            return httpx.Response(200, json=DETAIL_PAYLOAD, request=request)

        async with _client(handler) as client:
            result = await clinicaltrials.get_trial(client, "NCT00000001")

        self.assertIn("NCT00000001", result)
        self.assertIn("Example University", result)
        self.assertIn("Enrollment: 120", result)
        self.assertIn("https://clinicaltrials.gov/study/NCT00000001", result)

    async def test_provider_404_returns_safe_error(self):
        def handler(request):
            return httpx.Response(404, json={"error": "not found"}, request=request)

        async with _client(handler) as client:
            with self.assertRaisesRegex(
                clinicaltrials.ClinicalTrialsError, "study not found"
            ):
                await clinicaltrials.get_trial(client, "NCT00000001")

    async def test_malformed_payload_fails_closed(self):
        def handler(request):
            return httpx.Response(200, content=b"[]", request=request)

        async with _client(handler) as client:
            with self.assertRaisesRegex(
                clinicaltrials.ClinicalTrialsError, "unexpected response shape"
            ):
                await clinicaltrials.get_trial(client, "NCT00000001")

    async def test_response_size_limit(self):
        def handler(request):
            return httpx.Response(
                200,
                content=b"{" + b" " * clinicaltrials.MAX_RESPONSE_BYTES + b"}",
                request=request,
            )

        async with _client(handler) as client:
            with self.assertRaisesRegex(
                clinicaltrials.ClinicalTrialsError, "size limit"
            ):
                await clinicaltrials.get_trial(client, "NCT00000001")


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.handler = lambda request: httpx.Response(
            200, json=SEARCH_PAYLOAD, request=request
        )
        self.test_client = TestClient(main.app)
        self.test_client.__enter__()
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: self.handler(request))
        )
        main.app.state.http_client = self.client

    def tearDown(self):
        self.test_client.__exit__(None, None, None)

    def test_health_and_manifest(self):
        self.assertEqual(self.test_client.get("/health").json(), {"status": "ok"})
        manifest = self.test_client.get("/.well-known/omi-tools.json").json()
        self.assertEqual(len(manifest["tools"]), 3)
        self.assertEqual(
            {tool["name"] for tool in manifest["tools"]},
            {
                "search_clinical_trials",
                "find_recruiting_trials",
                "get_clinical_trial",
            },
        )

    def test_validation_errors_use_chat_tool_envelope(self):
        response = self.test_client.post(
            "/tools/search_clinical_trials", json={"uid": "ignored"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["result"])
        self.assertIn("invalid clinical-trials request", payload["error"])

    def test_endpoint_returns_result_and_ignores_uid(self):
        response = self.test_client.post(
            "/tools/search_clinical_trials",
            json={"uid": "not-forwarded", "condition": "diabetes"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["error"])
        self.assertIn("NCT00000001", payload["result"])


if __name__ == "__main__":
    unittest.main()
