"""
Client for interacting with the Metaculus API.
"""

import os
from typing import Any, Dict, List

import httpx

METACULUS_API_BASE_URL = "https://www.metaculus.com/api2"


class MetaculusAPI:
    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("METACULUS_TOKEN")
        if not self.token:
            raise ValueError(
                "Metaculus API token not provided or found in METACULUS_TOKEN env var."
            )

        self.headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            base_url=METACULUS_API_BASE_URL, headers=self.headers
        )

    async def close(self):
        await self.client.aclose()

    async def fetch_questions(
        self, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch questions from Metaculus.
        Example params: {"status": "open", "order_by": "-activity", "project": <project_id>}
        """
        try:
            response = await self.client.get("/questions/", params=params)
            response.raise_for_status()
            # Assuming the API returns a list of questions directly or under a 'results' key
            data = response.json()
            return data.get("results", data) if isinstance(data, dict) else data
        except httpx.HTTPStatusError as e:
            print(
                f"HTTP error fetching questions: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.RequestError as e:
            print(f"Request error fetching questions: {e}")
            raise

    async def submit_prediction(
        self, question_id: int, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit a prediction for a given question.
        Example prediction_data for binary: {"prediction": 0.75, "void": false}
        """
        try:
            response = await self.client.post(
                f"/questions/{question_id}/predict/", json=prediction_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(
                f"HTTP error submitting prediction for question {question_id}: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.RequestError as e:
            print(
                f"Request error submitting prediction for question {question_id}: {e}"
            )
            raise

    async def get_question_detail(self, question_id: int) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific question.
        """
        try:
            response = await self.client.get(f"/questions/{question_id}/")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(
                f"HTTP error fetching question {question_id}: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.RequestError as e:
            print(f"Request error fetching question {question_id}: {e}")
            raise


# Example usage (optional, for direct testing)
async def example_main():
    # Ensure METACULUS_TOKEN is set in your environment or pass it directly
    api = MetaculusAPI()
    try:
        print("Fetching open questions...")
        # You might want to filter by project or other criteria
        questions = await api.fetch_questions(
            {"status": "open", "limit": 5, "order_by": "-activity"}
        )
        if questions:
            print(f"Fetched {len(questions)} questions.")
            for q in questions:
                print(
                    f"- ID: {q['id']}, Title: {q['title_short'] if 'title_short' in q else q['title']}"
                )

            # Example: Fetch detail for the first question
            # first_q_id = questions[0]['id']
            # print(f"\nFetching details for question {first_q_id}...")
            # detail = await api.get_question_detail(first_q_id)
            # print(f"Resolution criteria: {detail.get('resolution_criteria', 'N/A')}")

            # Example: Submit a dummy prediction (BE CAREFUL WITH ACTUAL SUBMISSIONS)
            # print(f"\nSubmitting dummy prediction for question {first_q_id}...")
            # try:
            #     # This is a placeholder, actual prediction format depends on question type
            #     # For a binary question, it might be:
            #     # prediction_response = await api.submit_prediction(first_q_id, {"prediction": 0.6})
            #     # print(f"Prediction response: {prediction_response}")
            #     print("Prediction submission example commented out to prevent accidental submissions.")
            # except Exception as e:
            #     print(f"Error submitting dummy prediction: {e}")

        else:
            print("No questions fetched.")
    finally:
        await api.close()


if __name__ == "__main__":
    # import asyncio
    # asyncio.run(example_main())
    print(
        "MetaculusAPI class defined. Uncomment example_main and asyncio import to run example."
    )
