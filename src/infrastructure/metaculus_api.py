"""
Infrastructure layer for Metaculus API integration.

Provides mock client functionality for fetching dummy question data.
No actual API calls are made - uses dummy JSON data for testing purposes.
"""

import json
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class MetaculusAPIError(Exception):
    """Custom exception for Metaculus API related errors."""
    pass


@dataclass
class APIConfig:
    """Configuration for Metaculus API client."""
    base_url: str = "https://www.metaculus.com/api2/"
    timeout: int = 30
    max_retries: int = 3
    mock_mode: bool = True  # Always true for this implementation


class MetaculusAPI:
    """
    Mock Metaculus API client for fetching dummy question data.
    
    This is a mock implementation that generates dummy JSON data
    instead of making actual API calls to Metaculus.
    """

    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize the Metaculus API client.
        
        Args:
            config: API configuration. If None, uses default config.
        """
        self.config = config or APIConfig()
        self._dummy_data = self._generate_dummy_questions()

    def fetch_questions(self, limit: Optional[int] = None, 
                       status: str = "open",
                       category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch questions from the mock Metaculus API.
        
        Args:
            limit: Maximum number of questions to return
            status: Question status filter (open, closed, resolved)
            category: Question category filter
            
        Returns:
            List of question dictionaries in Metaculus API format
            
        Raises:
            MetaculusAPIError: If there's an error fetching questions
        """
        try:
            # Simulate API response delay
            import time
            time.sleep(0.1)
            
            # Filter dummy data based on parameters
            filtered_questions = self._filter_questions(
                self._dummy_data, status=status, category=category
            )
            
            # Apply limit if specified
            if limit is not None:
                filtered_questions = filtered_questions[:limit]
            
            return filtered_questions
            
        except Exception as e:
            raise MetaculusAPIError(f"Failed to fetch questions: {str(e)}") from e

    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the mock API data.
        
        Returns:
            Dictionary with API statistics
        """
        total_questions = len(self._dummy_data)
        open_questions = len([q for q in self._dummy_data if not q.get("is_resolved", False)])
        resolved_questions = total_questions - open_questions
        
        return {
            "total_questions": total_questions,
            "open_questions": open_questions,
            "resolved_questions": resolved_questions,
            "mock_mode": self.config.mock_mode,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def _filter_questions(self, questions: List[Dict[str, Any]], 
                         status: str = "open", 
                         category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter questions based on status and category.
        """
        filtered = questions.copy()
        
        # Filter by status
        if status == "open":
            filtered = [q for q in filtered if not q.get("is_resolved", False)]
        elif status == "resolved":
            filtered = [q for q in filtered if q.get("is_resolved", False)]
        
        # Filter by category if specified
        if category:
            filtered = [q for q in filtered if q.get("category") == category]
        
        return filtered

    def _generate_dummy_questions(self) -> List[Dict[str, Any]]:
        """
        Generate dummy question data for testing.
        """
        base_time = datetime.now(timezone.utc)
        
        dummy_questions = [
            {
                "id": 101,
                "title": "Will AI achieve AGI by 2030?",
                "description": "This question asks whether artificial general intelligence (AGI) will be achieved by the end of 2030.",
                "question_type": "binary",
                "url": "https://metaculus.com/questions/101/",
                "close_time": (base_time + timedelta(days=365*6)).isoformat(),
                "created_time": (base_time - timedelta(days=30)).isoformat(),
                "is_resolved": False,
                "resolution": None,
                "community_prediction": 0.35,
                "num_forecasters": 245,
                "category": "technology",
                "tags": ["ai", "agi", "technology"],
                "status": "open"
            },
            {
                "id": 105,
                "title": "Will SpaceX land on Mars by 2030?",
                "description": "Will SpaceX successfully land a crewed mission on Mars by December 31, 2030?",
                "question_type": "binary",
                "url": "https://metaculus.com/questions/105/",
                "close_time": (base_time + timedelta(days=365*6)).isoformat(),
                "created_time": (base_time - timedelta(days=120)).isoformat(),
                "is_resolved": False,
                "resolution": None,
                "community_prediction": 0.25,
                "num_forecasters": 445,
                "category": "technology",
                "tags": ["spacex", "mars", "space"],
                "status": "open"
            },
            {
                "id": 106,
                "title": "COVID-19 pandemic end date",
                "description": "This question was about when the COVID-19 pandemic would officially end according to WHO declaration.",
                "question_type": "date",
                "url": "https://metaculus.com/questions/106/",
                "close_time": (base_time - timedelta(days=30)).isoformat(),
                "created_time": (base_time - timedelta(days=400)).isoformat(),
                "is_resolved": True,
                "resolution": "2023-05-15",
                "community_prediction": None,
                "num_forecasters": 234,
                "category": "health",
                "tags": ["covid", "pandemic", "health"],
                "status": "resolved"
            }
        ]
        
        return dummy_questions
