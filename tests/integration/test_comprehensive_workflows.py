"""Comprehensive integration tests for complete forecasting workflows."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.entities.prediction import Prediction
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.agent import Agent, ReasoningStyle
from src.domain.entities.research_report import ResearchReport
from src.domain.value_objects.confidence import Confidence
from src.application.services.tournament_service import TournamentService
from src.application.services.learning_service import LearningService
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker
from src.infrastructure.resilience.retry_strategy import RetryStrategy
from src.infrastructure.security.rate_limiter import RateLimiter
from src.infrastructure.monitoring.metrics_collector import MetricsCollector


@pytest.mark.asyncio
class TestComprehensiveForecastingWorkflow:
    """Test complete forecasting workflows from question ingestion to prediction submission."""

    async def test_end_to_end_binary_question_workflow(self, test_data_factory, mock_llm_client, mock_search_client):
        """Test complete workflow for binary question processing."""
        # Setup
        question = test_data_factory.create_question(
            question_type=QuestionType.BINARY,
            text="Will AI achieve AGI by 2030?"
        )

        # Mock external services
        mock_search_client.search.return_value = {
            'results': [
                {
                    'title': 'AI Progress Report 2024',
                    'url': 'https://example.com/ai-progress',
                    'snippet': 'Recent advances in AI suggest accelerating progress toward AGI.',
                    'credibility_score': 0.9
                },
                {
                    'title': 'Expert Survey on AGI Timeline',
                    'url': 'https://example.com/expert-survey',
                    'snippet': 'Experts predict AGI between 2028-2035 with median estimate of 2031.',
                    'credibility_score': 0.85
                }
            ],
            'total': 2
        }

        mock_llm_client.generate_structured_response.return_value = {
            'reasoning_steps': [
                {
                    'step_number': 1,
                    'description': 'Analyze current AI capabilities',
                    'analysis': 'Current AI systems show rapid progress in language and reasoning',
                    'confidence': 0.8
                },
                {
                    'step_number': 2,
                    'description': 'Evaluate expert predictions',
                    'analysis': 'Expert consensus suggests AGI likely in early 2030s',
                    'confidence': 0.7
                },
                {
                    'step_number': 3,
                    'description': 'Consider potential obstacles',
                    'analysis': 'Technical and regulatory challenges may cause delays',
                    'confidence': 0.6
                }
            ],
            'final_prediction': 0.65,
            'confidence_level': 0.75,
            'reasoning_summary': 'Based on current progress and expert opinions, moderate probability of AGI by 2030'
        }

        # Create workflow components
        research_service = Mock()
        research_service.conduct_research = AsyncMock(return_value=test_data_factory.create_research_report(
            question_id=question.id,
            sources_count=2
        ))

        reasoning_service = Mock()
        reasoning_service.process_question = AsyncMock(return_value={
            'reasoning_steps': [
                test_data_factory.create_reasoning_step(1, "Step 1"),
                test_data_factory.create_reasoning_step(2, "Step 2"),
                test_data_factory.create_reasoning_step(3, "Step 3")
            ],
            'final_reasoning': 'Comprehensive analysis completed',
            'confidence': Confidence(0.75, "Multi-step analysis")
        })

        prediction_service = Mock()
        prediction_service.generate_prediction = AsyncMock(return_value=Prediction.create_binary(
            question_id=question.id,
            probability=0.65,
            confidence_level=0.75,
            confidence_basis="Multi-step reasoning with expert input",
            method="chain_of_thought",
            reasoning="Based on current AI progress and expert consensus",
            created_by="test_agent"
        ))

        # Execute workflow
        start_time = datetime.utcnow()

        # Step 1: Research
        research_report = await research_service.conduct_research(question)
        assert research_report is not None
        assert len(research_report.sources) == 2

        # Step 2: Reasoning
        reasoning_result = await reasoning_service.process_question(question)
        assert len(reasoning_result['reasoning_steps']) == 3
        assert reasoning_result['confidence'].level == 0.75

        # Step 3: Prediction
        prediction = await prediction_service.generate_prediction(question)
        assert prediction.is_binary_prediction()
        assert 0.0 <= prediction.get_binary_probability() <= 1.0
        assert prediction.confidence.level == 0.75

        # Verify workflow timing
        end_time = datetime.utcnow()
        workflow_duration = (end_time - start_time).total_seconds()
        assert workflow_duration < 30.0  # Should complete within 30 seconds

        # Verify prediction quality
        assert prediction.has_high_confidence() or prediction.confidence.level >= 0.7
        assert len(prediction.reasoning) > 10  # Should have substantial reasoning
        assert prediction.evidence_sources is not None

    async def test_ensemble_agent_workflow(self, test_data_factory, mock_llm_client):
        """Test ensemble agent workflow with multiple reasoning approaches."""
        question = test_data_factory.create_question(
            text="What will be the global temperature anomaly in 2025?"
        )

        # Mock different agent responses
        agent_responses = {
            'chain_of_thought': {
                'prediction': 0.72,
                'confidence': 0.8,
                'reasoning': 'Step-by-step analysis of climate trends'
            },
            'tree_of_thought': {
                'prediction': 0.68,
                'confidence': 0.75,
                'reasoning': 'Explored multiple reasoning branches'
            },
            'react': {
                'prediction': 0.75,
                'confidence': 0.85,
                'reasoning': 'Iterative reasoning with evidence gathering'
            }
        }

        # Create mock agents
        agents = []
        predictions = []

        for agent_type, response in agent_responses.items():
            agent = test_data_factory.create_agent(
                agent_id=f"{agent_type}_agent",
                reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT
            )
            agents.append(agent)

            prediction = Prediction.create_binary(
                question_id=question.id,
                probability=response['prediction'],
                confidence_level=response['confidence'],
                confidence_basis=f"{agent_type} analysis",
                method=agent_type,
                reasoning=response['reasoning'],
                created_by=agent.id
            )
            predictions.append(prediction)

        # Test ensemble aggregation
        ensemble_service = Mock()
        ensemble_service.aggregate_predictions = Mock(return_value={
            'final_prediction': 0.717,  # Weighted average
            'consensus_strength': 0.85,
            'prediction_variance': 0.0012,
            'agent_diversity_score': 0.6,
            'confidence_alignment': 0.8,
            'aggregation_method': 'confidence_weighted'
        })

        # Execute ensemble workflow
        ensemble_result = ensemble_service.aggregate_predictions(predictions)

        # Verify ensemble properties
        assert 0.0 <= ensemble_result['final_prediction'] <= 1.0
        assert ensemble_result['consensus_strength'] > 0.8  # High consensus
        assert ensemble_result['prediction_variance'] < 0.01  # Low variance
        assert ensemble_result['agent_diversity_score'] > 0.5  # Good diversity

        # Verify final prediction is reasonable average
        individual_predictions = [p.get_binary_probability() for p in predictions]
        avg_prediction = sum(individual_predictions) / len(individual_predictions)
        assert abs(ensemble_result['final_prediction'] - avg_prediction) < 0.1

    async def test_tournament_strategy_workflow(self, test_data_factory):
        """Test tournament strategy optimization workflow."""
        # Create tournament with diverse questions
        questions = [
            test_data_factory.create_question(1, scoring_weight=1.0, deadline_offset_days=30),
            test_data_factory.create_question(2, scoring_weight=3.0, deadline_offset_days=15),  # High value
            test_data_factory.create_question(3, scoring_weight=2.0, deadline_offset_days=2),   # Urgent
            test_data_factory.create_question(4, scoring_weight=1.5, deadline_offset_days=45),
            test_data_factory.create_question(5, scoring_weight=4.0, deadline_offset_days=7)    # High value + urgent
        ]

        tournament = test_data_factory.create_tournament(
            questions=questions,
            current_standings={'our_agent': 0.65, 'competitor1': 0.72, 'competitor2': 0.58}
        )

        # Mock tournament service
        tournament_service = TournamentService()
        tournament_service.question_categorizer = Mock()
        tournament_service.strategy_selector = Mock()
        tournament_service.timing_optimizer = Mock()

        # Mock strategy analysis
        tournament_service.analyze_tournament_questions = Mock(return_value={
            'high_value_questions': [questions[1], questions[4]],  # Weight >= 3.0
            'urgent_questions': [questions[2], questions[4]],      # Deadline <= 7 days
            'strategic_opportunities': [questions[1], questions[4]], # High value + good timing
            'recommended_focus': questions[4]  # Highest value + urgent
        })

        tournament_service.optimize_resource_allocation = Mock(return_value={
            'question_priorities': {
                questions[4].id: 0.9,  # Highest priority
                questions[1].id: 0.8,  # High priority
                questions[2].id: 0.7,  # Medium-high priority
                questions[3].id: 0.4,  # Medium priority
                questions[0].id: 0.3   # Low priority
            },
            'resource_allocation': {
                questions[4].id: 0.35,  # 35% of resources
                questions[1].id: 0.25,  # 25% of resources
                questions[2].id: 0.20,  # 20% of resources
                questions[3].id: 0.15,  # 15% of resources
                questions[0].id: 0.05   # 5% of resources
            },
            'expected_score_improvement': 0.08
        })

        # Execute strategy workflow
        strategy_analysis = tournament_service.analyze_tournament_questions(tournament.questions)
        resource_allocation = tournament_service.optimize_resource_allocation(
            tournament.questions,
            tournament.current_standings
        )

        # Verify strategy analysis
        assert len(strategy_analysis['high_value_questions']) == 2
        assert len(strategy_analysis['urgent_questions']) == 2
        assert strategy_analysis['recommended_focus'] == questions[4]

        # Verify resource allocation
        priorities = resource_allocation['question_priorities']
        assert priorities[questions[4].id] > priorities[questions[1].id]  # Urgent + high value prioritized
        assert priorities[questions[1].id] > priorities[questions[0].id]  # High value over low value

        # Verify resource allocation sums to 1.0
        total_allocation = sum(resource_allocation['resource_allocation'].values())
        assert abs(total_allocation - 1.0) < 0.01

        # Verify expected improvement
        assert resource_allocation['expected_score_improvement'] > 0

    async def test_error_handling_and_recovery_workflow(self, test_data_factory):
        """Test error handling and recovery in forecasting workflow."""
        question = test_data_factory.create_question()

        # Test circuit breaker behavior
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Simulate service failures
        failing_service = AsyncMock()
        failing_service.side_effect = Exception("Service unavailable")

        # Test circuit breaker opening
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_service)

        # Circuit should now be open
        assert circuit_breaker.state.name == 'OPEN'

        # Test retry strategy
        retry_strategy = RetryStrategy(max_attempts=3, base_delay=0.1, max_delay=1.0)

        # Mock service that fails twice then succeeds
        attempt_count = 0
        async def flaky_service():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:
                raise Exception("Temporary failure")
            return "Success"

        # Should succeed after retries
        result = await retry_strategy.execute(flaky_service)
        assert result == "Success"
        assert attempt_count == 3

        # Test graceful degradation
        degradation_manager = Mock()
        degradation_manager.get_available_search_providers = Mock(return_value=['duckduckgo'])
        degradation_manager.fallback_to_single_agent = Mock(return_value=test_data_factory.create_agent())

        # Simulate primary services failing
        available_providers = degradation_manager.get_available_search_providers()
        assert len(available_providers) >= 1  # Should have at least one fallback

        fallback_agent = degradation_manager.fallback_to_single_agent('chain_of_thought')
        assert fallback_agent is not None

    async def test_performance_monitoring_workflow(self, test_data_factory):
        """Test performance monitoring throughout forecasting workflow."""
        question = test_data_factory.create_question()

        # Mock metrics collector
        metrics_collector = MetricsCollector()
        metrics_collector.start_timer = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        metrics_collector.increment = Mock()
        metrics_collector.gauge = Mock()
        metrics_collector.histogram = Mock()

        # Simulate workflow with monitoring
        with metrics_collector.start_timer('forecasting_workflow_duration'):
            # Research phase
            with metrics_collector.start_timer('research_duration'):
                research_start = datetime.utcnow()
                await asyncio.sleep(0.1)  # Simulate research time
                research_end = datetime.utcnow()
                research_duration = (research_end - research_start).total_seconds()

                metrics_collector.histogram('research_duration_seconds', research_duration)
                metrics_collector.increment('research_requests_total')

            # Reasoning phase
            with metrics_collector.start_timer('reasoning_duration'):
                reasoning_start = datetime.utcnow()
                await asyncio.sleep(0.05)  # Simulate reasoning time
                reasoning_end = datetime.utcnow()
                reasoning_duration = (reasoning_end - reasoning_start).total_seconds()

                metrics_collector.histogram('reasoning_duration_seconds', reasoning_duration)
                metrics_collector.increment('reasoning_requests_total')

            # Prediction phase
            with metrics_collector.start_timer('prediction_duration'):
                prediction_start = datetime.utcnow()
                await asyncio.sleep(0.02)  # Simulate prediction time
                prediction_end = datetime.utcnow()
                prediction_duration = (prediction_end - prediction_start).total_seconds()

                metrics_collector.histogram('prediction_duration_seconds', prediction_duration)
                metrics_collector.increment('predictions_generated_total')

        # Verify monitoring calls
        assert metrics_collector.start_timer.call_count >= 4  # Workflow + 3 phases
        assert metrics_collector.histogram.call_count >= 3   # Duration metrics
        assert metrics_collector.increment.call_count >= 3   # Counter metrics

    async def test_security_validation_workflow(self, test_data_factory, malicious_inputs):
        """Test security validation throughout forecasting workflow."""
        # Test input validation
        input_validator = Mock()
        input_validator.validate_question_text = Mock(return_value=True)
        input_validator.sanitize_input = Mock(side_effect=lambda x: x.replace('<script>', ''))

        # Test with malicious inputs
        for xss_payload in malicious_inputs['xss_payloads']:
            sanitized = input_validator.sanitize_input(xss_payload)
            assert '<script>' not in sanitized

        # Test rate limiting
        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Should allow requests within limit
        for i in range(10):
            allowed = await rate_limiter.is_allowed(f"user_{i}")
            assert allowed

        # Should block requests over limit
        blocked = await rate_limiter.is_allowed("user_1")  # Same user, over limit
        assert not blocked

        # Test credential validation
        credential_manager = Mock()
        credential_manager.validate_api_key = Mock(return_value=True)
        credential_manager.mask_sensitive_data = Mock(return_value={'api_key': '***MASKED***'})

        # Verify credential handling
        assert credential_manager.validate_api_key('valid_key')
        masked_data = credential_manager.mask_sensitive_data({'api_key': 'secret123'})
        assert 'secret123' not in str(masked_data)

    async def test_learning_and_adaptation_workflow(self, test_data_factory):
        """Test learning and adaptation workflow."""
        # Create historical data
        questions = [test_data_factory.create_question(i) for i in range(1, 6)]
        predictions = []
        actual_outcomes = [0.8, 0.3, 0.9, 0.1, 0.7]  # Simulated actual outcomes

        for i, (question, outcome) in enumerate(zip(questions, actual_outcomes)):
            prediction = Prediction.create_binary(
                question_id=question.id,
                probability=0.6 + (i * 0.1),  # Varying predictions
                confidence_level=0.7 + (i * 0.05),
                confidence_basis=f"Analysis {i+1}",
                method="chain_of_thought",
                reasoning=f"Reasoning for question {i+1}",
                created_by="learning_agent"
            )
            predictions.append(prediction)

        # Mock learning service
        learning_service = LearningService()
        learning_service.analyze_prediction_accuracy = Mock(return_value={
            'overall_accuracy': 0.72,
            'calibration_score': 0.68,
            'brier_score': 0.18,
            'overconfidence_bias': 0.05,
            'accuracy_by_category': {
                'ai_development': 0.75,
                'technology': 0.70,
                'science': 0.68
            }
        })

        learning_service.identify_improvement_opportunities = Mock(return_value={
            'calibration_adjustment': -0.03,  # Slightly overconfident
            'category_specialization': {
                'ai_development': 'maintain_current_approach',
                'technology': 'increase_research_depth',
                'science': 'seek_expert_input'
            },
            'confidence_threshold_adjustment': 0.02
        })

        learning_service.update_agent_parameters = Mock(return_value={
            'updated_parameters': {
                'confidence_adjustment': -0.03,
                'research_depth_multiplier': 1.2,
                'expert_consultation_threshold': 0.75
            },
            'expected_improvement': 0.04
        })

        # Execute learning workflow
        accuracy_analysis = learning_service.analyze_prediction_accuracy(predictions, actual_outcomes)
        improvement_opportunities = learning_service.identify_improvement_opportunities(accuracy_analysis)
        updated_parameters = learning_service.update_agent_parameters(improvement_opportunities)

        # Verify learning analysis
        assert 0.0 <= accuracy_analysis['overall_accuracy'] <= 1.0
        assert 0.0 <= accuracy_analysis['calibration_score'] <= 1.0
        assert accuracy_analysis['brier_score'] >= 0.0

        # Verify improvement identification
        assert 'calibration_adjustment' in improvement_opportunities
        assert 'category_specialization' in improvement_opportunities

        # Verify parameter updates
        assert 'updated_parameters' in updated_parameters
        assert updated_parameters['expected_improvement'] > 0

    async def test_concurrent_question_processing_workflow(self, test_data_factory):
        """Test concurrent processing of multiple questions."""
        # Create multiple questions
        questions = [test_data_factory.create_question(i) for i in range(1, 11)]

        # Mock processing service
        async def process_single_question(question):
            # Simulate processing time
            await asyncio.sleep(0.1)
            return Prediction.create_binary(
                question_id=question.id,
                probability=0.5 + (question.id * 0.05),
                confidence_level=0.7,
                confidence_basis="Concurrent processing",
                method="parallel_processing",
                reasoning=f"Processed question {question.id}",
                created_by="concurrent_agent"
            )

        # Process questions concurrently
        start_time = datetime.utcnow()

        # Use asyncio.gather for concurrent processing
        predictions = await asyncio.gather(*[
            process_single_question(question) for question in questions
        ])

        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()

        # Verify concurrent processing
        assert len(predictions) == 10
        assert total_duration < 1.0  # Should be much faster than sequential (which would take ~1 second)

        # Verify all predictions are valid
        for prediction in predictions:
            assert prediction.is_binary_prediction()
            assert 0.0 <= prediction.get_binary_probability() <= 1.0
            assert prediction.confidence.level > 0.0

        # Verify predictions have different values (based on question ID)
        probabilities = [p.get_binary_probability() for p in predictions]
        assert len(set(probabilities)) > 1  # Should have different values

    async def test_resource_optimization_workflow(self, test_data_factory):
        """Test resource optimization during high-load scenarios."""
        # Create high-load scenario
        questions = [test_data_factory.create_question(i) for i in range(1, 101)]  # 100 questions

        # Mock resource manager
        resource_manager = Mock()
        resource_manager.get_available_resources = Mock(return_value={
            'cpu_usage': 0.65,
            'memory_usage': 0.70,
            'api_rate_limits': {
                'search_api': {'remaining': 450, 'limit': 500},
                'llm_api': {'remaining': 180, 'limit': 200}
            },
            'concurrent_capacity': 20
        })

        resource_manager.optimize_batch_size = Mock(return_value=15)
        resource_manager.prioritize_questions = Mock(return_value=questions[:20])  # Top 20 priority

        # Execute resource optimization
        available_resources = resource_manager.get_available_resources()
        optimal_batch_size = resource_manager.optimize_batch_size(available_resources)
        priority_questions = resource_manager.prioritize_questions(questions)

        # Verify resource optimization
        assert available_resources['cpu_usage'] < 0.8  # Under CPU limit
        assert available_resources['memory_usage'] < 0.8  # Under memory limit
        assert optimal_batch_size <= available_resources['concurrent_capacity']
        assert len(priority_questions) <= len(questions)

        # Simulate batch processing with optimized parameters
        batches = [priority_questions[i:i+optimal_batch_size]
                  for i in range(0, len(priority_questions), optimal_batch_size)]

        assert len(batches) <= 2  # Should fit in reasonable number of batches
        for batch in batches:
            assert len(batch) <= optimal_batch_size
