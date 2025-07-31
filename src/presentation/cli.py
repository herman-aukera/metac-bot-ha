"""Command Line Interface for Tournament Optimization System.

This module provides a comprehensive CLI for interacting with the tournament
optimization system, including question processing, tournament analysis,
and system monitoring.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
from tabulate import tabulate

from ..application.use_cases.process_tournament_question import (
    ProcessTournamentQuestionUseCase, ProcessingResult
)
from ..application.services.forecasting_pipeline import ForecastingPipeline
from ..application.services.tournament_service import TournamentService
from ..application.services.learning_service import LearningService
from ..domain.entities.question import Question, QuestionType, QuestionCategory
from ..infrastructure.logging.structured_logger import StructuredLogger
from ..infrastructure.monitoring.health_check_manager import HealthCheckManager


logger = StructuredLogger(__name__)


class TournamentCLI:
    """Main CLI class for tournament optimization system."""

    def __init__(self,
                 process_question_use_case: ProcessTournamentQuestionUseCase,
                 forecasting_pipeline: ForecastingPipeline,
                 tournament_service: TournamentService,
                 learning_service: LearningService,
                 health_check_manager: HealthCheckManager):
        """Initialize CLI with required services.

        Args:
            process_question_use_case: Main question processing use case
            forecasting_pipeline: Forecasting pipeline service
            tournament_service: Tournament strategy service
            learning_service: Learning and adaptation service
            health_check_manager: Health check manager
        """
        self.process_question_use_case = process_question_use_case
        self.forecasting_pipeline = forecasting_pipeline
        self.tournament_service = tournament_service
        self.learning_service = learning_service
        self.health_check_manager = health_check_manager

    async def process_question(self,
                             question_id: int,
                             tournament_id: Optional[int] = None,
                             force_reprocess: bool = False,
                             dry_run: bool = False,
                             output_format: str = 'json') -> Dict[str, Any]:
        """Process a single question through the forecasting pipeline.

        Args:
            question_id: ID of question to process
            tournament_id: Optional tournament context
            force_reprocess: Whether to force reprocessing
            dry_run: Whether to run without submitting
            output_format: Output format (json, table, summary)

        Returns:
            Processing result dictionary
        """
        try:
            # Load question (in real implementation, this would fetch from API)
            question = await self._load_question(question_id)

            if not question:
                return {
                    'success': False,
                    'error': f'Question {question_id} not found',
                    'question_id': question_id
                }

            # Execute processing
            result = await self.process_question_use_case.execute(
                question=question,
                tournament_id=tournament_id,
                force_reprocess=force_reprocess,
                submission_mode=not dry_run
            )

            # Format output
            output = self._format_processing_result(result, output_format)

            if result.success:
                logger.info(
                    "Question processed successfully via CLI",
                    extra={
                        'question_id': question_id,
                        'tournament_id': tournament_id,
                        'processing_time': result.processing_time,
                        'dry_run': dry_run
                    }
                )
            else:
                logger.error(
                    "Question processing failed via CLI",
                    extra={
                        'question_id': question_id,
                        'error': result.error_message
                    }
                )

            return output

        except Exception as e:
            logger.error(
                "CLI question processing failed",
                extra={
                    'question_id': question_id,
                    'error': str(e)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'question_id': question_id
            }

    async def process_tournament(self,
                               tournament_id: int,
                               max_questions: Optional[int] = None,
                               parallel_processing: bool = True,
                               dry_run: bool = False) -> Dict[str, Any]:
        """Process all questions in a tournament.

        Args:
            tournament_id: Tournament ID to process
            max_questions: Maximum number of questions to process
            parallel_processing: Whether to process questions in parallel
            dry_run: Whether to run without submitting

        Returns:
            Tournament processing results
        """
        try:
            # Get tournament and questions
            tournament = await self.tournament_service.get_tournament(tournament_id)
            if not tournament:
                return {
                    'success': False,
                    'error': f'Tournament {tournament_id} not found'
                }

            active_questions = tournament.get_active_questions()
            if max_questions:
                active_questions = active_questions[:max_questions]

            logger.info(
                "Starting tournament processing",
                extra={
                    'tournament_id': tournament_id,
                    'total_questions': len(active_questions),
                    'parallel_processing': parallel_processing,
                    'dry_run': dry_run
                }
            )

            # Process questions
            if parallel_processing:
                results = await self._process_questions_parallel(
                    active_questions, tournament_id, dry_run
                )
            else:
                results = await self._process_questions_sequential(
                    active_questions, tournament_id, dry_run
                )

            # Aggregate results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            summary = {
                'tournament_id': tournament_id,
                'total_questions': len(active_questions),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(active_questions) if active_questions else 0,
                'total_processing_time': sum(r.processing_time for r in results),
                'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
                'results': [self._format_processing_result(r, 'summary') for r in results]
            }

            logger.info(
                "Tournament processing completed",
                extra={
                    'tournament_id': tournament_id,
                    'successful': len(successful),
                    'failed': len(failed),
                    'success_rate': summary['success_rate']
                }
            )

            return summary

        except Exception as e:
            logger.error(
                "Tournament processing failed",
                extra={
                    'tournament_id': tournament_id,
                    'error': str(e)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'tournament_id': tournament_id
            }

    async def analyze_tournament_strategy(self,
                                        tournament_id: int,
                                        output_format: str = 'table') -> Dict[str, Any]:
        """Analyze tournament strategy and provide recommendations.

        Args:
            tournament_id: Tournament ID to analyze
            output_format: Output format (json, table, summary)

        Returns:
            Strategy analysis results
        """
        try:
            tournament = await self.tournament_service.get_tournament(tournament_id)
            if not tournament:
                return {
                    'success': False,
                    'error': f'Tournament {tournament_id} not found'
                }

            # Analyze strategy
            strategy_recommendation = await self.tournament_service.analyze_tournament_strategy(tournament)

            # Format output
            if output_format == 'table':
                output = self._format_strategy_table(strategy_recommendation)
            elif output_format == 'summary':
                output = self._format_strategy_summary(strategy_recommendation)
            else:
                output = {
                    'tournament_id': tournament_id,
                    'strategy_type': strategy_recommendation.strategy_type.value,
                    'confidence': strategy_recommendation.confidence.level,
                    'expected_score_impact': strategy_recommendation.expected_score_impact,
                    'risk_level': strategy_recommendation.risk_level,
                    'reasoning': strategy_recommendation.reasoning,
                    'question_allocation': strategy_recommendation.question_allocation,
                    'alternatives': [
                        {'strategy': alt[0].value, 'score': alt[1]}
                        for alt in strategy_recommendation.alternatives
                    ]
                }

            return output

        except Exception as e:
            logger.error(
                "Strategy analysis failed",
                extra={
                    'tournament_id': tournament_id,
                    'error': str(e)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'tournament_id': tournament_id
            }

    async def show_system_status(self) -> Dict[str, Any]:
        """Show comprehensive system status and health checks.

        Returns:
            System status information
        """
        try:
            # Run health checks
            health_results = await self.health_check_manager.run_all_checks()

            # Get system metrics
            pipeline_metrics = self.forecasting_pipeline.metrics

            # Get learning service stats
            learning_stats = await self.learning_service.get_system_stats()

            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': 'healthy' if all(
                    check.is_healthy for check in health_results.values()
                ) else 'degraded',
                'health_checks': {
                    name: {
                        'status': 'healthy' if check.is_healthy else 'unhealthy',
                        'message': check.message,
                        'last_check': check.timestamp.isoformat()
                    }
                    for name, check in health_results.items()
                },
                'pipeline_metrics': {
                    'cache_hit_rate': (
                        pipeline_metrics.cache_hits /
                        (pipeline_metrics.cache_hits + pipeline_metrics.cache_misses)
                        if (pipeline_metrics.cache_hits + pipeline_metrics.cache_misses) > 0 else 0
                    ),
                    'average_consensus_strength': pipeline_metrics.consensus_strength,
                    'recent_errors': pipeline_metrics.errors_encountered[-10:],  # Last 10 errors
                    'stage_timings': {
                        stage.value: timing
                        for stage, timing in pipeline_metrics.stage_timings.items()
                    }
                },
                'learning_stats': learning_stats
            }

            return status

        except Exception as e:
            logger.error(
                "System status check failed",
                extra={'error': str(e)},
                exc_info=True
            )
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': 'error',
                'error': str(e)
            }

    async def export_results(self,
                           tournament_id: Optional[int] = None,
                           question_ids: Optional[List[int]] = None,
                           output_file: Optional[str] = None,
                           format: str = 'json') -> Dict[str, Any]:
        """Export processing results to file.

        Args:
            tournament_id: Optional tournament ID to export
            question_ids: Optional specific question IDs to export
            output_file: Output file path
            format: Export format (json, csv, xlsx)

        Returns:
            Export result information
        """
        try:
            # Collect results based on parameters
            if tournament_id:
                results = await self.learning_service.get_tournament_results(tournament_id)
            elif question_ids:
                results = await self.learning_service.get_question_results(question_ids)
            else:
                results = await self.learning_service.get_all_recent_results()

            # Generate output file name if not provided
            if not output_file:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                if tournament_id:
                    output_file = f'tournament_{tournament_id}_results_{timestamp}.{format}'
                else:
                    output_file = f'forecasting_results_{timestamp}.{format}'

            # Export based on format
            if format == 'json':
                await self._export_json(results, output_file)
            elif format == 'csv':
                await self._export_csv(results, output_file)
            elif format == 'xlsx':
                await self._export_xlsx(results, output_file)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            return {
                'success': True,
                'output_file': output_file,
                'records_exported': len(results),
                'format': format
            }

        except Exception as e:
            logger.error(
                "Results export failed",
                extra={
                    'tournament_id': tournament_id,
                    'question_ids': question_ids,
                    'output_file': output_file,
                    'format': format,
                    'error': str(e)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e)
            }

    # Helper methods

    async def _load_question(self, question_id: int) -> Optional[Question]:
        """Load question by ID (placeholder implementation).

        Args:
            question_id: Question ID to load

        Returns:
            Question object or None if not found
        """
        # In real implementation, this would fetch from Metaculus API or database
        # For now, return a mock question for testing
        return Question(
            id=question_id,
            text=f"Sample question {question_id}",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=datetime.utcnow().replace(hour=23, minute=59),
            background="Sample background information",
            resolution_criteria="Sample resolution criteria",
            scoring_weight=1.0
        )

    async def _process_questions_parallel(self,
                                        questions: List[Question],
                                        tournament_id: int,
                                        dry_run: bool) -> List[ProcessingResult]:
        """Process questions in parallel.

        Args:
            questions: Questions to process
            tournament_id: Tournament ID
            dry_run: Whether to run without submitting

        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single_question(question: Question) -> ProcessingResult:
            async with semaphore:
                return await self.process_question_use_case.execute(
                    question=question,
                    tournament_id=tournament_id,
                    submission_mode=not dry_run
                )

        tasks = [process_single_question(q) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _process_questions_sequential(self,
                                          questions: List[Question],
                                          tournament_id: int,
                                          dry_run: bool) -> List[ProcessingResult]:
        """Process questions sequentially.

        Args:
            questions: Questions to process
            tournament_id: Tournament ID
            dry_run: Whether to run without submitting

        Returns:
            List of processing results
        """
        results = []
        for question in questions:
            result = await self.process_question_use_case.execute(
                question=question,
                tournament_id=tournament_id,
                submission_mode=not dry_run
            )
            results.append(result)
        return results

    def _format_processing_result(self,
                                result: ProcessingResult,
                                format: str) -> Dict[str, Any]:
        """Format processing result for output.

        Args:
            result: Processing result to format
            format: Output format

        Returns:
            Formatted result dictionary
        """
        base_result = {
            'correlation_id': result.correlation_id,
            'question_id': result.question_id,
            'success': result.success,
            'processing_time': result.processing_time
        }

        if format == 'summary':
            base_result.update({
                'final_prediction': (
                    result.final_forecast.prediction
                    if result.final_forecast else None
                ),
                'confidence': (
                    result.final_forecast.confidence.level
                    if result.final_forecast else None
                ),
                'research_sources': len(result.research_reports),
                'ensemble_size': len(result.ensemble_forecasts)
            })
        elif format == 'json':
            base_result.update({
                'final_forecast': (
                    self._serialize_forecast(result.final_forecast)
                    if result.final_forecast else None
                ),
                'research_reports': [
                    self._serialize_research_report(report)
                    for report in result.research_reports
                ],
                'ensemble_forecasts': [
                    self._serialize_forecast(forecast)
                    for forecast in result.ensemble_forecasts
                ],
                'metadata': result.metadata
            })

        if not result.success:
            base_result['error'] = result.error_message

        return base_result

    def _serialize_forecast(self, forecast) -> Dict[str, Any]:
        """Serialize forecast for JSON output."""
        return {
            'question_id': forecast.question_id,
            'prediction': forecast.prediction,
            'confidence': {
                'level': forecast.confidence.level,
                'basis': forecast.confidence.basis
            },
            'agent_id': forecast.agent_id,
            'timestamp': forecast.timestamp.isoformat(),
            'reasoning_steps': len(forecast.reasoning_trace),
            'evidence_sources': len(forecast.evidence_sources),
            'is_final': forecast.is_final
        }

    def _serialize_research_report(self, report) -> Dict[str, Any]:
        """Serialize research report for JSON output."""
        return {
            'question_id': report.question_id,
            'sources_count': len(report.sources),
            'research_quality_score': report.research_quality_score,
            'timestamp': report.timestamp.isoformat()
        }

    def _format_strategy_table(self, strategy_recommendation) -> str:
        """Format strategy recommendation as table."""
        headers = ['Metric', 'Value']
        data = [
            ['Strategy Type', strategy_recommendation.strategy_type.value],
            ['Confidence', f"{strategy_recommendation.confidence.level:.3f}"],
            ['Expected Score Impact', f"{strategy_recommendation.expected_score_impact:.3f}"],
            ['Risk Level', f"{strategy_recommendation.risk_level:.3f}"],
            ['Resource Requirements', str(strategy_recommendation.resource_requirements)],
            ['Top Questions', str(list(strategy_recommendation.question_allocation.keys())[:5])]
        ]

        return tabulate(data, headers=headers, tablefmt='grid')

    def _format_strategy_summary(self, strategy_recommendation) -> Dict[str, Any]:
        """Format strategy recommendation as summary."""
        return {
            'strategy': strategy_recommendation.strategy_type.value,
            'confidence': strategy_recommendation.confidence.level,
            'expected_impact': strategy_recommendation.expected_score_impact,
            'risk': strategy_recommendation.risk_level,
            'reasoning': strategy_recommendation.reasoning[:200] + '...' if len(strategy_recommendation.reasoning) > 200 else strategy_recommendation.reasoning
        }

    async def _export_json(self, results: List[Dict], output_file: str):
        """Export results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    async def _export_csv(self, results: List[Dict], output_file: str):
        """Export results to CSV file."""
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

    async def _export_xlsx(self, results: List[Dict], output_file: str):
        """Export results to Excel file."""
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)


# Click CLI commands

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """Tournament Optimization System CLI."""
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Initialize CLI (in real implementation, this would use dependency injection)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config


@cli.command()
@click.argument('question_id', type=int)
@click.option('--tournament-id', '-t', type=int, help='Tournament ID')
@click.option('--force', '-f', is_flag=True, help='Force reprocessing')
@click.option('--dry-run', '-d', is_flag=True, help='Run without submitting')
@click.option('--format', '-o', default='json', type=click.Choice(['json', 'table', 'summary']), help='Output format')
@click.pass_context
def process_question(ctx, question_id, tournament_id, force, dry_run, format):
    """Process a single question through the forecasting pipeline."""
    async def run():
        # Initialize CLI instance (placeholder)
        cli_instance = TournamentCLI(None, None, None, None, None)
        result = await cli_instance.process_question(
            question_id=question_id,
            tournament_id=tournament_id,
            force_reprocess=force,
            dry_run=dry_run,
            output_format=format
        )

        if format == 'json':
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            click.echo(result)

    asyncio.run(run())


@cli.command()
@click.argument('tournament_id', type=int)
@click.option('--max-questions', '-n', type=int, help='Maximum questions to process')
@click.option('--parallel/--sequential', default=True, help='Processing mode')
@click.option('--dry-run', '-d', is_flag=True, help='Run without submitting')
@click.pass_context
def process_tournament(ctx, tournament_id, max_questions, parallel, dry_run):
    """Process all questions in a tournament."""
    async def run():
        # Initialize CLI instance (placeholder)
        cli_instance = TournamentCLI(None, None, None, None, None)
        result = await cli_instance.process_tournament(
            tournament_id=tournament_id,
            max_questions=max_questions,
            parallel_processing=parallel,
            dry_run=dry_run
        )

        click.echo(json.dumps(result, indent=2, default=str))

    asyncio.run(run())


@cli.command()
@click.argument('tournament_id', type=int)
@click.option('--format', '-o', default='table', type=click.Choice(['json', 'table', 'summary']), help='Output format')
@click.pass_context
def analyze_strategy(ctx, tournament_id, format):
    """Analyze tournament strategy and provide recommendations."""
    async def run():
        # Initialize CLI instance (placeholder)
        cli_instance = TournamentCLI(None, None, None, None, None)
        result = await cli_instance.analyze_tournament_strategy(
            tournament_id=tournament_id,
            output_format=format
        )

        if format == 'table':
            click.echo(result)
        else:
            click.echo(json.dumps(result, indent=2, default=str))

    asyncio.run(run())


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health checks."""
    async def run():
        # Initialize CLI instance (placeholder)
        cli_instance = TournamentCLI(None, None, None, None, None)
        result = await cli_instance.show_system_status()

        click.echo(json.dumps(result, indent=2, default=str))

    asyncio.run(run())


@cli.command()
@click.option('--tournament-id', '-t', type=int, help='Tournament ID to export')
@click.option('--question-ids', '-q', help='Comma-separated question IDs')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'csv', 'xlsx']), help='Export format')
@click.pass_context
def export(ctx, tournament_id, question_ids, output, format):
    """Export processing results to file."""
    async def run():
        question_id_list = None
        if question_ids:
            question_id_list = [int(x.strip()) for x in question_ids.split(',')]

        # Initialize CLI instance (placeholder)
        cli_instance = TournamentCLI(None, None, None, None, None)
        result = await cli_instance.export_results(
            tournament_id=tournament_id,
            question_ids=question_id_list,
            output_file=output,
            format=format
        )

        click.echo(json.dumps(result, indent=2, default=str))

    asyncio.run(run())


if __name__ == '__main__':
    cli()
