#!/usr/bin/env python3
"""
CLI forecast runner script for Metaculus forecasting bot.

This script orchestrates the full forecasting pipeline by:
1. Loading questions from a JSON file using IngestionService
2. Processing each question through the Dispatcher
3. Displaying forecast results with probability and reasoning
4. Supporting optional submission to prediction platforms

Usage:
    python cli/run_forecast.py [questions_file] [--submit] [--limit N] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to path to import modules
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.application.ingestion_service import IngestionService, ValidationLevel
from src.application.dispatcher import Dispatcher, DispatcherConfig
from src.domain.entities.question import Question
from src.domain.entities.forecast import Forecast


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_questions_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load question data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing question data
        
    Returns:
        List of question dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If the file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e}", e.doc, e.pos)
    
    # Handle both list of questions and wrapped format
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        raise ValueError(f"Invalid file format. Expected list of questions or {{'questions': [...]}}")
    
    if not isinstance(questions, list):
        raise ValueError("Questions must be a list")
    
    return questions


def format_forecast_output(question: Question, forecast: Forecast) -> str:
    """
    Format forecast results for display.
    
    Args:
        question: The question that was forecasted
        forecast: The generated forecast
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"Question ID: {question.metaculus_id}")
    lines.append(f"Title: {question.title}")
    
    # Extract probability from forecast using the prediction property we added
    if forecast.prediction is not None:
        probability_percent = forecast.prediction * 100
        lines.append(f"Forecast: {probability_percent:.1f}%")
    elif forecast.final_prediction and forecast.final_prediction.result:
        if forecast.final_prediction.result.binary_probability is not None:
            probability_percent = forecast.final_prediction.result.binary_probability * 100
            lines.append(f"Forecast: {probability_percent:.1f}%")
        elif forecast.final_prediction.result.numeric_value is not None:
            lines.append(f"Forecast: {forecast.final_prediction.result.numeric_value}")
        else:
            lines.append("Forecast: No prediction value available")
    else:
        lines.append("Forecast: No prediction available")
    
    # Extract reasoning from predictions
    if forecast.reasoning_summary:
        lines.append(f"Reasoning: {forecast.reasoning_summary}")
    elif forecast.predictions and forecast.predictions[0].reasoning:
        lines.append(f"Reasoning: {forecast.predictions[0].reasoning}")
    else:
        lines.append("Reasoning: No reasoning provided")
    
    return "\n".join(lines)


def run_forecasting_pipeline(
    questions_file: Path,
    submit: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False
) -> int:
    """
    Run the complete forecasting pipeline.
    
    Args:
        questions_file: Path to JSON file containing questions
        submit: Whether to submit forecasts (placeholder for future implementation)
        limit: Maximum number of questions to process
        verbose: Enable verbose logging
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting forecasting pipeline with file: {questions_file}")
        
        # Step 1: Load questions from file
        logger.info("Loading questions from file...")
        raw_questions = load_questions_from_file(questions_file)
        logger.info(f"Loaded {len(raw_questions)} questions from file")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            raw_questions = raw_questions[:limit]
            logger.info(f"Limited to {len(raw_questions)} questions")
        
        # Step 2: Parse questions using IngestionService
        logger.info("Parsing questions...")
        ingestion_service = IngestionService(validation_level=ValidationLevel.LENIENT)
        questions, ingestion_stats = ingestion_service.parse_questions(raw_questions)
        
        logger.info(f"Successfully parsed {ingestion_stats.successful_parsed}/{ingestion_stats.total_processed} questions")
        if ingestion_stats.failed_parsing > 0:
            logger.warning(f"Failed to parse {ingestion_stats.failed_parsing} questions")
        
        if not questions:
            logger.error("No valid questions to process")
            return 1
        
        # Step 3: Process questions through Dispatcher
        logger.info("Running forecasting dispatcher...")
        
        # Configure dispatcher for offline processing (don't fetch from API)
        config = DispatcherConfig(
            batch_size=len(questions),  # Process all questions in one batch
            validation_level=ValidationLevel.LENIENT,
            enable_dry_run=submit is False  # Dry run unless submitting
        )
        dispatcher = Dispatcher(config=config)
        
        # Since we have pre-loaded questions, we need to modify the dispatcher approach
        # For now, we'll simulate the process by using the forecast service directly
        forecasts = []
        for question in questions:
            try:
                logger.debug(f"Generating forecast for question {question.metaculus_id}: {question.title}")
                forecast = dispatcher.forecast_service.generate_forecast(question)
                forecasts.append(forecast)
                
                # Display result immediately
                print("\n" + "="*80)
                print(format_forecast_output(question, forecast))
                print("="*80)
                
            except Exception as e:
                logger.error(f"Failed to generate forecast for question {question.metaculus_id}: {e}")
                continue
        
        # Step 4: Summary
        print(f"\n\nFORECAST SUMMARY:")
        print(f"Questions processed: {len(questions)}")
        print(f"Forecasts generated: {len(forecasts)}")
        print(f"Success rate: {len(forecasts)/len(questions)*100:.1f}%")
        
        if submit:
            print("\n⚠️  SUBMIT FLAG DETECTED")
            print("Forecast submission is not yet implemented.")
            print("This is a placeholder for future prediction platform integration.")
        
        logger.info("Forecasting pipeline completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            logger.exception("Full traceback:")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI forecast runner for Metaculus forecasting bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/questions.json
  %(prog)s data/questions.json --limit 5 --verbose
  %(prog)s data/questions.json --submit --limit 10
        """
    )
    
    parser.add_argument(
        'questions_file',
        type=Path,
        help='Path to JSON file containing questions to forecast'
    )
    
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit forecasts to prediction platform (placeholder for future implementation)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of questions to process'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Run the forecasting pipeline
    exit_code = run_forecasting_pipeline(
        questions_file=args.questions_file,
        submit=args.submit,
        limit=args.limit,
        verbose=args.verbose
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
