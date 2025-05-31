"""Main entry point for the Metaculus forecasting bot."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
import structlog
from rich.console import Console

from infrastructure.config.settings import Config
from pipelines.forecasting_pipeline import ForecastingPipeline
from infrastructure.external_apis.llm_client import LLMClient

# Initialize Typer app
app = typer.Typer(
    name="metaculus-bot-ha",
    help="Production-ready AI forecasting bot for Metaculus",
    rich_markup_mode="rich"
)

console = Console()
logger = structlog.get_logger(__name__)


@app.command()
def forecast(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    tournament_id: Optional[int] = typer.Option(
        None,
        "--tournament",
        "-t",
        help="Metaculus tournament ID to forecast on"
    ),
    max_questions: int = typer.Option(
        10,
        "--max-questions",
        "-n",
        help="Maximum number of questions to forecast"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without submitting predictions to Metaculus"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Run the forecasting bot on tournament questions."""
    
    # Setup basic logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = Config(config_path)
    
    if tournament_id:
        config.metaculus.tournament_id = tournament_id
    
    if dry_run:
        config.bot.publish_reports_to_metaculus = False
        console.print("[yellow]Running in dry-run mode - predictions will not be submitted[/yellow]")
    
    console.print(f"[green]Starting {config.bot.name} v{config.bot.version}[/green]")
    console.print(f"Tournament ID: {config.metaculus.tournament_id}")
    console.print(f"Max questions: {max_questions}")
    
    # Run the forecasting pipeline
    asyncio.run(run_forecasting_pipeline(config, max_questions))


async def run_forecasting_pipeline(config: Config, max_questions: int):
    """Run the main forecasting pipeline."""
    try:
        console.print("[blue]Initializing forecasting pipeline...[/blue]")
        
        # For now, just demonstrate the basic structure
        console.print(f"[green]Configuration loaded successfully![/green]")
        console.print(f"LLM Provider: {config.llm.provider}")
        console.print(f"Search Provider: {config.search.provider}")
        console.print("[yellow]Full pipeline implementation in progress...[/yellow]")
        
    except Exception as e:
        logger.error("Forecasting pipeline failed", error=str(e))
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()