"""
Structured logging for the Market Research Aggregator.
Uses Rich for beautiful, colored console output with agent-level tagging.
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

AGENT_COLORS = {
    "PLANNER": "cyan",
    "SEARCHER": "yellow",
    "WRITER": "green",
    "RAG": "magenta",
    "ORCHESTRATOR": "blue",
    "MCP": "red",
    "MAIN": "white",
}


def get_logger(agent_name: str = "MAIN") -> logging.Logger:
    """
    Create a logger tagged with the agent name.
    Each agent gets its own colored prefix in terminal output.
    """
    logger = logging.getLogger(f"mra.{agent_name}")

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            markup=True,
        )
        color = AGENT_COLORS.get(agent_name.upper(), "white")
        handler.setFormatter(
            logging.Formatter(f"[{color}][%(name)s][/{color}] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def log_step(agent_name: str, step: str, detail: str = ""):
    """Log a pipeline step with visual formatting."""
    color = AGENT_COLORS.get(agent_name.upper(), "white")
    msg = f"[bold {color}]▶ {step}[/bold {color}]"
    if detail:
        msg += f"  {detail}"
    console.print(msg)
