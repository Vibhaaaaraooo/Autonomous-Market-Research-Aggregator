"""
Market Research Aggregator - CLI Entry Point

Usage:
    python main.py research "Electric Vehicle startups 2026"
    python main.py research "AI in Healthcare" --output ./my-reports
    python main.py list-reports
    python main.py mcp-server
"""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from config import get_settings, validate_api_keys
from orchestrator.pipeline import ResearchPipeline
from mcp_server.server import save_report_to_disk, list_reports, run_mcp_server
from utils.logger import log_step

console = Console()
app = typer.Typer(
    name="market-research-aggregator",
    help="Autonomous AI-powered market research report generator",
    add_completion=False,
)


def show_banner():
    """Display the application banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════════════════╗
║       AUTONOMOUS MARKET RESEARCH AGGREGATOR              ║
║       ─────────────────────────────────────               ║
║       AI-Powered Deep-Dive Research Reports               ║
║                                                           ║
║   Planner → Searcher → RAG → Writer → Report             ║
╚══════════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic (e.g., 'Electric Vehicle startups 2026')"),
    output_dir: str = typer.Option("./reports", "--output", "-o", help="Output directory for reports"),
):
    """
    Generate a comprehensive market research report on any topic.

    This command runs the full pipeline:
    1. Planner Agent decomposes the topic into search queries
    2. Search Agent finds and scrapes relevant articles via Serper.dev
    3. RAG pipeline chunks and embeds the content
    4. Writer Agent synthesizes a structured markdown report
    5. Report is saved to your local file system
    """
    show_banner()

    # Validate configuration
    settings = get_settings()
    missing_keys = validate_api_keys(settings)

    if missing_keys:
        console.print(
            Panel(
                f"[bold red]Missing API keys: {', '.join(missing_keys)}[/bold red]\n\n"
                "Please set them in your .env file.\n"
                "Copy .env.example to .env and fill in your keys:\n\n"
                "  GROQ_API_KEY=your_key    → https://console.groq.com/keys\n"
                "  SERPER_API_KEY=your_key   → https://serper.dev/api-key",
                title="Configuration Error",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    settings.output_dir = output_dir

    console.print(f"\n[bold]Research Topic:[/bold] {topic}")
    console.print(f"[bold]Output Directory:[/bold] {Path(output_dir).resolve()}")
    console.print(f"[bold]LLM Model:[/bold] {settings.groq_model}")
    console.print(f"[bold]Embedding Model:[/bold] {settings.embedding_model}")
    console.print()

    # Run the pipeline
    pipeline = ResearchPipeline(settings)
    result = asyncio.run(pipeline.run(topic))

    if not result.success:
        console.print(
            Panel(
                "[bold red]Pipeline failed to generate a report.[/bold red]\n"
                "This could be due to:\n"
                "- No articles found for the topic\n"
                "- API rate limits\n"
                "- Network issues\n\n"
                "Try a different or broader topic.",
                title="Pipeline Failed",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    # Save report
    log_step("ORCHESTRATOR", "STEP 5/5", "Saving report to disk")
    file_path = save_report_to_disk(result.report_markdown, topic, output_dir)

    # Show results summary
    _show_results(result, file_path)


@app.command(name="list-reports")
def list_reports_cmd(
    output_dir: str = typer.Option("./reports", "--output", "-o", help="Reports directory"),
):
    """List all previously generated research reports."""
    reports = list_reports(output_dir)

    if not reports:
        console.print("[yellow]No reports found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Generated Reports", show_lines=True)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="green")

    for report in reports:
        table.add_row(
            report["filename"],
            f"{report['size_kb']} KB",
            report["modified"],
        )

    console.print(table)


@app.command(name="mcp-server")
def mcp_server_cmd(
    output_dir: str = typer.Option("./reports", "--output", "-o", help="Reports directory"),
):
    """
    Start the MCP server for IDE/tool integration.

    The MCP server exposes tools for saving, listing, and reading reports
    via the Model Context Protocol (stdio transport).

    Configure in VS Code / Claude Desktop:
    {
        "mcpServers": {
            "market-research": {
                "command": "python",
                "args": ["main.py", "mcp-server"]
            }
        }
    }
    """
    console.print("[bold]Starting MCP Server...[/bold]")
    asyncio.run(run_mcp_server(output_dir))


def _show_results(result, file_path: str):
    """Display pipeline results summary."""
    # Step timing table
    steps_table = Table(title="Pipeline Steps", show_lines=True)
    steps_table.add_column("Step", style="cyan")
    steps_table.add_column("Details")
    steps_table.add_column("Time", justify="right", style="green")

    for step in result.steps:
        name = step["step"].replace("_", " ").title()
        details = ", ".join(
            f"{k}: {v}" for k, v in step.items()
            if k not in ("step", "time_seconds")
        )
        steps_table.add_row(name, details, f"{step['time_seconds']}s")

    console.print()
    console.print(steps_table)

    # Final summary panel
    summary = f"""[bold green]✓ Report generated successfully![/bold green]

[bold]Topic:[/bold]         {result.topic}
[bold]Articles:[/bold]      {result.articles_found} sources analyzed
[bold]Chunks:[/bold]        {result.chunks_stored} text chunks in RAG
[bold]Report:[/bold]        {len(result.report_markdown.split()):,} words
[bold]Total Time:[/bold]    {result.total_time_seconds}s

[bold]Saved to:[/bold]      [link={file_path}]{file_path}[/link]"""

    console.print()
    console.print(Panel(summary, title="Research Complete", border_style="green"))


if __name__ == "__main__":
    app()
