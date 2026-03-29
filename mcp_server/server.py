"""
MCP Server - Model Context Protocol server for saving research reports.

Exposes tools via MCP that allow LLM agents or IDE integrations to:
  1. Save a markdown report to the local documents folder
  2. List previously generated reports
  3. Read a specific report

This can be used standalone (stdio transport) or integrated into
VS Code / Claude Desktop via MCP configuration.
"""

import os
import re
import json
import asyncio
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from utils.logger import get_logger

logger = get_logger("MCP")


def get_reports_dir(output_dir: str = "./reports") -> Path:
    """Get or create the reports output directory."""
    reports_path = Path(output_dir).resolve()
    reports_path.mkdir(parents=True, exist_ok=True)
    return reports_path


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a topic string."""
    # Remove special characters, keep alphanumeric, spaces, hyphens
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]  # Limit length


def save_report_to_disk(
    report_markdown: str,
    topic: str,
    output_dir: str = "./reports",
) -> str:
    """
    Save a markdown report to the local file system.

    Args:
        report_markdown: The full markdown report content
        topic: Research topic (used for filename)
        output_dir: Directory to save reports

    Returns:
        Absolute path of the saved file
    """
    reports_path = get_reports_dir(output_dir)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = sanitize_filename(topic)
    filename = f"{safe_topic}_{timestamp}.md"

    file_path = reports_path / filename
    file_path.write_text(report_markdown, encoding="utf-8")

    logger.info(f"Report saved: {file_path}")
    return str(file_path)


def list_reports(output_dir: str = "./reports") -> list[dict]:
    """List all saved reports with metadata."""
    reports_path = get_reports_dir(output_dir)
    reports = []

    for f in sorted(reports_path.glob("*.md"), key=os.path.getmtime, reverse=True):
        stat = f.stat()
        reports.append({
            "filename": f.name,
            "path": str(f),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return reports


def create_mcp_server(output_dir: str = "./reports") -> Server:
    """
    Create and configure the MCP server with report management tools.

    Tools exposed:
      - save_report: Save markdown report to disk
      - list_reports: List all saved reports
      - read_report: Read a specific report's content
    """
    server = Server("market-research-aggregator")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="save_report",
                description=(
                    "Save a market research report as a markdown file to the local "
                    "documents folder. Returns the absolute file path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The research topic (used for filename)",
                        },
                        "report_markdown": {
                            "type": "string",
                            "description": "The full markdown report content",
                        },
                    },
                    "required": ["topic", "report_markdown"],
                },
            ),
            Tool(
                name="list_reports",
                description="List all previously generated research reports.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="read_report",
                description="Read the content of a specific report by filename.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The report filename to read",
                        },
                    },
                    "required": ["filename"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "save_report":
            topic = arguments.get("topic", "untitled")
            markdown = arguments.get("report_markdown", "")

            if not markdown:
                return [TextContent(type="text", text="Error: No report content provided")]

            path = save_report_to_disk(markdown, topic, output_dir)
            return [TextContent(type="text", text=f"Report saved to: {path}")]

        elif name == "list_reports":
            reports = list_reports(output_dir)
            if not reports:
                return [TextContent(type="text", text="No reports found.")]
            return [TextContent(type="text", text=json.dumps(reports, indent=2))]

        elif name == "read_report":
            filename = arguments.get("filename", "")
            reports_path = get_reports_dir(output_dir)
            file_path = reports_path / filename

            # Prevent path traversal
            if not file_path.resolve().is_relative_to(reports_path.resolve()):
                return [TextContent(type="text", text="Error: Invalid file path")]

            if not file_path.exists():
                return [TextContent(type="text", text=f"Error: Report not found: {filename}")]

            content = file_path.read_text(encoding="utf-8")
            return [TextContent(type="text", text=content)]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_mcp_server(output_dir: str = "./reports"):
    """Run the MCP server using stdio transport."""
    server = create_mcp_server(output_dir)
    logger.info("Starting MCP server (stdio transport)...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
