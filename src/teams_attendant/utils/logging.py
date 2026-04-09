"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "teams-attendant.log"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | Path | None = _DEFAULT_LOG_FILE,
) -> None:
    """Configure structured logging for the application.

    Logs are written to both the console (stderr) and a log file.
    The log file always uses JSON format for easy machine parsing.
    Set *log_file* to ``None`` to disable file logging.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    # --- file logger setup ---
    file_logger: logging.Logger | None = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_logger = logging.getLogger("teams_attendant.file")
        file_logger.handlers.clear()
        file_logger.addHandler(file_handler)
        file_logger.setLevel(log_level)
        file_logger.propagate = False

    # Shared processors that run before forking to console/file
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if file_logger is not None:
        # Fork output: render for console AND write JSON to log file
        console_renderer: structlog.types.Processor
        if json_output:
            console_renderer = structlog.processors.JSONRenderer()
        else:
            console_renderer = structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
            )

        def _tee_to_file(
            logger: logging.Logger,
            method_name: str,
            event_dict: dict,
        ) -> dict:
            """Write a JSON copy of every log event to the log file."""
            # Render a JSON line for the file (separate from console output)
            json_line = structlog.processors.JSONRenderer()(
                logger, method_name, dict(event_dict),
            )
            file_logger.info(json_line)  # type: ignore[union-attr]
            return event_dict

        shared_processors.append(_tee_to_file)
        shared_processors.append(console_renderer)
    else:
        if json_output:
            shared_processors.append(structlog.processors.JSONRenderer())
        else:
            shared_processors.append(
                structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
