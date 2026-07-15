import sys
import os
from loguru import logger

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    level = "DEBUG" if debug else "INFO"

    logger.remove()  # drop default stderr handler (unstyled, wrong format)

    # ── Terminal ──────────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<dim>|</dim> <level>{level:<7}</level> "
            "<dim>|</dim> <cyan>{name}</cyan> "
            "<dim>></dim> {message}"
        ),
    )

    # ── File (full detail) ────────────────────────────────────────────────────
    from config.settings import LOG_DIR
    from pathlib import Path
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    logger.add(
        str(Path(LOG_DIR) / "2plus.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} > {message}",
        rotation="10 MB",
        retention=3,
        enqueue=True,
    )
