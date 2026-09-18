import logging
from typing import Optional
import os

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from rich.logging import RichHandler


class _CompilerCommandFilter(logging.Filter):
    """Silence distutils/setuptools compiler command echoes at INFO level."""

    _compiler_commands = {
        "ar",
        "cc",
        "c++",
        "gcc",
        "g++",
        "ld",
        "nvcc",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO or record.filename != "spawn.py":
            return True

        message = record.getMessage().strip()
        if not message:
            return True

        executable = os.path.basename(message.split(maxsplit=1)[0])
        return executable not in self._compiler_commands


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: Optional[bool] = None,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_hydra_handlers: bool = True,
    use_rich_handler: bool = True,
) -> None:
    """
    Configure the logging system for the entire codebase with Rich formatting.
    
    In distributed training, only the main process outputs logs while other processes are silenced.
    This function configures the root logger so all child loggers inherit the same configuration.
    
    Args:
        log_level: Logging level (default INFO), only applies to main process
        is_main_process: Whether this is the main process. If None, infer automatically.
        rich_handler_kwargs: Additional kwargs to pass to RichHandler
        formatter_kwargs: Additional kwargs to pass to Formatter (fmt and datefmt)
        preserve_hydra_handlers: Keep existing FileHandlers from Hydra (default True)
        use_rich_handler: Use RichHandler for formatted console logs. Set to False
            for plain one-line console records.

    Example:
        ```python
        # In a single-machine script
        from fastwam.utils.logging_config import setup_logging
        setup_logging()
        
        # In a distributed training script
        from accelerate import PartialState
        from fastwam.utils.logging_config import setup_logging
        
        distributed_state = PartialState()
        setup_logging(
            log_level=logging.INFO,
            is_main_process=distributed_state.is_main_process
        )
        ```
    """
    if is_main_process is None:
        is_main_process = _is_main_process()

    root_logger = logging.getLogger()
    
    if is_main_process:
        # Save existing FileHandlers (e.g., from Hydra) if requested
        existing_file_handlers = []
        if preserve_hydra_handlers:
            existing_file_handlers = [
                h for h in root_logger.handlers 
                if isinstance(h, logging.FileHandler)
            ]
        
        # Clear all default handlers on the root logger
        root_logger.handlers.clear()
        
        # Configure RichHandler parameters
        default_rich_kwargs = {
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
        if rich_handler_kwargs:
            default_rich_kwargs.update(rich_handler_kwargs)
        
        # Create console handler
        if use_rich_handler:
            console_handler = RichHandler(**default_rich_kwargs)
        else:
            console_handler = logging.StreamHandler()
        console_handler.addFilter(_CompilerCommandFilter())
        
        # Configure Formatter
        default_formatter_kwargs = {
            "fmt": "| >> %(message)s",
            "datefmt": "%m/%d [%H:%M:%S]",
        }
        if formatter_kwargs:
            default_formatter_kwargs.update(formatter_kwargs)
        
        formatter = logging.Formatter(**default_formatter_kwargs)
        console_handler.setFormatter(formatter)
        
        # Add handler and set logging level
        root_logger.addHandler(console_handler)

        # Restore existing FileHandlers from Hydra
        for handler in existing_file_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(log_level)
        
    else:
        # In non-main processes, set root logger level to ERROR to silence all logs
        root_logger.setLevel(logging.ERROR)


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # if not logger.handlers and _is_main_process():
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter(
    #         fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger
