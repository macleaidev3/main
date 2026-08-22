import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import faulthandler
import datetime
import re
from pathlib import Path
from typing import Tuple, IO

# ==========================================
# FIX: Custom Handler for Windows File Locks
# ==========================================
class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A custom rotating file handler that survives Windows File Lock collisions (WinError 32).
    If a background process is holding the file open, this suppresses the crash,
    re-opens the stream, and tries the rotation again later.
    """
    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            # The background process has the file locked. 
            # Swallow the error. Python's base FileHandler will automatically 
            # reopen the stream on the next log entry, preventing a crash.
            pass


def setup_logging() -> Tuple[IO, logging.Logger]:
    # 1. Ensure Sentinel directory exists on the Desktop
    desktop_dir = Path.home() / "Desktop" / "Sentinel Logs"
    desktop_dir.mkdir(parents=True, exist_ok=True)

    # ACTIVE file must remain static for rollover to work.
    log_path = desktop_dir / "Sentinel_Active_Session.log"
    
    # Dedicated file for faulthandler to prevent internal locking
    crash_log_path = desktop_dir / "Sentinel_Crash_Dumps.log"
    
    now = datetime.datetime.now()
    human_ts = now.strftime("%d/%m/%Y %H:%M:%S")

    # Open the dedicated crash file for faulthandler
    crash_file = open(crash_log_path, "a", buffering=1, encoding="utf-8")
    
    root_logger = logging.getLogger()
    app_logger = logging.getLogger("SentinelApp")

    # Only configure handlers if we haven't done it yet
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        app_logger.setLevel(logging.DEBUG)

        # Mute noisy third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # ==========================================
        # Use our new Safe Handler instead of the default one
        # ==========================================
        fh = SafeTimedRotatingFileHandler(
            filename=log_path,
            when="midnight",    # Keep as "midnight" for production (use "M" for testing)
            interval=1, 
            backupCount=45,
            encoding="utf-8"
        )
        
        # Format the archived files: execution.log.DD_MM_YYYY
        fh.suffix = "%d_%m_%Y"
        # fh.suffix = "%d_%m_%Y_%H-%M"
        fh.extMatch = re.compile(r"^\d{2}_\d{2}_\d{4}$")
        
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fh_formatter)
        root_logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.__stdout__)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fh_formatter)
        root_logger.addHandler(ch)

        # Write header ONLY ONCE on primary boot
        app_logger.info("=== Execution started/continued: %s ===", human_ts)

        # ==========================================
        # Redirect stdout and stderr safely with a Recursion Lock
        # ==========================================
        class _StreamToLogger:
            def __init__(self, log_func, original_stream):
                self._log = log_func
                self._original_stream = original_stream
                self._is_writing = False

            def write(self, buf):
                if not buf:
                    return
                buf = buf.rstrip("\n\r")
                if not buf:
                    return

                # Break the infinite loop if logging triggers an internal error
                if self._is_writing:
                    try:
                        self._original_stream.write(buf + "\n")
                        self._original_stream.flush()
                    except Exception:
                        pass
                    return

                self._is_writing = True
                try:
                    for line in buf.splitlines():
                        self._log(line)
                finally:
                    self._is_writing = False

            def flush(self):
                for handler in root_logger.handlers:
                    handler.flush()

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = _StreamToLogger(root_logger.info, original_stdout)
        sys.stderr = _StreamToLogger(root_logger.error, original_stderr)

        # Exception hooking
        def _excepthook(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        
        sys.excepthook = _excepthook

    # Enable faulthandler on the dedicated crash file
    try:
        faulthandler.enable(file=crash_file)
    except Exception:
        root_logger.exception("Failed to enable faulthandler to the crash log file")

    return crash_file, app_logger