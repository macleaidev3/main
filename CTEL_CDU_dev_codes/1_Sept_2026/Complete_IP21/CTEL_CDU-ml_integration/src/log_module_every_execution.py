import sys
import os
import glob
import logging
import faulthandler
import datetime
import threading
import time
from pathlib import Path
from typing import Tuple, IO

def _day_change_monitor(logger: logging.Logger):
    current_date = datetime.date.today()
    while True:
        time.sleep(60)
        new_date = datetime.date.today()
        if new_date > current_date:
            logger.info("=== Software continued running on: %s ===", new_date.strftime("%d/%m/%Y"))
            current_date = new_date

def _cleanup_old_logs(log_dir: Path, max_files: int = 50):
    search_pattern = log_dir / "*_execution.log"
    log_files = glob.glob(str(search_pattern))
    log_files.sort(key=os.path.getmtime)
    while len(log_files) > max_files:
        try:
            os.remove(log_files.pop(0))
        except Exception:
            pass

def setup_logging() -> Tuple[IO, logging.Logger]:
    # 1. Setup paths
    desktop_dir = Path.home() / "Desktop" / "Sentinel Logs"
    desktop_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # FIX: Use Environment Variables for Nuitka 
    # ==========================================
    env_log_path = os.environ.get("SENTINEL_LOG_PATH")
    is_main_process = False

    if env_log_path:
        # We are a child process. Inherit the exact file path from the parent.
        log_path = Path(env_log_path)
    else:
        # We are the true Main Process. Generate the file and set the env var.
        is_main_process = True
        now = datetime.datetime.now()
        log_filename = now.strftime("%d_%m_%Y_%H_%M_%S_execution.log")
        log_path = desktop_dir / log_filename
        
        # Save to environment so all child processes spawned after this inherit it
        os.environ["SENTINEL_LOG_PATH"] = str(log_path)
        
        # Only the main process does cleanup
        _cleanup_old_logs(desktop_dir, max_files=45)

    root_logger = logging.getLogger()
    app_logger = logging.getLogger("SentinelApp")

    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        app_logger.setLevel(logging.DEBUG)

        # File Handler
        fh = logging.FileHandler(filename=log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fh_formatter)
        root_logger.addHandler(fh)

        # Console Handler
        ch = logging.StreamHandler(sys.__stdout__)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fh_formatter)
        root_logger.addHandler(ch)

        # Only True Main Process writes the start header
        if is_main_process:
            app_logger.info("=== Execution started: %s ===", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            
            # Start Monitor
            day_monitor_thread = threading.Thread(target=_day_change_monitor, args=(app_logger,), daemon=True)
            day_monitor_thread.start()

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

        def _excepthook(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        
        sys.excepthook = _excepthook

    # Enable faulthandler
    try:
        with open(log_path, "a") as f:
            faulthandler.enable(file=f)
    except Exception:
        pass

    return None, app_logger