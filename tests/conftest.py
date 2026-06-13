"""
conftest.py — Pytest fixtures for Streamlit app Selenium tests.

Provides:
  - chrome_driver: Chrome WebDriver instance (selenium)
  - streamlit_app: starts the Streamlit server in subprocess, waits for ready
  - base_url: shortcut to the running app URL
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


REPO_DIR = Path(__file__).parent.parent
APP_PATH = REPO_DIR / "app.py"
DEFAULT_PORT = 8511  # use non-default port to avoid conflicts with manual runs


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Poll until the port is open or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def chrome_driver():
    """Chrome WebDriver configured for headless operation."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-web-security")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)
    yield driver
    driver.quit()


@pytest.fixture(scope="session")
def streamlit_app():
    """Start the Streamlit app in subprocess, return (port, process)."""
    port = int(os.environ.get("STREAMLIT_TEST_PORT", _find_free_port()))
    log_path = REPO_DIR / "reports" / ".test_streamlit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["STREAMLIT_THEME_BASE"] = "dark"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_GLOBAL_DISABLE_WIDGET_STATE_PERSISTENCE"] = "true"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_PATH),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO_DIR),
        env=env,
    )

    try:
        if not _wait_for_port(port, timeout=90):
            raise RuntimeError(
                f"Streamlit did not start on port {port}. "
                f"Check log: {log_path}"
            )
        # Give Streamlit a few extra seconds to render initial page
        time.sleep(3)
        yield port, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def app_url(streamlit_app):
    port, _ = streamlit_app
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def short_timeout():
    """Default wait time for elements."""
    return 15
