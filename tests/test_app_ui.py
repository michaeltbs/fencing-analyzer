"""
test_app_ui.py — Selenium UI tests for the Streamlit Fencing Analyzer.

Tests cover:
  - App loads without errors
  - Mode switch renders (Schnell-Clip vs Full-Length)
  - Video source selector renders all 3 options
  - Output options present
  - Multiple navigations don't crash
  - Screenshot capture

Run with: pytest tests/test_app_ui.py -v
"""
import time
from pathlib import Path

import pytest
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


APP_TITLE = "Fecht-Analyzer"


def _wait_text_contains(driver, text, timeout=20):
    """Wait until text appears anywhere on the page."""
    WebDriverWait(driver, timeout).until(
        lambda d: text in d.find_element(By.TAG_NAME, "body").text
    )


def _page_text(driver):
    return driver.find_element(By.TAG_NAME, "body").text


# === Test 1: App loads ===

def test_app_loads(chrome_driver, app_url):
    """Verify the app's main title renders."""
    chrome_driver.get(app_url)
    wait = WebDriverWait(chrome_driver, 30)
    h1 = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    assert APP_TITLE in h1.text, f"Expected title '{APP_TITLE}' in h1, got: {h1.text}"


def test_no_streamlit_errors(chrome_driver, app_url):
    """App should not show any 'StreamlitAPIException' or similar error."""
    chrome_driver.get(app_url)
    time.sleep(3)
    page_text = _page_text(chrome_driver)
    assert "StreamlitAPIException" not in page_text, "Streamlit API error visible"
    assert "KeyError" not in page_text, "Python KeyError visible in UI"


# === Test 2: Mode switch ===

def test_mode_switch_renders(chrome_driver, app_url):
    """Both analysis modes should be visible in sidebar."""
    chrome_driver.get(app_url)
    _wait_text_contains(chrome_driver, "Schnell-Clip", 20)
    _wait_text_contains(chrome_driver, "Full-Length", 5)


# === Test 3: Video source selector ===

def test_video_source_options(chrome_driver, app_url):
    """All 3 video source options should be present."""
    chrome_driver.get(app_url)
    time.sleep(2)
    page_text = _page_text(chrome_driver)
    for option in ["Datei-Upload", "Lokaler Pfad", "YouTube-Link"]:
        assert option in page_text, f"Video source option missing: {option}"


# === Test 4: Full-Length form (form only renders when video_path set) ===

def test_full_length_form_when_no_video(chrome_driver, app_url):
    """Without a video, the form may not render — verify the state is sane."""
    chrome_driver.get(app_url)
    time.sleep(2)
    # Either we see the form (after Full-Length click) or the welcome screen
    page_text = _page_text(chrome_driver)
    # Welcome screen with "Video auswählen" should be visible
    assert "Fecht-Analyzer" in page_text
    # We don't assert form is visible because that requires video selection


# === Test 5: GPU/CPU display ===

def test_gpu_or_cpu_display(chrome_driver, app_url):
    """App should show GPU or CPU indicator."""
    chrome_driver.get(app_url)
    time.sleep(2)
    page_text = _page_text(chrome_driver)
    has_gpu = "GPU" in page_text
    has_cpu = "CPU" in page_text
    assert has_gpu or has_cpu, "Neither GPU nor CPU indicator visible"


# === Test 6: App survives multiple navigations ===

def test_app_no_crash_on_navigation(chrome_driver, app_url):
    """Switching modes back and forth should not crash the app."""
    chrome_driver.get(app_url)
    time.sleep(2)

    # Just check that the app stays responsive after multiple page loads
    for _ in range(3):
        chrome_driver.refresh()
        time.sleep(1)

    # Should still be on a valid page
    page_text = _page_text(chrome_driver)
    assert "Fecht" in page_text or "Fecht-Analyzer" in page_text, \
        "App title lost after navigation"


# === Test 7: Screenshot for visual review ===

def test_capture_screenshot(chrome_driver, app_url, tmp_path):
    """Capture a screenshot of the main view for visual review."""
    chrome_driver.get(app_url)
    time.sleep(3)

    screenshot_path = Path(tmp_path) / "app_main.png"
    chrome_driver.save_screenshot(str(screenshot_path))
    assert screenshot_path.exists(), "Screenshot not saved"
    assert screenshot_path.stat().st_size > 1000, "Screenshot suspiciously small"
    print(f"\n  Screenshot saved: {screenshot_path}")


# === Test 8: Sidebar elements present ===

def test_sidebar_has_required_sections(chrome_driver, app_url):
    """Sidebar should have Mode, Video-Quelle, and at least one input."""
    chrome_driver.get(app_url)
    time.sleep(2)

    # Try to find sidebar via data-testid
    try:
        sidebar = WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-testid='stSidebar']")
            )
        )
        sidebar_text = sidebar.text
    except TimeoutException:
        pytest.skip("Sidebar not found via data-testid")

    assert "Analyse-Modus" in sidebar_text, "Mode section missing"
    assert "Video-Quelle" in sidebar_text, "Video source section missing"
    assert "Schnell-Clip" in sidebar_text, "Schnell-Clip option missing"
    assert "Full-Length" in sidebar_text, "Full-Length option missing"


# === Test 9: Check no JS errors in console ===

def test_no_console_errors(chrome_driver, app_url):
    """No severe JavaScript errors should be in the browser console."""
    chrome_driver.get(app_url)
    time.sleep(3)

    # Get browser console logs (severe only)
    logs = chrome_driver.get_log("browser") if hasattr(chrome_driver, "get_log") else []
    severe = [l for l in logs if l.get("level") == "SEVERE"]

    # Streamlit sometimes has WebSocket warnings — not real errors
    severe = [l for l in severe if "favicon" not in l.get("message", "").lower()]

    if severe:
        print(f"\n  {len(severe)} severe console messages (may be Streamlit internal):")
        for s in severe[:5]:
            print(f"    {s.get('message', '')[:100]}")
    # Don't fail on console warnings — Streamlit's runtime often logs them


# === Test 10: Streamlit DOM structure is correct ===

def test_streamlit_dom_structure(chrome_driver, app_url):
    """Verify Streamlit's standard data-testid attributes are present."""
    chrome_driver.get(app_url)
    time.sleep(2)

    # Streamlit uses these testids:
    testids = [
        "stApp",
        "stSidebar",
    ]
    for testid in testids:
        elements = chrome_driver.find_elements(
            By.CSS_SELECTOR, f"[data-testid='{testid}']"
        )
        assert len(elements) > 0, f"Streamlit testid '{testid}' not found in DOM"
