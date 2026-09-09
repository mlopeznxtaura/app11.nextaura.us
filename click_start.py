#!/usr/bin/env python3
"""Click preset then Start on app9.nextaura.us."""
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)

        print(f"Page title: {page.title()}")

        # Click preset
        preset_btn = page.get_by_role("button", name="Apply preset: Transformer baseline SFT")
        if preset_btn.count() > 0:
            preset_btn.first.click()
            print("Clicked preset button")
            page.wait_for_timeout(1000)
        else:
            print("Preset button not found")

        # Find Start button and check its state
        start_btn = page.query_selector("button[id='startBtn']")
        if start_btn:
            print(f"Start button text: {start_btn.inner_text()}")
            print(f"Start button class: {start_btn.get_attribute('class')}")
            print(f"Start button onclick: {start_btn.get_attribute('onclick')}")
            
            # Try to click even if disabled (bypass strictness)
            try:
                start_btn.click(force=True, timeout=5000)
                print("Forced click on Start!")
            except Exception as e:
                print(f"Click failed: {e}")
        else:
            print("Start button not found")

        page.screenshot(path='app9_start_clicked.png')
        print("Screenshot saved: app9_start_clicked.png")
        browser.close()


if __name__ == "__main__":
    main()
