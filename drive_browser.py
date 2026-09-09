#!/usr/bin/env python3
"""Drive app9.nextaura.us GUI using browser agent."""
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)

        print(f"Title: {page.title()}")

        # Get all buttons
        buttons = page.query_selector_all('button')
        print(f"Found {len(buttons)} buttons")

        # Click the Apply preset button
        for btn in buttons:
            try:
                text = btn.inner_text()
                if "Apply preset" in text or "Start" in text:
                    btn.click()
                    print(f"Clicked: {text[:50]}")
                    break
            except:
                continue

        # Get form inputs
        inputs = page.query_selector_all('input')
        print(f"Found {len(inputs)} input fields")

        # Take screenshot
        page.screenshot(path='app9_driven.png')
        print("Screenshot saved: app9_driven.png")

        browser.close()


if __name__ == "__main__":
    main()
