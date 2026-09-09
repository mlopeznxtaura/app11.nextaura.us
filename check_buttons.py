#!/usr/bin/env python3
"""Check button states on app9.nextaura.us."""
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)

        buttons = page.query_selector_all('button')
        for btn in buttons[:20]:
            try:
                text = btn.inner_text()
                class_attr = btn.get_attribute('class')
                disabled = 'disabled' in (class_attr or '').lower()
                status = "OK" if not disabled else "DISABLED"
                print(f"[{status}] {text[:50]}")
            except:
                pass

        browser.close()


if __name__ == "__main__":
    main()
