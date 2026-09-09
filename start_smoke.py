#!/usr/bin/env python3
"""Start smoke test on app9.nextaura.us."""
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)

        print("Page title:", page.title())

        # Click a smoke test recipe
        smoke_btn = page.get_by_role("button", name="10M Multimodal smoke")
        if smoke_btn.count() > 0:
            smoke_btn.first.click()
            print("Clicked smoke test button")
            page.wait_for_timeout(1000)
        else:
            print("Smoke button not found")

        # Check if Start is now enabled
        start_btn = page.query_selector("button[id='startBtn']")
        if start_btn:
            class_attr = start_btn.get_attribute('class')
            print("Start button class:", class_attr)
            if "disabled" not in (class_attr or "").lower():
                start_btn.click(force=True, timeout=5000)
                print("Clicked Start!")
            else:
                print("Start still disabled")
        else:
            print("Start button not found")

        page.screenshot(path='app9_smoke_clicked.png')
        print("Screenshot saved")
        browser.close()


if __name__ == "__main__":
    main()
