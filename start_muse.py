#!/usr/bin/env python3
"""Start Muse-Glimmer SFT via UI interaction on app9.nextaura.us."""
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)

        print("Page title:", page.title())

        # Click the Muse-Glimmer preset
        muse_btn = page.get_by_role("button", name="Muse-Glimmer-30B QLoRA SFT v2")
        if muse_btn.count() > 0:
            muse_btn.first.click()
            print("Clicked Muse-Glimmer preset")
            page.wait_for_timeout(1000)
        else:
            print("Muse-Glimmer button not found")

        # Check the config that was applied
        import requests
        try:
            r = requests.get('https://app9.nextaura.us/api/status', timeout=5)
            d = r.json()
            print("Model backend:", d.get('model_backend'))
            print("Tokenizer:", d.get('tokenizer_id'))
            print("HF dataset:", d.get('hf_dataset_id')[:50] if d.get('hf_dataset_id') else "N/A")
            print("Data source:", d.get('data_source'))
            print("Target GB:", d.get('target_gb'))
        except Exception as e:
            print("Status check failed:", e)

        # Try to click Start if enabled
        start_btn = page.query_selector("button[id='startBtn']")
        if start_btn:
            class_attr = start_btn.get_attribute('class') or ""
            if "disabled" not in class_attr.lower():
                start_btn.click(force=True, timeout=5000)
                print("Clicked Start!")
            else:
                print("Start button is disabled")
        else:
            print("Start button not found")

        browser.close()


if __name__ == "__main__":
    main()
