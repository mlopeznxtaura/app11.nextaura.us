#!/usr/bin/env python3
"""Inspect app9.nextaura.us UI."""
import sys
from playwright.sync_api import sync_playwright

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://app9.nextaura.us', timeout=30000)
        
        print("=" * 60)
        print(f"TITLE: {page.title()}")
        print(f"URL: {page.url}")
        print("=" * 60)
        
        # Buttons
        buttons = page.query_selector_all('button')
        print(f"\nButtons ({len(buttons)} found):")
        for i, btn in enumerate(buttons):
            try:
                text = btn.inner_text()
                if text.strip():
                    print(f"  {i+1}. {text}")
            except:
                pass
        
        # Form Labels
        labels = page.query_selector_all('label')
        print(f"\nForm Labels ({len(labels)} found):")
        for i, label in enumerate(labels[:15]): # Show first 15
            try:
                text = label.inner_text()
                if text.strip():
                    print(f"  {i+1}. {text}")
            except:
                pass
        
        # Inputs (names/types)
        inputs = page.query_selector_all('input')
        print(f"\nInputs ({len(inputs)} found):")
        for i, inp in enumerate(inputs[:10]):
            try:
                id = inp.get_attribute('id') or 'N/A'
                name = inp.get_attribute('name') or 'N/A'
                p = inp.get_attribute('placeholder') or ''
                print(f"  {i+1}. id='{id}' name='{name}' placeholder='{p}'")
            except:
                pass
        
        # Screenshot
        page.screenshot(path='app9_inspect.png')
        print("\n\nScreenshot saved: app9_inspect.png")
        
        browser.close()

if __name__ == "__main__":
    main()
