# Browser Agent for MCP Integration

This module provides browser automation capabilities integrated with MCP (Model Context Protocol) for agentic workflows.

## Architecture

- **BrowserAgent**: Core agent for navigation and interaction
- **MCPBrowserServer**: MCP server exposing browser tools to external clients

## Setup

```bash
pip install playwright playwright-stealth
playwright install chromium
```

## Usage

### Basic Navigation
```python
from browser_agent import BrowserAgent, MCPBrowserServer
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    agent = BrowserAgent(page, model_client)
    
    result = await agent.navigate("https://example.com")
```

### MCP Integration
```python
server = MCPBrowserServer(agent, port=4000)
await server.start()
```

### Execute Natural Language Tasks
```python
task = "Search for 'weather in San Francisco' and capture the temperature"
result = await agent.execute_task(task)
```

## Tools Available

| Tool | Description |
|------|-------------|
| browser_navigate | Navigate to a URL |
| browser_click | Click an element |
| browser_fill | Fill input field |
| browser_task | Execute complex task with vision-based reasoning |
| browser_screenshot | Capture page screenshot |

## Dependencies

- `playwright` - Browser automation
- `playwright-stealth` - Anti-detection
- LLM client for vision-based reasoning (e.g., Gemini, Claude)
