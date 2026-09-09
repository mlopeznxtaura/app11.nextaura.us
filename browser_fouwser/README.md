# Fouwser Browser Agent

Agentic Chromium browser that integrates MCP server for external control.

## Features
- Forked Chromium with built-in AI agent capabilities
- MCP server interface for tools/agents (Claude-Code, Gemini-CLI)
- Native Chrome interface maintained

## Integration
The MCP server exposes browser control capabilities:
- Navigation and browsing
- DOM inspection and interaction
- Screenshot capture
- Tab management

## Install
```bash
pip install fouwser-mcp
```

## MCP Configuration
```json
{
  "fouwser": {
    "mcp_server": "python",
    "command": "fouwser-mcp-server"
  }
}
```
