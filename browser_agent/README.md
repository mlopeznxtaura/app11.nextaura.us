# Browser Agents for app9.nextaura.us

Integrated multi-backend browser automation combining three specialized agents.

## Architecture

| Backend | Purpose | Privacy |
|---------|---------|---------|
| **fouwser** | MCP server access for external agents | Local + cloud |
| **ondevice** | Fully local WebLLM/WebGPU agent | 100% on-device |
| **lmnr** | Vision-based reasoning for complex tasks | Cloud (optional) |

## Features

### Unified Agent
- **Single API** - Call `UnifiedBrowserAgent` regardless of backend
- **Auto-routing** - Intelligent backend selection based on task type
- **MCP server** - Expose all capabilities via MCP

### Task Routing
| Task Type | Backend |
|-----------|---------|
| Private/local tasks | ondevice |
| Vision/screenshot analysis | lmnr |
| MCP/CLI automation | fouwser |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables:

```bash
# Default backend
export BROWSER_DEFAULT=lmnr

# Fouwser (MCP server)
export FOUWSER_ENABLED=true
export FOUWSER_PORT=4000

# On-device (private)
export ONDEVICE_ENABLED=true
export ONDEVICE_WEBGPU=true
export ONDEVICE_OFFLINE=true

# LMNR (vision-based)
export LMNR_ENABLED=true
export LMNR_MODEL=gemini-2.5-pro
export LMNR_API_KEY=your_key
```

## Usage

### Unified Agent
```python
from browser_agent import UnifiedBrowserAgent, BrowserConfig

# Load config from env
config = BrowserConfig.from_env()
agent = UnifiedBrowserAgent(config)
await agent.initialize()

# Navigate (auto-selects backend)
result = await agent.navigate("https://example.com")

# Execute task (vision-based)
result = await agent.execute_task("Find pricing info and summarize")
```

### MCP Server
```python
from browser_agent.integrator import MCPUnifiedServer

server = MCPUnifiedServer(agent)
# Connect to MCP client (e.g., claude-code, gemini-cli)
```

### Direct Backend Access
```python
# Use on-device only (privacy)
result = await agent.execute_task("private task", backend="ondevice")

# Use lmnr for vision
result = await agent.execute_task("analyze screenshot", backend="lmnr")
```

## MCP Tools

Available via `MCPUnifiedServer`:
- `browser_navigate` - Navigate to URL
- `browser_task` - Execute task with reasoning
- `browser_click` - Click element
- `browser_status` - Check backend health

## Privacy

| Backend | API Keys | Local Processing | Offline |
|---------|----------|------------------|---------|
| fouwser | Optional | Partial | No |
| ondevice | No | 100% | Yes |
| lmnr | Required | No | No |

## License

See individual backend repositories:
- [fouwser](https://github.com/alphanome-ai/Fouwser)
- [on-device-browser-agent](https://github.com/RunanywhereAI/on-device-browser-agent)
- [lmnr-ai/index](https://github.com/lmnr-ai/index)
