# lmnr-ai/index Browser Agent

SOTA open-source browser agent with vision-based reasoning.

## Features
- Vision-based reasoning LLMs (e.g., Gemini 2.5 Pro)
- Autonomous task execution
- Code-based Agent class for programmatic use
- CLI tool available
- Built-in observability for action tracking

## Installation
```bash
pip install index-agent
```

## Usage
### Python API
```python
from index import Agent

agent = Agent(api_key="your-llm-api-key")
result = agent.run("Find the latest news about AI and summarize")
```

### CLI
```bash
index "Find the latest news about AI and summarize"
```

## Observability
Actions are logged and tracked:
```python
agent.trace("find_news")  # Track specific actions
```
