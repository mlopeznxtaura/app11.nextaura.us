# On-Device Browser Agent

Privacy-focused browser automation using WebLLM and WebGPU.

## Features
- Runs models locally on device (no API keys needed)
- WebGPU accelerated inference
- WebLLM integration for on-device LLM
- Multi-agent system (Planner + Navigator)
- Offline support after model download (~1GB)

## Integration
```bash
# Install local browser agent
pip install on-device-browser-agent
```

## Config
```json
{
  "ondevice": {
    "model_path": "./models/ondevice-llm",
    "webgpu": true,
    "agents": ["planner", "navigator"]
  }
}
```

## Privacy
- All processing happens on-device
- No cloud API calls
- No data sent externally
