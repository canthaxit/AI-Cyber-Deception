# MCP Server for AI Platforms

Model Context Protocol server for Claude Desktop and other AI platforms.

## Files

- **anomaly_mcp_server.py** - MCP server implementation
- **claude_mcp_config.json** - Claude Desktop configuration

## What is MCP?

Model Context Protocol (MCP) allows AI tools like Claude Desktop to access your anomaly detection system as a service, enabling natural language interaction with security logs.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r ../config/requirements_api.txt
pip install mcp
```

### 2. Configure Claude Desktop

**Windows:**
```bash
copy claude_mcp_config.json %APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```bash
cp claude_mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Linux:**
```bash
cp claude_mcp_config.json ~/.config/Claude/claude_desktop_config.json
```

### 3. Restart Claude Desktop

### 4. Use with Claude

Now you can ask Claude:
```
"Load the anomaly detection models and analyze the logs in my tests folder"
"Check auth.json for brute force attacks"
"What high-severity anomalies were detected today?"
"Investigate the security threats from IP 10.0.0.100"
```

## Available MCP Tools

Claude can use these tools:

### 1. load_anomaly_models
Load trained detection models.
```json
{
  "model_dir": "../core/anomaly_outputs"
}
```

### 2. analyze_logs
Analyze log data for threats.
```json
{
  "log_data": "[{\"timestamp\": \"...\", \"user\": \"...\"}]",
  "format": "json"
}
```

### 3. analyze_log_file
Analyze logs from a file path.
```json
{
  "filepath": "../tests/test_logs_attack.json"
}
```

### 4. get_detection_stats
Get model statistics and info.
```json
{}
```

## Manual Testing

Start the MCP server manually:
```bash
python anomaly_mcp_server.py
```

The server communicates via stdio (standard input/output) using JSON-RPC protocol.

## Configuration

### Update Paths

Edit `claude_mcp_config.json`:
```json
{
  "mcpServers": {
    "log-anomaly-detection": {
      "command": "python",
      "args": [
        "C:\\Users\\jimmy\\.local\\bin\\anomaly-detection\\mcp\\anomaly_mcp_server.py"
      ],
      "env": {
        "MODEL_DIR": "C:\\Users\\jimmy\\.local\\bin\\anomaly-detection\\core\\anomaly_outputs"
      }
    }
  }
}
```

**Important**: Use absolute paths!

### Auto-load Models

The server automatically loads models from `MODEL_DIR` on startup if available.

## Integration with Other AI Tools

The MCP server can work with any tool that supports Model Context Protocol:

1. **Claude Desktop** - Natural language security analysis
2. **Custom AI agents** - Via MCP SDK
3. **Automation tools** - Programmatic access

## Example Conversation with Claude

```
User: "Can you analyze the logs in my tests folder?"

Claude: [Uses analyze_log_file tool]
"I've analyzed the logs and found 8 security anomalies:

- 7 brute force attacks from IP 10.0.0.100
  User: admin, multiple failed logins in 35 seconds

- 1 data exfiltration attempt
  User: root accessed /etc/shadow at 2:00 AM

The brute force pattern shows automated credential stuffing.
I recommend blocking IP 10.0.0.100 and investigating the root access."
```

## Troubleshooting

**MCP server not appearing in Claude:**
1. Check config file location
2. Verify Python path is correct
3. Ensure models exist at MODEL_DIR
4. Restart Claude Desktop

**"Models not loaded" error:**
1. Train models first: `cd ../core && python log_anomaly_detection_lite.py`
2. Check MODEL_DIR path in config
3. Verify `anomaly_outputs/` directory exists

**Permission errors:**
1. Ensure Python is in PATH
2. Check file permissions
3. Run Claude Desktop with appropriate permissions

## Documentation

- **MCP Integration**: [`../docs/README_SCALING.md#mcp-integration`](../docs/README_SCALING.md)
- **Integration Guide**: [`../docs/README_INTEGRATIONS.md`](../docs/README_INTEGRATIONS.md)
- **MCP Protocol**: https://modelcontextprotocol.io/

## Advanced Usage

### Custom MCP Client

```python
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client("python", ["anomaly_mcp_server.py"]) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # List available tools
            tools = await session.list_tools()

            # Call tool
            result = await session.call_tool(
                "analyze_logs",
                {"log_data": "[...]", "format": "json"}
            )

            print(result)

asyncio.run(main())
```

### Multiple Models

You can configure multiple detection systems:
```json
{
  "mcpServers": {
    "anomaly-production": {
      "command": "python",
      "args": ["path/to/production/anomaly_mcp_server.py"],
      "env": {"MODEL_DIR": "path/to/prod/models"}
    },
    "anomaly-staging": {
      "command": "python",
      "args": ["path/to/staging/anomaly_mcp_server.py"],
      "env": {"MODEL_DIR": "path/to/staging/models"}
    }
  }
}
```

## Security

- MCP servers run locally with your user permissions
- No data is sent to external servers
- Models and logs stay on your machine
- Use file permissions to restrict access
