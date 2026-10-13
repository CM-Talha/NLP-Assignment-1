import asyncio
import sys
import json
import os
import subprocess
from contextlib import AsyncExitStack
from fastmcp import Client
from colab_mcp.session import ColabTransport
from colab_mcp.websocket_server import ColabWebSocketServer, COLAB, SCRATCH_PATH

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

async def run_colab_workflow():
    print("=" * 70, flush=True)
    print("GOOGLE COLAB MCP WORKFLOW RUNNER", flush=True)
    print("=" * 70, flush=True)

    async with AsyncExitStack() as stack:
        wss = await stack.enter_async_context(ColabWebSocketServer())
        token = wss.token
        port = wss.port
        colab_url = f"{COLAB}{SCRATCH_PATH}#mcpProxyToken={token}&mcpProxyPort={port}"
        
        print(f"\n[1] Colab WebSocket Bridge Server listening on port: {port}", flush=True)
        print(f"[2] Token: {token}", flush=True)
        print(f"[3] Launching Google Colab session URL directly in Chrome:\n    {colab_url}\n", flush=True)
        
        # Launch Chrome directly with quoted URL to prevent Windows '&' stripping
        if os.path.exists(CHROME_PATH):
            subprocess.Popen([CHROME_PATH, colab_url])
        else:
            import webbrowser
            webbrowser.open_new(colab_url)
        
        print("Waiting up to 180 seconds for Google Colab UI to connect via WebSocket...", flush=True)
        print("NOTE: Please switch to the opened Chrome tab and ensure the page has loaded.", flush=True)
        
        try:
            await asyncio.wait_for(wss.connection_live.wait(), timeout=180.0)
            print("\n" + "=" * 70, flush=True)
            print(">>> [SUCCESS] Google Colab connected via Colab MCP WebSocket! <<<", flush=True)
            print("=" * 70 + "\n", flush=True)
        except asyncio.TimeoutError:
            print("\n[ERROR] Connection timed out after 180s.", flush=True)
            print("Troubleshooting:", flush=True)
            print(" 1. Ensure Chrome is logged in to your Google Account.", flush=True)
            print(f" 2. Manually open this URL in Chrome:\n    {colab_url}", flush=True)
            print(" 3. In Colab, press Ctrl+Shift+P -> 'Connect to MCP server' if prompted.", flush=True)
            return

        client = await stack.enter_async_context(Client(ColabTransport(wss)))
        
        # 1. Discover tools
        tools = await client.list_tools()
        print(f"[Phase 1] Discovered {len(tools)} Colab MCP tools:", flush=True)
        for t in tools:
            print(f"  • {t.name}: {t.description}", flush=True)
            
        # 2. Inspect active cells
        print("\n[Phase 2] Inspecting Active Colab Notebook Cells...", flush=True)
        cells_result = await client.call_tool("get_cells", {"includeOutputs": True})
        
        content_text = ""
        for c in cells_result:
            if hasattr(c, "text"):
                content_text += c.text
            elif isinstance(c, dict) and "text" in c:
                content_text += c["text"]
            else:
                content_text += str(c)
                
        try:
            cells_data = json.loads(content_text)
            print(f"Successfully retrieved {len(cells_data)} existing cells in Colab session.", flush=True)
        except Exception:
            print("Raw get_cells response snippet:", content_text[:300], flush=True)
            cells_data = []

        # 3. Environment & GPU diagnostics
        print("\n[Phase 3] Executing Environment Diagnostics & GPU Verification...", flush=True)
        env_code = (
            "!nvidia-smi\n"
            "import sys, os, torch\n"
            "print('Python Version:      ', sys.version.split()[0])\n"
            "print('PyTorch Version:     ', torch.__version__)\n"
            "print('CUDA Accelerator:    ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No GPU)')\n"
            "if torch.cuda.is_available():\n"
            "    print('VRAM Allocated:      ', round(torch.cuda.memory_allocated(0)/(1024**2), 2), 'MB')\n"
            "    print('VRAM Reserved:       ', round(torch.cuda.memory_reserved(0)/(1024**2), 2), 'MB')\n"
        )
        
        first_cell_id = cells_data[0].get("id") if (cells_data and len(cells_data) > 0) else None
        if first_cell_id:
            print(f"Updating cell '{first_cell_id}' with diagnostics code...", flush=True)
            await client.call_tool("update_cell", {"cellId": first_cell_id, "content": env_code})
            print("Executing diagnostics cell...", flush=True)
            run_res = await client.call_tool("run_code_cell", {"cellId": first_cell_id})
            print("--- Execution Output ---", flush=True)
            print(run_res, flush=True)
        else:
            print("Adding diagnostic cell...", flush=True)
            add_res = await client.call_tool("add_code_cell", {"content": env_code})
            print("Add result:", add_res, flush=True)

        print("\n[Phase 4] Colab MCP step-by-step connection verified successfully!", flush=True)

if __name__ == "__main__":
    asyncio.run(run_colab_workflow())
