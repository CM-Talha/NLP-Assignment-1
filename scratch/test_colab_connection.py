import asyncio
import webbrowser
from contextlib import AsyncExitStack
from fastmcp import Client
from colab_mcp.session import ColabTransport
from colab_mcp.websocket_server import ColabWebSocketServer, COLAB, SCRATCH_PATH

async def main():
    async with AsyncExitStack() as stack:
        wss = await stack.enter_async_context(ColabWebSocketServer())
        url = f"{COLAB}{SCRATCH_PATH}#mcpProxyToken={wss.token}&mcpProxyPort={wss.port}"
        print(f"Colab WebSocket Server running on port {wss.port}")
        print(f"Opening Browser URL: {url}")
        webbrowser.open_new(url)
        
        print("Waiting up to 60s for Colab tab to load and connect...")
        try:
            await asyncio.wait_for(wss.connection_live.wait(), timeout=60.0)
            print(">>> Google Colab connected successfully! <<<")
            
            client = await stack.enter_async_context(Client(ColabTransport(wss)))
            tools = await client.list_tools()
            print("Available Tools from Colab:")
            for t in tools:
                print(f" - {t.name}: {t.description}")
                
            cells_res = await client.call_tool("get_cells", {"includeOutputs": False})
            print("\nInitial Cells in Colab tab:")
            print(cells_res)
            
        except asyncio.TimeoutError:
            print("Timed out waiting for Colab tab to connect.")

if __name__ == "__main__":
    asyncio.run(main())
