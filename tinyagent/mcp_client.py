import asyncio
import logging
from typing import Optional, List, Callable, Dict, Any, Union, Tuple
from contextlib import AsyncExitStack
from tinyagent.mcp_client import StdioServerParameters, stdio_client, ClientSession

class MCPClient:
    """
    Async MCPClient that can manage one or multiple MCP server connections.
    Supports callbacks on tool events and async context management.
    """
    def __init__(self,
                 server_parameters: Union[
                     dict, Tuple[str, List[str]], List[Union[dict, Tuple[str, List[str]]]]
                 ],
                 logger: Optional[logging.Logger] = None):
        # Normalize server parameters into a list of dicts
        if not isinstance(server_parameters, list):
            server_parameters = [server_parameters]
        params_list: List[dict] = []
        for p in server_parameters:
            if isinstance(p, tuple):
                cmd, args = p
                params_list.append({"command": cmd, "args": args})
            elif isinstance(p, dict):
                params_list.append(p)
            else:
                raise ValueError(f"Invalid server parameter type: {type(p)}")
        self._server_params = params_list
        self.exit_stack = AsyncExitStack()
        self.logger = logger or logging.getLogger(__name__)
        self.callbacks: List[Callable] = []
        self.sessions: List[ClientSession] = []
        self.tool_map: Dict[str, ClientSession] = {}
        self.logger.debug("MCPClient initialized with %d server(s)", len(self._server_params))

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def connect(self) -> None:
        """
        Connect to all configured MCP servers and initialize sessions.
        Builds internal tool-to-session map.
        """
        # Start each MCP server session
        for params in self._server_params:
            sp = StdioServerParameters(**params)
            stdio, sock_write = await self.exit_stack.enter_async_context(
                stdio_client(sp)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, sock_write)
            )
            await session.initialize()
            self.sessions.append(session)
        # Build tool map for dispatching
        await self._build_tool_map()

    async def _build_tool_map(self):
        self.tool_map = {}
        for session in self.sessions:
            resp = await session.list_tools()
            for tool in resp.tools:
                self.tool_map[tool.name] = session

    async def list_tools(self) -> List:
        """
        List all available tools across all MCP sessions.
        """
        all_tools = []
        for session in self.sessions:
            resp = await session.list_tools()
            for tool in resp.tools:
                all_tools.append(tool)
        return all_tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Call a named tool on the appropriate MCP session.
        """
        await self._run_callbacks("tool_start", tool_name=name, arguments=arguments)
        session = self.tool_map.get(name)
        if session is None:
            err = f"No MCP session registered for tool '{name}'"
            await self._run_callbacks("tool_end", tool_name=name,
                                      arguments=arguments, error=err, success=False)
            raise ValueError(err)
        try:
            resp = await session.call_tool(name, arguments)
            result = resp.content
            await self._run_callbacks("tool_end", tool_name=name,
                                      arguments=arguments, result=result, success=True)
            return result
        except Exception as e:
            await self._run_callbacks("tool_end", tool_name=name,
                                      arguments=arguments, error=str(e), success=False)
            raise

    def register_callback(self, callback: Callable):
        self.callbacks.append(callback)

    async def _run_callbacks(self, event: str, **kwargs):
        for cb in self.callbacks:
            await cb(event, **kwargs)

    async def close(self):
        # Close all sessions and cleanup
        try:
            await self.exit_stack.aclose()
        except (RuntimeError, asyncio.CancelledError) as e:
            self.logger.error(f"Error during client cleanup: {e}")
        finally:
            self.sessions = []
            self.tool_map = {}
            self.exit_stack = AsyncExitStack()
