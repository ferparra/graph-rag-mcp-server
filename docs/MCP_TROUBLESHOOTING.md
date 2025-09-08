# Troubleshooting MCP Servers with FastMCP: Developer Guide

Building Model Context Protocol (MCP) servers with **FastMCP** is powerful, but diagnosing issues can be challenging. This guide covers common problem areas – *Connection errors, Serialization bugs, Concurrency issues, and Performance bottlenecks* – and provides solutions. It applies to MCP servers installed via Astral **UVX** commands, running in remote GitHub or containerized environments, and uses modern, strongly-typed Python (`asyncio`-based). Each section below includes typical symptoms, root causes, debugging tips, and resolution steps with code examples.

## 1. Connection Errors

Connection problems are top priority, as they prevent clients from interacting with your MCP server at all. Whether your server is launched locally (e.g. via `fastmcp run` or UVX), in a container, or on a remote host, use the checklist below to diagnose and fix connectivity issues.

### Common Symptoms

* Clients cannot connect or stay connected.
* Errors like **"Server did not connect"**, **"Not connected"**, **"SSE connection not established"**, or **"Missing session ID"**.
* Astral UVX may fail with *"client closed"* errors.
* HTTP/SSE may return HTTP 400 or 500 errors.

### Checklist – Diagnosing Connection Issues

* **Transport Mismatch:** Ensure you run the server with the expected transport protocol (STDIO vs SSE vs HTTP).
* **Port Binding and Networking:** Bind to `0.0.0.0` in containers, expose ports, and check firewalls.
* **Astral UVX Specifics:** UVX may have platform-specific bugs. Run manually with `fastmcp run` for debugging.
* **Session Handshake:** Always initialize and notify before tool calls. Skipping handshake causes *missing session* errors.
* **Stalled Connections:** Use debug logging (`log_level="DEBUG"`) to identify handshake or startup issues.
* **Dependencies & Env Vars:** Check for missing modules or environment variables in GitHub/containerized setups.

### Example – Async Connectivity Test

```python
import asyncio
from fastmcp import Client

async def ping_server():
    try:
        client = Client("http://localhost:8000/mcp")
        async with client:
            result = await client.call_tool("list_tools", {})
            print("Connected! Tools available:", result)
    except Exception as e:
        print("Connection failed:", e)

asyncio.run(ping_server())
```

---

## 2. Serialization Bugs

Serialization issues occur when converting tool inputs/outputs to JSON. FastMCP uses Pydantic, so unsupported types or invalid data cause errors.

### Common Symptoms

* Tool registration fails with schema errors.
* Exceptions like `TypeError: Object of type X is not JSON serializable`.
* Returned data is `null` or truncated.

### Root Causes

* Unsupported parameter or return types (e.g., custom classes, `set`).
* Non-JSONable return values (file handles, bytes, complex objects).
* Validation errors when using Pydantic models incorrectly.

### Solutions

* **Use Supported Types:** Stick to `int, float, str, bool, List, Dict`, or Pydantic models.
* **Wrap Custom Objects:** Represent them as dicts or models.
* **Avoid Non-JSON Data:** Use base64, artifacts, or references for binary/large data.
* **Custom Serializers:** Use `tool_serializer` for special formats (YAML, etc.).
* **Unit Test Serialization:** Test functions directly and run `json.dumps` to validate outputs.

### Example – Fixing Unsupported Types

```python
from fastmcp import FastMCP, ToolError
from pydantic import BaseModel
from typing import List, Dict, Any

mcp = FastMCP("SerializationDemo")

class UserModel(BaseModel):
    name: str

@mcp.tool
def good_tool(names: List[str]) -> Dict[str, Any]:
    if not names:
        raise ToolError("No names provided")
    user = UserModel(name=names[0])
    return {"user": user.dict()}
```

---

## 3. Concurrency Issues

Concurrency bugs arise when multiple requests run simultaneously. `asyncio` enables concurrency, but misuse can lead to race conditions or deadlocks.

### Common Problems

* Race conditions on shared state.
* Mixing session context between clients.
* Blocking code freezing the event loop.
* Deadlocks with improper locks.

### Best Practices

* **Use Async for I/O:** Always use async functions for DB/HTTP/file I/O.
* **Avoid Globals:** Or guard with `asyncio.Lock`.
* **Offload Blocking Work:** Use `asyncio.to_thread` for CPU-heavy tasks.
* **Per-Session Context:** Use `ctx: Context` for request-specific state.
* **Test with Concurrency:** Simulate concurrent requests with `asyncio.gather`.

### Example – Safe Concurrency

```python
import asyncio

counter_lock = asyncio.Lock()
counter = 0

@mcp.tool
async def increment() -> int:
    global counter
    async with counter_lock:
        counter += 1
        return counter
```

---

## 4. Performance Bottlenecks

Performance issues cause slow responses or high resource usage. They often result from blocking code, inefficient processing, or network overhead.

### Causes

* Blocking sync code.
* Sequential execution of independent tasks.
* Large payloads without optimization.
* Inefficient data structures or algorithms.

### Solutions

* **Optimize Async Usage:** Replace blocking I/O with async libraries.
* **Parallelize with Gather:** Run independent tasks concurrently.
* **Cache Results:** Use `functools.lru_cache` or in-memory caches.
* **Offload CPU Tasks:** Use `asyncio.to_thread` or optimized libraries (NumPy, C extensions).
* **Monitor Resources:** Log timings, profile code, and benchmark under load.

### Example – Offloading CPU Work

```python
import hashlib, asyncio

@mcp.tool
async def hash_file(path: str) -> str:
    def compute_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    hash_val = await asyncio.to_thread(compute_hash, path)
    return {"sha256": hash_val}
```

---

## Conclusion

This guide outlined the four most common categories of MCP server issues when using FastMCP:

* **Connection errors:** Fix with correct transport, handshake, and environment setup.
* **Serialization bugs:** Stick to JSON-serializable types and use Pydantic models.
* **Concurrency issues:** Use async, isolate context, and offload blocking work.
* **Performance bottlenecks:** Optimize I/O, parallelize, cache, and profile.

By applying these practices, developers can build MCP servers that are robust, scalable, and reliable across environments like Astral UVX, GitHub-hosted servers, and containers.
