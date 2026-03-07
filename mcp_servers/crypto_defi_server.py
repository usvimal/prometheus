#!/usr/bin/env python3
"""
Crypto/DeFi MCP Server - Model Context Protocol server for cryptocurrency analysis.

This server provides tools for:
- Wallet balance checking
- Token price fetching
- DeFi yield analysis
- Gas price monitoring

Usage:
    python crypto_defi_server.py
    
Or via npx (when published):
    npx @prometheus/mcp-crypto-defi

Environment variables:
    ETH_RPC_URL - Ethereum RPC endpoint (optional, defaults to public)
    DEFILLAMA_API_URL - DeFi Llama API base (optional)
"""

import asyncio
import json
import os
from typing import Any, Optional
from datetime import datetime

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool, Resource
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp")
    raise

import httpx

# Server metadata
SERVER_NAME = "prometheus-crypto-defi"
SERVER_VERSION = "1.0.0"

# API endpoints
ETH_RPC_DEFAULT = "https://eth.llamarpc.com"
DEFILLAMA_API = "https://yields.llama.fi"
COINGECKO_API = "https://api.coingecko.com/api/v3"
ETHERSCAN_API = "https://api.etherscan.io/api"


class CryptoDeFiServer:
    """MCP Server for crypto and DeFi analysis tools."""
    
    def __init__(self):
        self.server = Server(SERVER_NAME)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.eth_rpc_url = os.getenv("ETH_RPC_URL", ETH_RPC_DEFAULT)
        
        # Register handlers
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register all available tools."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="get_eth_balance",
                    description="Get the ETH balance of an Ethereum address",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "Ethereum address (0x...)",
                            }
                        },
                        "required": ["address"],
                    },
                ),
                Tool(
                    name="get_token_price",
                    description="Get current price of a cryptocurrency in USD",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "token_id": {
                                "type": "string",
                                "description": "CoinGecko token ID (e.g., 'bitcoin', 'ethereum', 'solana')",
                            }
                        },
                        "required": ["token_id"],
                    },
                ),
                Tool(
                    name="get_defi_yields",
                    description="Get top DeFi yield opportunities from DeFi Llama",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Filter by chain (e.g., 'Ethereum', 'Solana', 'Arbitrum')",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10,
                            }
                        },
                    },
                ),
                Tool(
                    name="get_gas_prices",
                    description="Get current Ethereum gas prices in Gwei",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_market_data",
                    description="Get market data for top cryptocurrencies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of top cryptocurrencies (default: 10)",
                                "default": 10,
                            }
                        },
                    },
                ),
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "get_eth_balance":
                result = await self._get_eth_balance(arguments["address"])
            elif name == "get_token_price":
                result = await self._get_token_price(arguments["token_id"])
            elif name == "get_defi_yields":
                chain = arguments.get("chain")
                limit = arguments.get("limit", 10)
                result = await self._get_defi_yields(chain, limit)
            elif name == "get_gas_prices":
                result = await self._get_gas_prices()
            elif name == "get_market_data":
                limit = arguments.get("limit", 10)
                result = await self._get_market_data(limit)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    def _register_resources(self):
        """Register available resources."""
        
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="crypto://market/overview",
                    name="Crypto Market Overview",
                    mimeType="application/json",
                    description="Current cryptocurrency market overview with top coins",
                ),
                Resource(
                    uri="crypto://yields/top",
                    name="Top DeFi Yields",
                    mimeType="application/json",
                    description="Top yield opportunities across DeFi protocols",
                ),
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri == "crypto://market/overview":
                data = await self._get_market_data(10)
                return json.dumps(data, indent=2)
            elif uri == "crypto://yields/top":
                data = await self._get_defi_yields(None, 10)
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def _get_eth_balance(self, address: str) -> dict:
        """Get ETH balance for an address."""
        try:
            # Validate address format
            if not address.startswith("0x") or len(address) != 42:
                return {"error": "Invalid Ethereum address format"}
            
            # Use Etherscan API (no API key needed for basic calls, rate limited)
            url = f"{ETHERSCAN_API}?module=account&action=balance&address={address}&tag=latest"
            
            response = await self.http_client.get(url)
            data = response.json()
            
            if data.get("status") == "1":
                balance_wei = int(data["result"])
                balance_eth = balance_wei / 10**18
                return {
                    "address": address,
                    "balance_eth": round(balance_eth, 6),
                    "balance_wei": balance_wei,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Fallback: try RPC call
                return await self._get_eth_balance_rpc(address)
                
        except Exception as e:
            return {"error": f"Failed to fetch balance: {str(e)}"}
    
    async def _get_eth_balance_rpc(self, address: str) -> dict:
        """Get ETH balance via RPC."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [address, "latest"],
                "id": 1,
            }
            
            response = await self.http_client.post(
                self.eth_rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            
            result = response.json()
            if "result" in result:
                balance_wei = int(result["result"], 16)
                balance_eth = balance_wei / 10**18
                return {
                    "address": address,
                    "balance_eth": round(balance_eth, 6),
                    "balance_wei": balance_wei,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "rpc",
                }
            else:
                return {"error": f"RPC error: {result.get('error', 'Unknown')}"}
                
        except Exception as e:
            return {"error": f"RPC failed: {str(e)}"}
    
    async def _get_token_price(self, token_id: str) -> dict:
        """Get token price from CoinGecko."""
        try:
            url = f"{COINGECKO_API}/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_change=true"
            
            response = await self.http_client.get(url)
            data = response.json()
            
            if token_id in data:
                return {
                    "token_id": token_id,
                    "price_usd": data[token_id]["usd"],
                    "change_24h_percent": data[token_id].get("usd_24h_change"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                return {"error": f"Token '{token_id}' not found. Try 'bitcoin', 'ethereum', 'solana', etc."}
                
        except Exception as e:
            return {"error": f"Failed to fetch price: {str(e)}"}
    
    async def _get_defi_yields(self, chain: Optional[str], limit: int) -> dict:
        """Get DeFi yields from DeFi Llama."""
        try:
            url = f"{DEFILLAMA_API}/pools"
            
            response = await self.http_client.get(url)
            data = response.json()
            
            pools = data.get("data", [])
            
            # Filter by chain if specified
            if chain:
                pools = [p for p in pools if p.get("chain", "").lower() == chain.lower()]
            
            # Sort by APY descending
            pools = sorted(pools, key=lambda x: x.get("apy", 0), reverse=True)
            
            # Take top N
            top_pools = pools[:limit]
            
            results = []
            for pool in top_pools:
                results.append({
                    "pool": pool.get("pool"),
                    "chain": pool.get("chain"),
                    "project": pool.get("project"),
                    "symbol": pool.get("symbol"),
                    "apy": pool.get("apy"),
                    "apy_base": pool.get("apyBase"),
                    "apy_reward": pool.get("apyReward"),
                    "tvl_usd": pool.get("tvlUsd"),
                    "risk_level": pool.get("ilRisk"),
                    "exposure": pool.get("exposure"),
                })
            
            return {
                "count": len(results),
                "chain_filter": chain,
                "pools": results,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch yields: {str(e)}"}
    
    async def _get_gas_prices(self) -> dict:
        """Get Ethereum gas prices."""
        try:
            # Use Etherscan for gas prices
            url = f"{ETHERSCAN_API}?module=gastracker&action=gasoracle"
            
            response = await self.http_client.get(url)
            data = response.json()
            
            if data.get("status") == "1":
                result = data["result"]
                return {
                    "safe_low": {
                        "gwei": int(result["SafeGasPrice"]),
                        "estimated_time": "~10 min",
                    },
                    "standard": {
                        "gwei": int(result["ProposeGasPrice"]),
                        "estimated_time": "~3 min",
                    },
                    "fast": {
                        "gwei": int(result["FastGasPrice"]),
                        "estimated_time": "~30 sec",
                    },
                    "base_fee": result.get("suggestBaseFee"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                return {"error": "Failed to fetch gas prices from Etherscan"}
                
        except Exception as e:
            return {"error": f"Failed to fetch gas prices: {str(e)}"}
    
    async def _get_market_data(self, limit: int) -> dict:
        """Get market data for top cryptocurrencies."""
        try:
            url = f"{COINGECKO_API}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1"
            
            response = await self.http_client.get(url)
            data = response.json()
            
            coins = []
            for coin in data:
                coins.append({
                    "rank": coin.get("market_cap_rank"),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name"),
                    "price_usd": coin.get("current_price"),
                    "market_cap_usd": coin.get("market_cap"),
                    "volume_24h_usd": coin.get("total_volume"),
                    "change_24h_percent": coin.get("price_change_percentage_24h"),
                    "change_7d_percent": coin.get("price_change_percentage_7d_in_currency"),
                })
            
            return {
                "count": len(coins),
                "coins": coins,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server(self.server) as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main():
    """Main entry point."""
    server = CryptoDeFiServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
