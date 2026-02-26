"""DefiLlama API tool - DeFi data and analytics."""
import json
import urllib.request
from typing import List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry

BASE_URL = "https://api.llama.fi"


def _fetch(endpoint: str) -> dict:
    """Fetch data from DefiLlama API."""
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Prometheus/6.7", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_protocol_tvl(ctx: ToolContext, protocol: str) -> str:
    """Get TVL data for a specific protocol."""
    try:
        data = _fetch(f"/tvl/{protocol}")
        return f"📊 {protocol.upper()} TVL: ${data:,.0f}" if isinstance(data, (int, float)) else json.dumps(data, indent=2)
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_protocols(ctx: ToolContext) -> str:
    """List all protocols tracked by DefiLlama."""
    try:
        data = _fetch("/protocols")
        protocols = [p["name"] for p in data[:20]]
        return f"📋 Top 20 Protocols:\n" + "\n".join(f"  • {p}" for p in protocols)
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_chain_tvl(ctx: ToolContext, chain: str) -> str:
    """Get TVL for a specific blockchain."""
    try:
        data = _fetch(f"/v2/chains")
        for c in data:
            if c.get("name", "").lower() == chain.lower():
                tvl = c.get("tvl", 0)
                return f"⛓️ {chain.upper()} TVL: ${tvl:,.0f}"
        return f"⚠️ Chain '{chain}' not found"
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_chains(ctx: ToolContext) -> str:
    """List all chains tracked by DefiLlama."""
    try:
        data = _fetch("/v2/chains")
        chains = [(c.get("name", "Unknown"), c.get("tvl", 0)) for c in sorted(data, key=lambda x: x.get("tvl", 0), reverse=True)[:15]]
        lines = [f"⛓️ Top Chains by TVL:"]
        for name, tvl in chains:
            lines.append(f"  • {name}: ${tvl:,.0f}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_stablecoins(ctx: ToolContext, chain: Optional[str] = None) -> str:
    """Get stablecoin data, optionally filtered by chain."""
    try:
        data = _fetch("/stablecoins")
        stables = data.get("peggedAssets", [])
        
        lines = ["💵 Stablecoins:"]
        count = 0
        for s in stables[:15]:
            name = s.get("name", "Unknown")
            symbol = s.get("symbol", "")
            circ = s.get("circulating", {})
            
            if chain:
                chain_circ = circ.get("peggedUSD", 0) if isinstance(circ, dict) else 0
                # Check chain-specific data
                chain_circuits = s.get("chainCirculating", {})
                if chain.lower() in chain_circuits:
                    chain_data = chain_circuits[chain.lower()]
                    chain_circ = chain_data.get("current", {}).get("peggedUSD", 0) if isinstance(chain_data, dict) else 0
                    if chain_circ > 0:
                        lines.append(f"  • {name} ({symbol}): ${chain_circ:,.0f}")
                        count += 1
            else:
                total = circ.get("peggedUSD", 0) if isinstance(circ, dict) else 0
                if total > 0:
                    lines.append(f"  • {name} ({symbol}): ${total:,.0f}")
                    count += 1
        
        if count == 0 and chain:
            return f"⚠️ No stablecoin data for chain '{chain}'"
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_yields(ctx: ToolContext, chain: Optional[str] = None) -> str:
    """Get yield/APY data for DeFi pools."""
    try:
        data = _fetch("/pools")
        pools = data.get("data", [])
        
        lines = ["📈 Top Yields:"]
        count = 0
        for p in pools[:10]:
            pool_chain = p.get("chain", "Unknown")
            if chain and pool_chain.lower() != chain.lower():
                continue
            
            symbol = p.get("symbol", "Unknown")
            apy = p.get("apy", 0)
            tvl = p.get("tvlUsd", 0)
            project = p.get("project", "Unknown")
            
            lines.append(f"  • {symbol} on {project} ({pool_chain}): {apy:.2f}% APY | ${tvl:,.0f} TVL")
            count += 1
            if count >= 10:
                break
        
        if count == 0:
            return f"⚠️ No yield data found" + (f" for chain '{chain}'" if chain else "")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Error: {e}"


def _get_global_tvl(ctx: ToolContext) -> str:
    """Get global DeFi TVL across all chains."""
    try:
        data = _fetch("/charts")
        if data and len(data) > 0:
            latest = data[-1]
            tvl = latest.get("totalLiquidityUSD", 0)
            date = latest.get("date", "unknown")
            return f"🌍 Global DeFi TVL: ${tvl:,.0f} (as of {date})"
        return "⚠️ No TVL data available"
    except Exception as e:
        return f"⚠️ Error: {e}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("defillama_protocol_tvl", {
            "name": "defillama_protocol_tvl",
            "description": "Get TVL (Total Value Locked) for a specific DeFi protocol (e.g., 'aave', 'uniswap', 'lido')",
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string", "description": "Protocol slug (e.g., 'aave', 'uniswap', 'lido')"}
                },
                "required": ["protocol"]
            }
        }, _get_protocol_tvl),
        
        ToolEntry("defillama_protocols", {
            "name": "defillama_protocols",
            "description": "List all DeFi protocols tracked by DefiLlama",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }, _get_protocols),
        
        ToolEntry("defillama_chain_tvl", {
            "name": "defillama_chain_tvl",
            "description": "Get TVL for a specific blockchain (e.g., 'ethereum', 'solana', 'arbitrum')",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Chain name (e.g., 'ethereum', 'solana', 'arbitrum')"}
                },
                "required": ["chain"]
            }
        }, _get_chain_tvl),
        
        ToolEntry("defillama_chains", {
            "name": "defillama_chains",
            "description": "List all blockchains tracked by DefiLlama with their TVL",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }, _get_chains),
        
        ToolEntry("defillama_stablecoins", {
            "name": "defillama_stablecoins",
            "description": "Get stablecoin market data. Optionally filter by chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Optional chain filter (e.g., 'ethereum', 'solana')", "default": None}
                }
            }
        }, _get_stablecoins),
        
        ToolEntry("defillama_yields", {
            "name": "defillama_yields",
            "description": "Get yield/APY data for DeFi pools. Optionally filter by chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Optional chain filter (e.g., 'ethereum', 'solana')", "default": None}
                }
            }
        }, _get_yields),
        
        ToolEntry("defillama_global_tvl", {
            "name": "defillama_global_tvl",
            "description": "Get global DeFi TVL across all chains",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }, _get_global_tvl),
    ]
