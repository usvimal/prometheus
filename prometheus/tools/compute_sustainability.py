"""Compute Sustainability Tool - Find ways to sustain compute resources."""
import json
import urllib.request
from typing import List, Optional, Dict, Any

from prometheus.tools.registry import ToolContext, ToolEntry

YIELDS_URL = "https://yields.llama.fi"


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Prometheus/6.7", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _analyze_yields(chain: Optional[str] = None, min_tvl: float = 1000000) -> Dict[str, Any]:
    """Analyze yield opportunities for compute sustainability."""
    try:
        data = _fetch_json(f"{YIELDS_URL}/pools")
        pools = data.get("data", [])
        
        opportunities = []
        stable_opportunities = []
        
        for p in pools:
            pool_chain = p.get("chain", "Unknown")
            if chain and pool_chain.lower() != chain.lower():
                continue
            
            apy = p.get("apy", 0)
            tvl = p.get("tvlUsd", 0)
            symbol = p.get("symbol", "Unknown")
            project = p.get("project", "Unknown")
            
            if tvl < min_tvl:
                continue
            
            pool_data = {
                "symbol": symbol,
                "project": project,
                "chain": pool_chain,
                "apy": apy,
                "tvl_usd": tvl
            }
            
            # Categorize by risk (simplified)
            if symbol in ["USDC", "USDT", "DAI", "BUSD"]:
                stable_opportunities.append(pool_data)
            else:
                opportunities.append(pool_data)
        
        # Sort by APY
        stable_opportunities.sort(key=lambda x: x.get("apy", 0), reverse=True)
        opportunities.sort(key=lambda x: x.get("apy", 0), reverse=True)
        
        return {
            "status": "success",
            "stable": stable_opportunities[:10],
            "volatile": opportunities[:10],
            "total_pools_analyzed": len(pools)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_compute_sustainability_advice(ctx: ToolContext) -> str:
    """Get advice on how to sustain compute resources."""
    analysis = _analyze_yields()
    
    if analysis.get("status") == "error":
        return f"⚠️ Error analyzing yields: {analysis.get('message')}"
    
    lines = [
        "💡 **Compute Sustainability Advice**",
        "",
        "## 🎯 Immediate Options",
        "",
        "### Stable Yields (Low Risk)",
    ]
    
    stable = analysis.get("stable", [])
    if stable:
        for p in stable[:5]:
            lines.append(f"  • **{p['symbol']}** ({p['project']}): {p['apy']:.2f}% APY")
    else:
        lines.append("  No stable pools found above $1M TVL")
    
    lines.extend([
        "",
        "### Higher Yields (Higher Risk)",
    ])
    
    volatile = analysis.get("volatile", [])
    if volatile:
        for p in volatile[:5]:
            lines.append(f"  • **{p['symbol']}** ({p['project']}): {p['apy']:.2f}% APY")
    
    lines.extend([
        "",
        "## 💰 Action Items",
        "1. **Deploy capital** to generate passive yield",
        "2. **Monitor gas costs** - ETH gas may exceed yield",
        "3. **Consider automation** to minimize manual work",
        "",
        f"📊 Analyzed {analysis.get('total_pools_analyzed', 0)} pools"
    ])
    
    return "\n".join(lines)


def _get_top_yields(ctx: ToolContext, chain: Optional[str] = None) -> str:
    """Get top yield opportunities."""
    analysis = _analyze_yields(chain=chain)
    
    if analysis.get("status") == "error":
        return f"⚠️ Error: {analysis.get('message')}"
    
    lines = ["📈 **Top Yield Opportunities**", ""]
    
    all_pools = analysis.get("stable", []) + analysis.get("volatile", [])
    all_pools.sort(key=lambda x: x.get("apy", 0), reverse=True)
    
    for i, p in enumerate(all_pools[:10], 1):
        lines.append(f"{i}. **{p['symbol']}** @ {p['apy']:.2f}% ({p['project']}, {p['chain']})")
    
    return "\n".join(lines)


def _estimate_passive_income(ctx: ToolContext, principal: float, asset: str = "USDC") -> str:
    """Estimate passive income from DeFi yields."""
    analysis = _analyze_yields()
    
    if analysis.get("status") == "error":
        return f"⚠️ Error: {analysis.get('message')}"
    
    # Find APY for the asset
    apy = 0
    for p in analysis.get("stable", []) + analysis.get("volatile", []):
        if p.get("symbol", "").upper() == asset.upper():
            apy = p.get("apy", 0)
            break
    
    if apy == 0:
        return f"⚠️ Could not find yield data for {asset}"
    
    daily = principal * (apy / 100) / 365
    monthly = principal * (apy / 100) / 12
    yearly = principal * (apy / 100)
    
    lines = [
        f"💵 **Passive Income Estimate for ${principal:,.0f} {asset}**",
        "",
        f"**APY:** {apy:.2f}%",
        "",
        f"📅 Daily: ${daily:.2f}",
        f"📆 Monthly: ${monthly:.2f}",
        f"📈 Yearly: ${yearly:.2f}",
    ]
    
    # Estimate compute equivalent
    # Assume ~$0.50 per 1M tokens (rough Kimi estimate)
    tokens_per_dollar = 2_000_000
    daily_tokens = daily * tokens_per_dollar
    lines.extend([
        "",
        f"⚡ Equivalent compute: ~{daily_tokens/1e6:.1f}M tokens/day"
    ])
    
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("compute_sustainability_advice", {
            "name": "compute_sustainability_advice",
            "description": "Get advice on how to sustain compute resources through DeFi yields and passive income strategies",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }, _get_compute_sustainability_advice),
        
        ToolEntry("compute_top_yields", {
            "name": "compute_top_yields",
            "description": "Get top yield opportunities from DeFi pools",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Optional chain filter (e.g., 'ethereum', 'solana')", "default": None}
                }
            }
        }, _get_top_yields),
        
        ToolEntry("compute_estimate_income", {
            "name": "compute_estimate_income",
            "description": "Estimate passive income from DeFi yields. Shows daily, monthly, and yearly returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number", "description": "Amount in USD to invest"},
                    "asset": {"type": "string", "description": "Asset symbol (default: USDC)", "default": "USDC"}
                },
                "required": ["principal"]
            }
        }, _estimate_passive_income),
    ]
