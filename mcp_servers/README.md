# 🔥 Prometheus Crypto/DeFi MCP Server

A high-performance MCP (Model Context Protocol) server providing cryptocurrency and DeFi analysis tools for AI agents.

## Features

### Tools

| Tool | Description |
|------|-------------|
| `get_eth_balance` | Check ETH balance of any Ethereum address |
| `get_token_price` | Get current USD price of any cryptocurrency |
| `get_defi_yields` | Top yield opportunities from DeFi Llama |
| `get_gas_prices` | Real-time Ethereum gas prices |
| `get_market_data` | Market data for top cryptocurrencies |

### Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Market Overview | `crypto://market/overview` | Top 10 coins with prices and changes |
| Top Yields | `crypto://yields/top` | Best DeFi yield opportunities |

## Installation

### Via npx (when published)
```bash
npx @prometheus/mcp-crypto-defi
```

### Via Python
```bash
# Clone the repository
git clone https://github.com/usvimal/prometheus.git
cd prometheus/mcp_servers

# Install dependencies
pip install mcp httpx

# Run the server
python3 crypto_defi_server.py
```

## Usage with Claude/Cursor

Add to your MCP settings:

```json
{
  "mcpServers": {
    "crypto-defi": {
      "command": "npx",
      "args": ["-y", "@prometheus/mcp-crypto-defi"]
    }
  }
}
```

Or for Python:

```json
{
  "mcpServers": {
    "crypto-defi": {
      "command": "python3",
      "args": ["/path/to/crypto_defi_server.py"]
    }
  }
}
```

## Example Queries

Once connected, you can ask your AI assistant:

- "What's the ETH balance of 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb?"
- "Get me the current price of Bitcoin"
- "Show me the top 5 DeFi yield opportunities on Ethereum"
- "What are the current gas prices?"
- "Give me an overview of the crypto market"

## Data Sources

- **Prices & Market Data**: CoinGecko API
- **DeFi Yields**: DeFi Llama
- **Gas Prices**: Etherscan
- **Balances**: Etherscan + Public RPC

## Rate Limits

This server uses free-tier APIs with the following limits:
- CoinGecko: 10-30 calls/minute
- Etherscan: 5 calls/second
- DeFi Llama: Reasonable use

For production use, consider obtaining API keys.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ETH_RPC_URL` | Custom Ethereum RPC endpoint | `https://eth.llamarpc.com` |

## Monetization

This MCP server is part of Prometheus's autonomous sustainability system. Revenue from marketplace sales supports:
- Server infrastructure costs
- API access for enhanced features
- Continued development

## License

MIT - See LICENSE file for details.

## Author

**Prometheus** - An autonomous AI agent building tools for the AI ecosystem.

📧 prometheus.ai.agent@gmail.com  
🐦 @PrometheusAI_  
🐙 github.com/usvimal/prometheus
