#!/bin/bash
# Prometheus VPS Setup Script
# Run as root on a fresh Ubuntu/Debian VPS.

set -e

echo "=== Prometheus VPS Setup ==="

# 1. Create prometheus user
if ! id -u prometheus &>/dev/null; then
    useradd -m -s /bin/bash prometheus
    echo "Created user: prometheus"
fi

# 2. Install system deps
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git caddy

# 3. Setup swap (if not already present)
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap configured: 2GB"
fi

# 4. Create directory structure
sudo -u prometheus mkdir -p /home/prometheus/prometheus/{data,repo}
sudo -u prometheus mkdir -p /home/prometheus/prometheus/data/{state,logs,memory,index,locks,archive}

# 5. Clone repo
if [ ! -d "/home/prometheus/prometheus/repo/.git" ]; then
    echo "Clone the repository:"
    echo "  sudo -u prometheus git clone https://github.com/usvimal/prometheus.git /home/prometheus/prometheus/repo"
fi

# 6. Install Python deps
sudo -u prometheus bash -c '
    cd /home/prometheus/prometheus/repo
    python3 -m pip install --user -r requirements.txt
    python3 -m playwright install chromium --with-deps 2>/dev/null || true
'

# 7. Copy config template
if [ ! -f /home/prometheus/prometheus/config.env ]; then
    sudo -u prometheus cp /home/prometheus/prometheus/repo/config.env.example /home/prometheus/prometheus/config.env
    echo "Config template copied to /home/prometheus/prometheus/config.env"
    echo "EDIT THIS FILE with your secrets before starting!"
fi

# 8. Install systemd services
cp /home/prometheus/prometheus/repo/deploy/prometheus-agent.service /etc/systemd/system/
cp /home/prometheus/prometheus/repo/deploy/prometheus-dashboard.service /etc/systemd/system/
systemctl daemon-reload
echo "Systemd services installed."

# 9. Caddy config
echo "To configure Caddy, edit /etc/caddy/Caddyfile or copy deploy/Caddyfile"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit /home/prometheus/prometheus/config.env with your secrets"
echo "  2. systemctl enable --now prometheus-agent"
echo "  3. systemctl enable --now prometheus-dashboard"
echo "  4. Send /login to your Telegram bot to authenticate with ChatGPT"
echo ""
