# CAsk Backend - AlmaLinux VPS Deployment Guide

Complete step-by-step guide to deploy the CAsk backend on your AlmaLinux VPS.

> **Note**: This guide assumes you already have a PostgreSQL database URL (e.g., from Neon, Supabase, or another cloud provider).

---

## 1. Initial VPS Setup

### 1.1 Connect to VPS
```bash
ssh root@YOUR_VPS_IP
```

### 1.2 Update System & Install Dependencies
```bash
# Update packages
dnf update -y

# Install EPEL repository (for extra packages)
dnf install -y epel-release

# Install required dependencies
dnf install -y python3.11 python3.11-pip git nginx certbot python3-certbot-nginx
dnf groupinstall -y "Development Tools"

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## 2. Clone & Setup Application

### 2.1 Create App Directory
```bash
mkdir -p /opt/cask
cd /opt/cask
```

### 2.2 Clone Repository
```bash
# Clone directly into the ca-docs folder
git clone https://github.com/Pranav322/ca-docs.git ca-docs
cd ca-docs
```

### 2.3 Setup Python Environment
```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2.4 Create Environment File
```bash
nano .env
```
...

---

## 3. Setup Systemd Service

### 3.1 Create Service File
```bash
nano /etc/systemd/system/cask.service
```

Add this content:
```ini
[Unit]
Description=CAsk Backend API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/cask/ca-docs
EnvironmentFile=/opt/cask/ca-docs/.env
ExecStart=/opt/cask/ca-docs/.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8001 --workers 4 --loop uvloop --http httptools
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 3.2 Enable & Start Service
```bash
systemctl daemon-reload
systemctl enable cask
systemctl start cask

# Check status
systemctl status cask
```

---

## 4. Setup Nginx Reverse Proxy

### 4.1 Create Nginx Config
```bash
nano /etc/nginx/conf.d/cask.conf
```

Add this content (replace `api.yourdomain.com` with your domain or use `_` for IP access):
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;  # Or use _ for any domain/IP

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE streaming support (important for chat!)
        proxy_buffering off;
        proxy_read_timeout 3600s;
        chunked_transfer_encoding on;
    }
}
```

### 4.2 Test & Enable Nginx
```bash
# Test config
nginx -t

# Start nginx
systemctl enable nginx
systemctl start nginx
```

### 4.3 Setup SSL (if using domain)
```bash
certbot --nginx -d api.yourdomain.com
```

---

## 5. Configure Firewall

```bash
# Enable firewalld
systemctl enable firewalld
systemctl start firewalld

# Allow HTTP, HTTPS, SSH
firewall-cmd --permanent --add-service=http
firewall-cmd --permanent --add-service=https
firewall-cmd --permanent --add-service=ssh
firewall-cmd --reload
```

---

## 6. Test Your Deployment

```bash
# Test locally
curl http://localhost:8001/api/health

# Test via nginx
curl http://YOUR_VPS_IP/api/health

# Should return: {"status":"healthy",...}
```

---

## 7. Update Frontend

In your frontend code, update the API URL:

**File: `src/hooks/useChat.ts`**
```typescript
const API_BASE_URL = "https://api.yourdomain.com";  // or http://YOUR_VPS_IP
```

**File: `src/hooks/useCurriculum.ts`**
```typescript
const API_BASE_URL = "https://api.yourdomain.com";
```

Then push to GitHub and redeploy on Vercel.

---

## 8. Quick Commands Reference

| Command | Description |
|---------|-------------|
| `systemctl status cask` | Check if backend is running |
| `systemctl restart cask` | Restart backend |
| `systemctl stop cask` | Stop backend |
| `journalctl -u cask -f` | View live logs |
| `journalctl -u cask -n 100` | View last 100 log lines |
| `curl localhost:8001/api/health` | Test API locally |
| `nginx -t` | Test nginx config |
| `systemctl reload nginx` | Reload nginx |

---

## 9. Troubleshooting

### Service Won't Start
```bash
# Check logs
journalctl -u cask -n 50

# Try running manually to see errors
cd /opt/cask/ca-docs
source .venv/bin/activate
source .env
uvicorn api:app --host 127.0.0.1 --port 8001
```

### 502 Bad Gateway from Nginx
```bash
# Check if cask service is running
systemctl status cask

# Check SELinux (AlmaLinux has it enabled by default)
setsebool -P httpd_can_network_connect 1
```

### Database Connection Issues
```bash
# Test database URL
cd /opt/cask/ca-docs
source .venv/bin/activate
python -c "from database import VectorDatabase; db = VectorDatabase(); print('Connected!')"
```

### Update Code from GitHub
```bash
cd /opt/cask/ca-docs
git pull
source .venv/bin/activate
uv sync
systemctl restart cask
```

---

## 10. SELinux Fix (Important for AlmaLinux!)

AlmaLinux has SELinux enabled by default, which might block nginx from connecting to your backend:

```bash
# Allow nginx to connect to network
setsebool -P httpd_can_network_connect 1

# If still having issues, check audit log
ausearch -m avc -ts recent
```

---

## Summary Checklist

- [ ] SSH into VPS
- [ ] Update system with `dnf update -y`
- [ ] Install dependencies (python3.11, git, nginx, uv)
- [ ] Clone your backend repo
- [ ] Setup venv and install dependencies
- [ ] Create `.env` file with your credentials
- [ ] Create systemd service and start it
- [ ] Configure nginx reverse proxy
- [ ] Setup firewall rules
- [ ] Fix SELinux permissions
- [ ] (Optional) Setup SSL with certbot
- [ ] Update frontend API URL and redeploy
