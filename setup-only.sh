#!/bin/bash

################################################################################
# Setup Only - Install dependencies without starting services
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$BACKEND_DIR/venv"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Bangla NLP - Setup Only (No Auto-Start)            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
echo -e "${BLUE}[1/4] Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found!${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Node.js
echo -e "${BLUE}[2/4] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found!${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"

# Setup Backend
echo -e "${BLUE}[3/4] Setting up Backend...${NC}"
cd "$BACKEND_DIR"

if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists, skipping creation${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source venv/bin/activate
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Setup Frontend
echo -e "${BLUE}[4/4] Setting up Frontend...${NC}"
cd "$FRONTEND_DIR"

if [ -d "node_modules" ]; then
    echo -e "${YELLOW}Node modules already exist, skipping installation${NC}"
else
    echo "Installing Node dependencies..."
    npm install --silent
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup Complete!                                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}To start services manually:${NC}"
echo -e "  Backend:  ${GREEN}./start-backend.sh${NC}"
echo -e "  Frontend: ${GREEN}./start-frontend.sh${NC}"
echo ""
echo -e "${BLUE}To stop services:${NC}"
echo -e "  Backend:  ${GREEN}./stop-backend.sh${NC}"
echo -e "  Frontend: ${GREEN}./stop-frontend.sh${NC}"
echo ""
echo -e "${BLUE}Or use the original script to auto-start both:${NC}"
echo -e "  ${GREEN}./run.sh${NC}"
echo ""
