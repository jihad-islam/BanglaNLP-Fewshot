#!/bin/bash

################################################################################
# Start Backend Only - Manual Control Script
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo -e "${BLUE}Starting Bangla NLP Backend...${NC}"

# Check if already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port 8000 is already in use!${NC}"
    echo "To stop existing backend, run: ./stop-backend.sh"
    exit 1
fi

# Start backend
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run: ./setup-only.sh first"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo -e "${BLUE}Starting server on http://localhost:8000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop${NC}"
echo ""

python main.py
