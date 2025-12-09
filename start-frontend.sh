#!/bin/bash

################################################################################
# Start Frontend Only - Manual Control Script
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo -e "${BLUE}Starting Bangla NLP Frontend...${NC}"

# Check if already running
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port 3000 is already in use!${NC}"
    echo "To stop existing frontend, run: ./stop-frontend.sh"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${RED}Error: Node modules not found!${NC}"
    echo "Please run: ./setup-only.sh first"
    exit 1
fi

# Start frontend
cd "$FRONTEND_DIR"
echo -e "${GREEN}✓ Starting React development server${NC}"
echo -e "${BLUE}Frontend will open at http://localhost:3000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop${NC}"
echo ""

npm start
