#!/bin/bash

################################################################################
# Stop Frontend - Kill frontend processes
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Bangla NLP Frontend...${NC}"

# Check if frontend is running
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    # Kill processes on port 3000
    PID=$(lsof -ti:3000)
    kill $PID 2>/dev/null
    sleep 2
    
    # Force kill if still running
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        kill -9 $PID 2>/dev/null
    fi
    
    echo -e "${GREEN}✓ Frontend stopped${NC}"
else
    echo -e "${YELLOW}Frontend is not running${NC}"
fi

# Also kill any stray npm start processes
if pgrep -f "node.*react-scripts" > /dev/null; then
    pkill -f "node.*react-scripts"
    echo -e "${GREEN}✓ Cleaned up React processes${NC}"
fi

echo -e "${GREEN}Done!${NC}"
