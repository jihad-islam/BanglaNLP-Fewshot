#!/bin/bash

################################################################################
# Stop Backend - Kill all backend processes
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Bangla NLP Backend...${NC}"

# Check if backend is running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    # Kill processes on port 8000
    PID=$(lsof -ti:8000)
    kill $PID 2>/dev/null
    sleep 2
    
    # Force kill if still running
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        kill -9 $PID 2>/dev/null
    fi
    
    echo -e "${GREEN}✓ Backend stopped${NC}"
else
    echo -e "${YELLOW}Backend is not running${NC}"
fi

# Also kill any stray python main.py processes
if pgrep -f "python.*main.py" > /dev/null; then
    pkill -f "python.*main.py"
    echo -e "${GREEN}✓ Cleaned up background processes${NC}"
fi

echo -e "${GREEN}Done!${NC}"
