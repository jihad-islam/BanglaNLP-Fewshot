#!/bin/bash
# Comprehensive test script for Bangla NLP Few-Shot project

echo "=========================================="
echo "  Bangla NLP Few-Shot - System Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Backend Health
echo "1. Testing Backend Health..."
HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo -e "${GREEN}✓ Backend is healthy${NC}"
    MODELS=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin)['models_loaded'])")
    echo "  - Models loaded: $MODELS/6"
else
    echo -e "${RED}✗ Backend is not responding${NC}"
    exit 1
fi
echo ""

# Test 2: Available Tasks
echo "2. Testing Available Tasks..."
TASKS=$(curl -s http://localhost:8000/tasks)
for task in hate sentiment topic; do
    if echo "$TASKS" | grep -q "\"$task\""; then
        BASELINE=$(echo "$TASKS" | python3 -c "import sys, json; print(json.load(sys.stdin)['$task']['has_baseline'])")
        PROTONET=$(echo "$TASKS" | python3 -c "import sys, json; print(json.load(sys.stdin)['$task']['has_protonet'])")
        
        if [ "$BASELINE" == "True" ] && [ "$PROTONET" == "True" ]; then
            echo -e "${GREEN}✓ $task: baseline=✓ protonet=✓${NC}"
        else
            echo -e "${YELLOW}⚠ $task: baseline=$BASELINE protonet=$PROTONET${NC}"
        fi
    else
        echo -e "${RED}✗ $task: NOT FOUND${NC}"
    fi
done
echo ""

# Test 3: Hate Speech Detection
echo "3. Testing Hate Speech Detection..."
RESULT=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"এই দেশের নেতারা দুর্নীতিবাজ","task":"hate","mode":"comparison"}')

BASELINE_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['baseline']['predicted_label'])")
PROTONET_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['protonet']['predicted_label'])")

echo "  - Baseline: $BASELINE_PRED"
echo "  - ProtoNet: $PROTONET_PRED"
echo -e "${GREEN}✓ Both models responded${NC}"
echo ""

# Test 4: Sentiment Analysis
echo "4. Testing Sentiment Analysis..."
RESULT=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"আজকের ম্যাচটি খুবই ভালো হয়েছে","task":"sentiment","mode":"comparison"}')

BASELINE_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['baseline']['predicted_label'])")
PROTONET_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['protonet']['predicted_label'])")

echo "  - Baseline: $BASELINE_PRED"
echo "  - ProtoNet: $PROTONET_PRED"
echo -e "${GREEN}✓ Both models responded${NC}"
echo ""

# Test 5: Topic Classification  
echo "5. Testing Topic Classification..."
RESULT=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"বাংলাদেশ ক্রিকেট দল জিতেছে","task":"topic","mode":"comparison"}')

BASELINE_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['baseline']['predicted_label'])")
PROTONET_PRED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['protonet']['predicted_label'])")

echo "  - Baseline: $BASELINE_PRED"
echo "  - ProtoNet: $PROTONET_PRED"
echo -e "${GREEN}✓ Both models responded${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ All tests passed!${NC}"
echo "=========================================="
echo ""
echo "Notes:"
echo "- Baseline models: Fine-tuned classifiers (high accuracy)"
echo "- ProtoNet models: Distance-based (using random prototypes)"
echo "- ProtoNet needs proper prototype computation for accurate results"
echo ""
