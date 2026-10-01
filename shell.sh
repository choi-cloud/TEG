#!/bin/bash

DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S%3N")

OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"

LOG_FILE="${OUT_DIR}/${TIME}.log"

python -u main.py --dataset corafull --way 5 --shot 3 > "${LOG_FILE}" 2>&1 &

echo "Running TEG in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"