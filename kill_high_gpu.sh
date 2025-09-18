#!/bin/bash

# Get the output of nvidia-smi
output=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)

# Check if nvidia-smi returned valid output
if [[ -z "$output" ]]; then
    echo "No active GPU processes or incompatible nvidia-smi query. Exiting."
    exit 1
fi

# Parse the output
echo "$output" | while IFS=',' read -r pid memory; do
    # Trim whitespace (in case of inconsistencies)
    pid=$(echo "$pid" | xargs)
    memory=$(echo "$memory" | xargs)

    # Validate PID and memory values
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        # && "$memory" =~ ^[0-9]+$ && $memory -gt 1024
        # Check the command of the process
        cmd=$(ps -p $pid -o comm= 2>/dev/null)
        if [[ $cmd == python* ]]; then
            echo "Killing Python process $pid using $memory MiB of GPU memory."
            kill -9 $pid
        fi
    fi
done

