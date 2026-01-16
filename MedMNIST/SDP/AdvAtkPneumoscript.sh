#!/bin/bash
COUNT=0
K=0
LIMIT=100
OUTPUT_FILE="resultados_abs.txt"
> "$OUTPUT_FILE"
while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
    python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --p 100
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    EPSILON="$(bc <<< "$K + 1")"
    echo $EPSILON
done