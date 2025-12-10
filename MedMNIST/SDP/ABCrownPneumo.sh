#!/bin/bash
COUNT=0
EPSILON=0.00
LIMIT=0.2
OUTPUT_FILE="resultados1.txt"
> "$OUTPUT_FILE"
while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
    python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --epsilon $EPSILON --mode "rel"
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    EPSILON="$(bc <<< "$EPSILON + 0.001")"
    echo $EPSILON
done