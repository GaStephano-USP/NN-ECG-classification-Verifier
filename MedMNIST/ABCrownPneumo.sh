#!/bin/bash
COUNT=0
EPSILON=0.00
LIMIT=1.00
OUTPUT_FILE="resultados.txt"
> "$OUTPUT_FILE"
while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
    python3 ./MedMNIST/VNNLIBmakerPneumonia.py --epsilon $EPSILON
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST --csv_name instances.csv)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    EPSILON="$(bc <<< "$EPSILON + 0.01")"
    echo $EPSILON
done