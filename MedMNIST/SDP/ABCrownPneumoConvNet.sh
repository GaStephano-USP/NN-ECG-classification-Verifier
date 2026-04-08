#!/bin/bash
COUNT=0
LIMIT=45
K=0
MODE='Rot'
OUTPUT_FILE="resultadospneumomnist.txt"
ANGLE=0
> "$OUTPUT_FILE"

elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --l1 15 --l2 28 --c1 15 --c2 28
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done

elif [ "$MODE" == "Rot" ]; then

    while [ "$(bc <<< "$ANGLE < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --angle $ANGLE --mode "Rot"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        ANGLE="$(bc <<< "$ANGLE + 0.5")"
        echo $ANGLE  
    done
fi