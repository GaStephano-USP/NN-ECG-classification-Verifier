#!/bin/bash
COUNT=0
EPSILON=0.00
LIMIT=50
K=0
MODE='SnP'
OUTPUT_FILE="resultadospneumomnist.txt"
> "$OUTPUT_FILE"
if [ "$MODE" == 'rel_abs' ]; then

    while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --epsilon $EPSILON --mode "rel"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        EPSILON="$(bc <<< "$EPSILON + 0.001")"
        echo $EPSILON
    done

elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --l1 15 --l2 28 --c1 15 --c2 28
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done
fi 