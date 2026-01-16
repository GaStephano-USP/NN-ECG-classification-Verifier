#!/bin/bash
COUNT=0
EPSILON=0.00
LIMIT=0.2
K=0
MODE='SnP'
OUTPUT_FILE="resultadosoctmnist.txt"
> "$OUTPUT_FILE"
if [ "$MODE" == 'rel_abs' ]; then
    while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --epsilon $EPSILON --mode "rel"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        EPSILON="$(bc <<< "$EPSILON + 0.001")"
        echo $EPSILON
    done
elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" 
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs//OCTMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done
fi 