#!/bin/bash
COUNT=0
EPSILON=0.000
LIMIT=784
K=708
MODE="Crop"
P0x=0
P0y=0
SEED=60601
OUTPUT_FILE="results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_Crop.txt"
ANGLE=0
> "$OUTPUT_FILE"
start=`date +%s`
if [ "$MODE" == 'rel_abs' ]; then

    while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
        echo $EPSILON
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --epsilon $EPSILON --mode "rel"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        EPSILON="$(bc <<< "$EPSILON + 0.001")"
    done

elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        echo $K
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --seed $SEED --p 0 --model CNN --model_path trained_models/PneumoniaMNIST/CNN/PneumoniaMNISTCNN3_Max.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_pneumoniaMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
    done

elif [ "$MODE" == "Rot" ]; then
    while [ "$(bc <<< "$ANGLE < $LIMIT")" == "1" ]; do
        echo $ANGLE
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --angle $ANGLE --mode "Rot" --model CNN --model_path trained_models/PneumoniaMNIST/CNN/PneumoniaMNISTCNN3_Max.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_pneumoniaMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        ANGLE="$(bc <<< "$ANGLE + 4")"  
    done
elif [ "$MODE" == "Crop" ]; then
    while [ "$(bc <<< "$P0y < 25")" == "1" ]; do
        P0x=0
        while [ "$(bc <<< "$P0x < 25")" == "1" ]; do
            python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --mode "Crop" --P0 $P0x $P0y --altura 3 --largura 3 --model CNN --model_path trained_models/PneumoniaMNIST/CNN/PneumoniaMNISTCNN3_Max.pth
            output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_pneumoniaMNIST.yaml )
            match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
            echo "$match" >> "$OUTPUT_FILE"
            P0x="$(bc <<< "$P0x + 1")"  
        done
        P0y="$(bc <<< "$P0y + 1")"
    done
fi
end=`date +%s`
runtime=$((end-start))
echo "$runtime" >> "$OUTPUT_FILE"