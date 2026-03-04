#!/bin/bash
# Run all datasets with lr=[0.001,0.001], hidden_dim=[768,768]
# eurlex_ev gets 300 epochs, all others get 70
# Max 3 concurrent processes

cd "$(dirname "$0")"

COMMON="--lr-energy 0.001 --lr-task 0.001 --hidden-dim 768 --energy-hidden 768"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

DATASETS=(bibtex delicious genbase expr_fun eurlex_ev cal500 spo_fun)
MAX_PARALLEL=3
PIDS=()
NAMES=()

wait_for_slot() {
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                NEW_PIDS+=("${PIDS[$i]}")
                NEW_NAMES+=("${NAMES[$i]}")
            else
                wait "${PIDS[$i]}"
                echo "[DONE] ${NAMES[$i]} exited with code $?"
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        NAMES=("${NEW_NAMES[@]}")
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done
}

for ds in "${DATASETS[@]}"; do
    if [ "$ds" = "eurlex_ev" ]; then
        EPOCHS=300
    else
        EPOCHS=70
    fi

    wait_for_slot

    echo "[START] $ds (epochs=$EPOCHS)"
    python train.py --dataset "$ds" --epochs $EPOCHS $COMMON --log-dir "$LOG_DIR" > /dev/null 2>&1 &
    PIDS+=($!)
    NAMES+=("$ds")
done

# Wait for all remaining
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    echo "[DONE] ${NAMES[$i]} exited with code $?"
done

echo ""
echo "All runs complete. Logs in $LOG_DIR/"
