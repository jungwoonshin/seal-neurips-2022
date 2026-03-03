#!/bin/bash
# Run SEAL-2.0 configs on bibtex data for 2 epochs each (CPU, quick validation)
# Logs results to seal2_test_results.txt

LOG_FILE="seal2_test_results.txt"
VENV=".venv_seal/bin"

echo "=== SEAL-2.0 Test Run $(date) ===" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run each config with overrides for quick CPU testing
for config_name in "bibtex_seal2_structured_nce" "bibtex_seal2_refinement" "bibtex_seal2_full"; do
    echo "======================================" >> "$LOG_FILE"
    echo "Config: $config_name" >> "$LOG_FILE"
    echo "Start: $(date)" >> "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"

    SERIAL_DIR="/tmp/seal2_test_${config_name}"
    rm -rf "$SERIAL_DIR"

    echo "Running $config_name ..."

    $VENV/allennlp train \
        "example_configs/${config_name}.json" \
        --serialization-dir "$SERIAL_DIR" \
        --include-package seal \
        --overrides '{"trainer": {"num_epochs": 2, "cuda_device": -1, "patience": null}, "data_loader": {"batch_size": 16}}' \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    echo "" >> "$LOG_FILE"
    echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
    echo "End: $(date)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Extract metrics if available
    if [ -f "$SERIAL_DIR/metrics_epoch_1.json" ]; then
        echo "--- Epoch 1 Metrics ---" >> "$LOG_FILE"
        cat "$SERIAL_DIR/metrics_epoch_1.json" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi

    echo "Finished $config_name (exit code: $EXIT_CODE)"
done

echo "" >> "$LOG_FILE"
echo "=== All tests completed $(date) ===" >> "$LOG_FILE"
echo "Results saved to $LOG_FILE"
