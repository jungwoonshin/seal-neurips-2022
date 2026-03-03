#!/bin/bash
# Run SEAL-2.0 configs on bibtex data for 2 epochs each (CPU)
# Writes epoch-level results to seal2_experiment_results.txt immediately

RESULTS="seal2_experiment_results.txt"
VENV=".venv_seal/bin"

echo "=== SEAL-2.0 Experiment Results $(date) ===" > "$RESULTS"

run_config() {
    local config_name="$1"
    local serial_dir="/tmp/seal2_test_${config_name}"
    local log_file="/tmp/seal2_log_${config_name}.txt"
    rm -rf "$serial_dir"

    echo "" >> "$RESULTS"
    echo "===== Config: $config_name =====" >> "$RESULTS"
    echo "Start: $(date)" >> "$RESULTS"

    $VENV/allennlp train \
        "example_configs/${config_name}.json" \
        --serialization-dir "$serial_dir" \
        --include-package seal \
        --overrides '{"trainer": {"num_epochs": 2, "cuda_device": -1, "patience": null, "callbacks": []}, "data_loader": {"batch_size": 16}}' \
        > "$log_file" 2>&1
    local exit_code=$?

    echo "Exit code: $exit_code" >> "$RESULTS"

    # Extract epoch summary tables (Training | Validation lines)
    grep -A 20 "Training |  Validation" "$log_file" | grep -E "(Training|MAP|fixed_f1|relaxed_f1|loss|nce|sampler|_inf)" | sed 's/.*console_logger - //' >> "$RESULTS"

    # Extract final JSON metrics
    if grep -q "best_epoch" "$log_file"; then
        echo "" >> "$RESULTS"
        echo "--- Final Metrics ---" >> "$RESULTS"
        sed -n '/"best_epoch"/,/^}/p' "$log_file" | sed 's/.*common.util - //' >> "$RESULTS"
    fi

    echo "End: $(date)" >> "$RESULTS"
    echo "" >> "$RESULTS"

    # Flush immediately so tail -f sees it
    sync
}

for config_name in "bibtex_seal2_structured_nce" "bibtex_seal2_refinement" "bibtex_seal2_full"; do
    echo ">>> Running $config_name ..."
    run_config "$config_name"
    echo ">>> Done $config_name — results appended to $RESULTS"
done

echo "=== All experiments completed $(date) ===" >> "$RESULTS"
echo "All done. Results in $RESULTS"
