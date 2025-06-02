#!/bin/bash

# This script runs multiple training scripts sequentially and logs their output.
# It also tracks the success or failure of each script and summarizes the results.
# Ensure this script is run from the root directory of the project
# Usage: bash scripts/train_all.sh

# Define paths relative to this script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/../logs"

# Create logs directory if it doesn't exist
mkdir -p "$log_dir"

# List of training script filenames (assumed in same directory)
train_scripts=("train_agent_32b_think_empty.sh" "train_agent_32b_think_weighted.sh")

# Main log file path
main_log="${log_dir}/run.log"
: > "$main_log"  # Clear previous main log

# Track failed scripts
failed_scripts=()

echo "Training process started: $(date)" | tee -a "$main_log"
echo "" | tee -a "$main_log"

# Loop through and run each script
for script in "${train_scripts[@]}"; do
  script_path="${script_dir}/${script}"
  script_name=$(basename "$script" .sh)
  log_file="${log_dir}/${script_name}.log"

  echo ">>> Running $script_path (log: $log_file)..." | tee -a "$main_log"
  echo "------ $script_path started at: $(date) ------" > "$log_file"

  # Time tracking
  start_time=$(date +%s)

  # Execute the script and redirect output
  bash "$script_path" >> "$log_file" 2>&1
  status=$?

  end_time=$(date +%s)
  duration=$((end_time - start_time))

  if [ $status -ne 0 ]; then
    echo "!!! $script FAILED (Duration: ${duration} seconds)." | tee -a "$main_log"
    failed_scripts+=("$script")
  else
    echo "*** $script completed successfully (Duration: ${duration} seconds)." | tee -a "$main_log"
  fi

  echo "------ $script_path ended at: $(date)" >> "$log_file"
  echo "------ Duration: ${duration} seconds" >> "$log_file"
  echo "" | tee -a "$main_log"
done

# Final summary
echo "----------------------------------------" | tee -a "$main_log"
echo "Training process finished: $(date)" | tee -a "$main_log"

if [ ${#failed_scripts[@]} -eq 0 ]; then
  echo "All training scripts completed successfully." | tee -a "$main_log"
else
  echo "The following scripts failed:" | tee -a "$main_log"
  for failed in "${failed_scripts[@]}"; do
    echo "  - $failed" | tee -a "$main_log"
  done
fi
