

# Target folder path
TARGET_DIR="../../lm-evaluation-harness/sampling_64_responses"
TEST_SCRIPT="batch_reward_corrected.py"
TASK_NAME="AIME"

# Traverse all subfolders in the target folder
count=0

for subdir in "$TARGET_DIR"/*; do
    if [ -d "$subdir" ]; then
        dirname=$(basename "$subdir")
        # Only process folders with specific names
        if [ "$dirname" = "Qwen__Qwen2.5-7B" ] || [[ "$dirname" == *"UWNSL__Qwen2.5-7B-deepscaler"* ]]; then
            echo "Processing: $subdir"
            python "$TEST_SCRIPT" "$subdir" "$TASK_NAME"
            count=$((count + 1))
        else
            echo "Skipping: $subdir"
        fi
    fi
done