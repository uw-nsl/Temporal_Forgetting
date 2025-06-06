
models=(
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_32"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_64"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_96"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_128"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_160"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_192"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_224"
    "UWNSL/Qwen2.5-7B-deepscaler_4k_step_256"
    "Qwen/Qwen2.5-7B"
)
tasks=("AIME24" "AMC" "AIME25")

max_model_tokens=16000
max_gen_tokens=16000
model_args="tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=$max_model_tokens,dtype=bfloat16"
output_path="Sampling_Responses"
batch_size="auto"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do

        SANTIZED_MODEL_SAVE_LABEL=$(echo ${model} | sed 's/\//__/g')
        SAMPLE_FILE="${output_path}/${SANTIZED_MODEL_SAVE_LABEL}/samples_${task}_"*".jsonl"
        if ls $SAMPLE_FILE 2>/dev/null; then
            echo "The file already exists: $model - $task"
            continue
        fi    

        echo "Running lm_eval with model: $model, task: $task"
        lm_eval --model vllm \
            --model_args pretrained="$model",$model_args \
            --gen_kwargs do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=$max_gen_tokens\
            --tasks "$task" \
            --batch_size "$batch_size" \
            --log_samples \
            --trust_remote_code \
            --output_path "$output_path" \
            --apply_chat_template \


        SANTIZED_MODEL_SAVE_LABEL=$(echo ${model} | sed 's/\//__/g')
        echo ${SANTIZED_MODEL_SAVE_LABEL}
        python math_metric_llm_eval_general.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} --task ${task}


    done
done
