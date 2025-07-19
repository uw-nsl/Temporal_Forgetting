# Best-of-N Mathematical Problem Evaluation Project

This project uses Qwen2.5-Math-PRM-72B or Qwen2.5-Math-PRM-7B reward models to evaluate reward scores for mathematical problem responses.

## Project Structure

```
best-of-n/
├── README.md                     # Project documentation
└── qwen_reward/                  # Qwen reward model related code
    ├── batch_reward_corrected.py # Main batch scoring script
    ├── test.sh                   # Batch execution script
    └── prime_math/               # Mathematical utility modules
```

## Features

### Main Functions
- Use Qwen reward model to score mathematical problem answers
- Support batch processing of multiple problems and answers
- Calculate ORM (Outcome Reward Model) and PRM (Process Reward Model) scores
- Automatically detect invalid answers (e.g., missing \\boxed{} tags)



## Usage

### 1. Direct Python Script Usage

```bash
python qwen_reward/batch_reward_corrected.py <folder_path> <task_name>
```

**Parameter Description:**
- `<folder_path>`: Folder path containing sample files (e.g., ../lm-evaluation-harness/sampling_64_responses/Qwen__Qwen2.5-7B)
- `<task_name>`: Task name (e.g., AIME, AIME25, AMC)


### 2. Using Script `test.sh`

The `test.sh` script is used for processing multiple folders.

#### test.sh Usage

1. **Modify Configuration Variables**:
   ```bash
   # Edit test.sh file and modify the following variables:
   TARGET_DIR="../../lm-evaluation-harness/sampling_64_responses"
   TEST_SCRIPT="batch_reward_corrected.py"
   TASK_NAME="AIME"
   ```

2. **Run Script**:
   ```bash
   cd qwen_reward
   bash test.sh
   ```

#### Current Configuration

The script is currently configured to process folders with the following patterns:
- `Qwen__Qwen2.5-7B`
- Folders containing `UWNSL__Qwen2.5-7B-deepscaler_4k`
(modify tesh.sh to change them)

Also modify reward_model_name and batch_size in batch_reward_corrected.py 

### 3. Output Format

The script generates an output file named `QwenReward72B_{task_name}_prm_scored.json`, containing:

```json
[
  {
    "doc_id": "Problem ID",
    "problem": "Problem description", 
    "standard_answer": "Standard answer",
    "responses": [
      {
        "response": "Response content",
        "resp_idx": 0,
        "orm_score": 0.85,
        "prm_score": 0.82,
        "all_step_scores": [0.8, 0.85, 0.8]
      }
    ]
  }
]
```

## Scoring Metrics Description

- **ORM Score (Outcome Reward Score)**: Scoring based on the correctness of the final answer
- **PRM Score (Process Reward Score)**: Average score based on the problem-solving process
- **Step Scores**: Detailed scores for each problem-solving step

## Notes
Invalid answers (missing \\boxed{} or containing "your_answer") will be automatically marked as 0 points


