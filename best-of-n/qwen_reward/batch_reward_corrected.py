



import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import json
import prime_math



def is_valid_response(response):
    """
    Check if a response is valid:
    1. If there is no answer contained in \\boxed{}, return False
    2. If it contains "your_answer" string, return False
    
    Args:
        response (str): Response text
        
    Returns:
        bool: True if response is valid, False otherwise
    """
    boxed_part = last_boxed_only_string(response)
    if boxed_part is None:
        return False
        
    if "your_answer" in remove_boxed(boxed_part):
        return False
        
    return True

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\fbox{" in s:  # 添加对 \fbox 的处理
        left = "\\fbox{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]

    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    
    # assert s[: len(left)] == left
    # assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string




def split_response_by_paragraphs(response_str):
    """
    Split response string into paragraph list by double newlines (\n\n)
    
    Args:
        response_str (str): Complete response string
        
    Returns:
        list: List of paragraphs split by double newlines
    """
    paragraphs = response_str.split('\n\n')
    # Filter out empty paragraphs
    return [p for p in paragraphs if p.strip()]

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def batch_inference(model, tokenizer, data_list):
    """
    Batch inference function that processes multiple question-answer pairs
    
    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer for the model
        data_list: Data list containing multiple questions and answers
        
    Returns:
        list: List of reward scores for each question-answer pair
    """
    all_conversation_strs = []
    
    for data in data_list:
        paragraphs = split_response_by_paragraphs(data['response'])
        
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(paragraphs) + "<extra_0>"},
        ]
        
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        all_conversation_strs.append(conversation_str)
    
    input_ids = tokenizer(
        all_conversation_strs, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )["input_ids"].to(model.device)
    
    outputs = model(input_ids=input_ids)
    
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_rewards = make_step_rewards(outputs[0], token_masks)
    
    return step_rewards


class QwenRewardModel:
    """
    Qwen model-based reward model
    """
    def __init__(self, reward_model_name="Qwen/Qwen2.5-Math-PRM-72B", device="auto", batch_size=1):
        self.model_name = reward_model_name
        self.device = device
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        self.batch_size = batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        
        print(f"Loaded reward model: {model_name}")


    def score(self, questions, answers_list):
        """
        Score multiple answers for multiple questions
        
        Args:
            questions: List of questions
            answers_list: List of answer lists (each question corresponds to a group of answers)
            
        Returns:
            list: List of scores for all answers of each question
        """
        all_scores = []
        
        for q_idx, question in enumerate(questions):
            question_scores = []
            answers = answers_list[q_idx] if q_idx < len(answers_list) else []
            
            if not answers:
                all_scores.append([])
                continue
            
            # Split answer list into small batches for processing
            step_scores = []
            valid_answers = []
            invalid_answer_indices = []
            
            # Pre-check if each answer is valid
            for i, answer in enumerate(answers):
                if not is_valid_response(answer):
                    invalid_answer_indices.append(i)
                    # Give 0 score directly for invalid answers
                    step_scores.append([0.0])
                else:
                    valid_answers.append(answer)
            
            # Batch process valid answers
            if valid_answers:
                valid_batch_scores = []
                for i in range(0, len(valid_answers), self.batch_size):
                    batch_answers = valid_answers[i:i+self.batch_size]
                    
                    # Prepare batch inference data
                    batch_data = []
                    for answer in batch_answers:
                        data_item = {
                            "system": self.system_prompt,
                            "query": question,
                            "response": answer
                        }
                        batch_data.append(data_item)
                    
                    # Perform inference on each small batch
                    batch_scores = batch_inference(self.model, self.tokenizer, batch_data)
                    valid_batch_scores.extend(batch_scores)
                    
                    # Clean up memory
                    torch.cuda.empty_cache()
            
            # Recombine scores of valid and invalid answers in correct order
            combined_scores = []
            valid_score_idx = 0
            for i in range(len(answers)):
                if i in invalid_answer_indices:
                    combined_scores.append([0.0])
                else:
                    combined_scores.append(valid_batch_scores[valid_score_idx])
                    valid_score_idx += 1
                    
            question_scores = combined_scores
            all_scores.append(question_scores)
        
        return all_scores



# Example usage
if __name__ == "__main__":
    import sys
    import os
    import glob
    import json
    
    def process_folder(folder_path, task_name):
        # Extract model name
        model_name = os.path.basename(folder_path)
        
        # Find all files starting with samples_task_name_
        sample_files = glob.glob(os.path.join(folder_path, f"samples_{task_name}_*.jsonl"))
        
        # Initialize QwenRewardModel
        reward_model = QwenRewardModel(batch_size=1)
        
        # Processed data organized by problem
        organized_data = {}
        
        for file_path in sample_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            if not data:
                print(f"Warning: No data in file {file_path}")
                continue
                
            for item in data:
                if "doc" not in item or "resps" not in item or not item["resps"]:
                    print(f"Warning: Incorrect data item format: {item}")
                    continue
                    
                problem = item["doc"]["problem"]
                standard_answer = item["doc"]["answer"]
                doc_id = item["doc_id"]
                
                if not item['resps'][0]:
                    print(f"Warning: Empty response list, doc_id: {doc_id}")
                    continue
                
                # Initialize data structure for the problem
                if doc_id not in organized_data:
                    organized_data[doc_id] = {
                        "doc_id": doc_id,
                        "problem": problem,
                        "standard_answer": standard_answer,
                        "responses": []
                    }
                
                # Collect all answers for current document
                question = [problem]
                answers = item['resps'][0]  # This is a list of answers


                print(f"Number of questions: {len(question)}")
                print(f"Number of responses: {len(answers)}")

                try:
                    # Batch calculate scores
                    scores = reward_model.score(question, [answers])

                    print(f"Score dimensions: {len(scores)}x{len(scores[0])}")
                    
                    # Process scores for each answer
                    for resp_idx, (response, score) in enumerate(zip(answers, scores[0])):
                        # Check if answer is valid
                        if not is_valid_response(response):
                            orm_score = 0.0
                            prm_score = 0.0
                            all_step_scores = [0.0]
                        else:
                            # Extract scores
                            orm_score = float(score[-1]) if score else 0
                            prm_score = float(np.mean(score)) if score else 0
                            all_step_scores = [float(s) for s in score]
                        
                        # Add to organized data
                        response_data = {
                            "response": response,
                            "resp_idx": resp_idx,
                            "orm_score": orm_score,
                            "prm_score": prm_score,
                            "all_step_scores": all_step_scores
                        }
                        
                        organized_data[doc_id]["responses"].append(response_data)
                        print(f"Doc ID: {doc_id}, Response Index: {resp_idx}, ORM score: {orm_score}, PRM score: {prm_score}")
                
                except Exception as e:
                    print(f"Error processing data, doc_id: {doc_id}, error: {str(e)}")
                    # Set scores to 0 when error occurs
                    for resp_idx, response in enumerate(answers):
                        response_data = {
                            "response": response,
                            "resp_idx": resp_idx,
                            "orm_score": 0.0,
                            "prm_score": 0.0,
                            "all_step_scores": [0.0]
                        }
                        organized_data[doc_id]["responses"].append(response_data)


        
        # Convert to list format for output
        output_data = list(organized_data.values())
        
        # Save processed results
        output_file = os.path.join(folder_path, f"QwenReward72B_{task_name}_prm_scored.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processing completed, results saved to: {output_file}")
        return output_file

    if len(sys.argv) != 3:
        print("Usage: python batch_reward_corrected.py <folder_path> <task_name>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    task_name = sys.argv[2]
    process_folder(folder_path, task_name)