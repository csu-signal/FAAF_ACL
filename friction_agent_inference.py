
import os
import pandas as pd
from collections import defaultdict
from collections.abc import Mapping
import wandb
 

try:
   wandb.init(
       project="friction_agent_inference",
       name=f"friction_inference_{time.strftime('%Y%m%d_%H%M%S')}",
    
       settings=wandb.Settings(
           _service_wait=300,
           start_method="thread"
       )
   )
except Exception as e:
   print(f"Failed to initialize wandb: {e}")
   # Optionally disable wandb
   os.environ["WANDB_DISABLED"] = "true"


from safetensors import safe_open
from datasets import Dataset,load_dataset, DatasetDict
from datasets import load_from_disk
import re
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict
from itertools import combinations
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)





@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    per_device_eval_batch_size: Optional[int] = field(default=10, metadata={"help": "eval batch size per device"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
 
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )
 
 
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def create_preference_dataset_friction(data):
    """
    Creates the simplest preference dataset where original friction is chosen 
    and lowest scored relevant statement is rejected
    """
    pairs = []
    
    for key, entry in data.items():
        original_friction = entry['friction_data_original']['friction_statement']
        original_rationale = entry['friction_data_original']['rationale']
        relevant_statements = entry['gpt_friction_rogue_rankings']['relevant_statements']
        
        # Find the lowest scored relevant statement
        lowest_relevant = min(relevant_statements, key=lambda x: x['relevance_score'])
        
        pair = {
            'chosen': original_friction,
            'rejected': lowest_relevant['statement'],
            'chosen_score': 11,  # Assigning higher score to original
            'rejected_score': lowest_relevant['relevance_score'],
            'context': entry['friction_data_original']['previous_utterance_history'],
            'task_state': entry['friction_data_original']['task_summary'],
            'belief_state': entry['friction_data_original']['belief_summary'],
           'dialog_id': entry['friction_data_original']['dialog_id'],
            'friction_participant': entry['friction_data_original']['friction_participant'],
          'chosen_rationale': original_rationale,
            'rejected_rationale': lowest_relevant['rationale'],
            'rationale_present': entry['friction_data_original']['rationale_present']
        
        }
        pairs.append(pair)
    
    return Dataset.from_pandas(pd.DataFrame(pairs))

def create_train_test_split(dataset, test_size=0.1, seed=42):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        'train': split['train'],
        'test': split['test']
    })

# Function to count occurrences of tags in a string
def count_tags(text, tags):
    tag_counts = defaultdict(int)
    for tag in tags:
        tag_counts[tag] += len(re.findall(re.escape(tag), text))
    return tag_counts

# Function to parse content within specific tags: gets friction intervention after model.generate (on newly generated tokens)
def parse_tags(text, tags):
    parsed_data = {tag: [] for tag in tags}
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
        parsed_data[tag].extend(matches)
    return parsed_data

# Function to handle friction tag logic
def handle_friction_logic(text):
    '''
    This function processes a text string to extract or construct a "friction" snippet by:

    Returning the text following a <friction> tag if present, unless a closing </friction> tag is found.
    If no <friction> tags exist, it constructs a snippet by extracting the first, second-to-last, 
    and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
    
    '''
    if "<friction>" not in text and "</friction>" not in text:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        elif sentences:
            return " ".join(sentences)
        else:
            return ""
    elif "<friction>" in text and "</friction>" not in text:
        friction_start = text.find("<friction>") + len("<friction>")
        return text[friction_start:].strip()
    else:
        return ""  # Friction is complete, no need to handle further



def process_data_template(example):

    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
    "Finally, generate a nuanced friction statement in a conversational style based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a friction statement is needed. This monologue must be enclosed between the "
    "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
    "- Incorrect assumptions\n"
    "- False beliefs\n"
    "- Rash decisions\n"
    "- Missing evidence.\n\n"
    "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
    "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
)
 

    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)



    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }


    return text 

def process_wtd_simulated_dataset(dataset, split, 
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24):

    if sanity_check:
        dataset = dataset.shuffle().select(range(1000))
    print(dataset)
 
    original_columns = dataset.column_names
    selected_columns = ['prompt', 'chosen', 'rejected']
    # removable_columns = [ x for x in original_columns if x not in selected_columns]
    def return_prompt_and_responses(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

 

    processed_dataset = dataset.map(
        return_prompt_and_responses,
  
        num_proc= num_proc,
     
    )
    # Filter the dataset based on prompt and response length constraints
    print(f"Filtered dataset before length constraint (train): {len(processed_dataset)}")
    print(f"Getting prompt-length stats BEFORE filtering:")
    # analyze_lengths(processed_dataset, split = split)

    filtered_dataset = processed_dataset.filter(
        lambda x: (len(x["chosen"]) <= script_args.max_length) and 
                (len(x["rejected"]) <= script_args.max_length)
    )
    print(f"Filtered dataset after length constraint (train): {len(filtered_dataset)}")
    print(f"Getting prompt-length stats AFTER filtering:")
    # analyze_lengths(filtered_dataset)
 
    print("Sample of filtered dataset (train):")
    print(filtered_dataset[:2])  # Show sample from filtered dataset
    return filtered_dataset
    

def load_model_and_tokenizer(checkpoint_path):
    """
    Loads SFT model and tokenizer for training and evaluation (locally for now)
    using flash attention 2 and return them
    """
    print("checkpoint_path", checkpoint_path)
    if not any(keyword in checkpoint_path for keyword in ['sft', 'dpo']):
        model_kwargs = dict(
#         use_cache=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype = torch.bfloat16,
        device_map='cuda'
    )

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.pad_token = tokenizer.eos_token  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        return model, tokenizer
    else:
        # Handle 'sft' or 'dpo' models
        if 'baselm' in checkpoint_path.split("/")[-1].split("_")[-1]:

            base_model_dir = "/".join(checkpoint_path.split("/")[:-1])
            print("base_model_dir", base_model_dir)
            
            lora_model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,  # Local path
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

            # lora_model = AutoPeftModelForCausalLM.from_pretrained(base_model_dir, device_map="auto", torch_dtype=torch.bfloat16)
            lora_merged_model = lora_model.merge_and_unload()

            output_merged_dir = os.path.join(base_model_dir, "final_merged_checkpoint")
            # lora_merged_model.save_pretrained(output_merged_dir, safe_serialization=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            tokenizer.padding_side = 'right'
            return lora_merged_model, tokenizer

        else:   
            model_kwargs = dict(
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map='cuda'
            )
            
            # Determine the appropriate directory for the model
            base_model_dir = "/".join(checkpoint_path.split("/")[:-1])
            print("base_model_dir:", base_model_dir)
            output_merged_dir = (
                os.path.join(base_model_dir, "sft_merged_checkpoint") 
                if 'sft' in checkpoint_path 
                else os.path.join(base_model_dir, "final_merged_checkpoint")
            )
            print(f"loaded model from: {output_merged_dir}")
            # Load the specific model checkpoint
            model = AutoModelForCausalLM.from_pretrained(output_merged_dir, **model_kwargs)
    
            tokenizer_path = "/".join(output_merged_dir.split("/")[:-1])
            print("tokenizer_path", tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            # tokenizer.pad_token = tokenizer.eos_token  # use unk rather than eos token to prevent endless generation
            tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            tokenizer.padding_side = 'right'
            return model, tokenizer

    
def load_model_and_tokenizer_2(checkpoint_path):
    base_model_dir = "/".join(checkpoint_path.split("/")[:-1])
    print("base_model_dir", base_model_dir)
    
    lora_model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path,  # Local path
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

    # lora_model = AutoPeftModelForCausalLM.from_pretrained(base_model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    lora_merged_model = lora_model.merge_and_unload()

    output_merged_dir = os.path.join(base_model_dir, "final_merged_checkpoint")
    # lora_merged_model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return lora_merged_model, tokenizer

def load_tokenizer_only(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    return tokenizer


def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]



def generate_multiple_sequences_with_intrinsic_metrics(model, tokenizer, prompts, generation_args, device, strategy="beam_search", batched=False):
    if batched:

        tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'

        # tokenizer.pad_token = tokenizer.eos_token  # Use eos_token for pad token #made this change for llama3 only for ppo inference
        # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # tokenizer.padding_side = 'left'
        cleaned_prompts = [prompt.replace("\n", " ") for prompt in prompts]
        inputs = tokenizer(cleaned_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        if strategy == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                num_beams=generation_args["num_beams"],
                num_return_sequences=generation_args["num_return_sequences"],
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_k_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_k=generation_args["top_k"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                min_length=generation_args["min_length"],
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_p_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        else:
            raise ValueError("Unsupported strategy. Use 'beam_search', 'top_k_sampling', or 'top_p_sampling'.")

        # Compute transition scores
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices if strategy == "beam_search" else None, normalize_logits=True
    )

    # Decode the generated tokens for each prompt in the batch
    generated_texts = []
    all_generated_texts = []
    transition_scores_dict = {}
    perplexities_dict = {}
    metrics_dict = {}

    for i in range(0, len(outputs.sequences), generation_args["num_return_sequences"]):
        prompt_texts = []
        batch_transition_scores = []
        batch_perplexities = []
        batch_metrics = []
        prompt_only = []
        for j in range(generation_args["num_return_sequences"]):
            sequence_index = i + j  # Global index for the current sequence
            output = outputs.sequences[sequence_index]
            prompt_length = input_ids.shape[-1]  # Length of the input prompt
            new_tokens = output[prompt_length:]  # Get only the generated tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = output[:prompt_length]
            prompt_text =  tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            # Align logits with the generated tokens for the current sequence
            # Align logits with the generated tokens for the current sequence
            # logits = torch.stack(outputs.scores, dim=0)  # Shape: (max_new_tokens, num_return_sequences, vocab_size)

            # Align logits with the generated tokens for the current sequence
            logits = torch.stack(outputs.scores, dim=0).cpu().numpy()  # Move logits to CPU for NumPy compatibility
                                    # Use log_softmax for stability
            log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).cpu().numpy()

            # Convert log_probs to probabilities
            probs = np.exp(log_probs)

            # Ensure numerical stability and clamp probabilities
            probs = np.clip(probs, 1e-10, 1.0)

            # Compute metrics
            token_ids = new_tokens.cpu().numpy()  # Ensure new_tokens is on CPU
            probs_for_sequence = probs[:, sequence_index, :]
            log_probs_for_sequence = log_probs[:, sequence_index, :] 
            nll = -np.sum(np.log(probs_for_sequence[np.arange(len(token_ids)), token_ids])) / len(token_ids)
            nll_perplexity = np.exp(nll)
            predictive_entropy = -np.sum(probs_for_sequence * np.log(probs_for_sequence), axis=-1).mean()
            expected_conditional_entropy = -np.mean(np.log(probs_for_sequence[np.arange(len(token_ids)), token_ids]))
            mutual_information = max(predictive_entropy - expected_conditional_entropy, 0.0)
          
            # Transition-based perplexity (if available)
            scores = transition_scores[i + j]  # Transition scores from `generate`
            log_probs_transition = transition_scores[j].cpu().numpy()  # Transition scores for the current sequence
            log_probs_transition = np.clip(log_probs_transition, a_min=-1e10, a_max=None)
            perplexity = np.exp(-np.mean(log_probs_transition))
            print(f"Predictive Entropy: {predictive_entropy}")
            print(f"Expected Conditional Entropy: {expected_conditional_entropy}")
            print(f"Mutual Information: {mutual_information}")

            print(f"NLL for sequence {sequence_index}: {nll}")
            print(f"Predictive entropy for sequence {sequence_index}: {predictive_entropy}")
            print(f"Mutual information for sequence {sequence_index}: {mutual_information}")
            print(f"Perplexity for sequence {sequence_index}: {perplexity}")
            print(f"NLL Perplexity for sequence {sequence_index}: {nll_perplexity}")

            # Save the first set of metrics
            batch_metrics.append({
                "nll": nll,
                "predictive_entropy": predictive_entropy,
                "mutual_information": mutual_information,
                "perplexity": perplexity,
                "expected_conditional_entropy": expected_conditional_entropy,
                "nll_perplexity": nll_perplexity
            })

            # Metrics computation for the second set using log probabilities
            token_log_probs = log_probs_for_sequence[np.arange(len(token_ids)), token_ids]
            token_log_probs = np.nan_to_num(token_log_probs)  # Handle NaN values

            nll_1 = -np.sum(token_log_probs) / len(token_ids)
            nll_perplexity_1 = np.exp(nll_1)
            predictive_entropy_1 = -np.sum(probs_for_sequence * log_probs_for_sequence, axis=-1).mean()
            expected_conditional_entropy_1 = -np.mean(token_log_probs)
            mutual_information_1 = max(predictive_entropy_1 - expected_conditional_entropy_1, 0.0)

            # Save the second set of metrics
            batch_metrics.append({
                "nll_1": nll_1,
                "predictive_entropy_1": predictive_entropy_1,
                "mutual_information_1": mutual_information_1,
                "perplexity_1": perplexity,
                "expected_conditional_entropy_1": expected_conditional_entropy_1,
                "nll_perplexity_1": nll_perplexity_1
            })
             # Debugging values
           
            batch_transition_scores.append((new_tokens, scores))
            prompt_texts.append(generated_text)
            prompt_only.append(prompt_text)

        generated_texts.append(prompt_texts)
        all_generated_texts.extend(prompt_only)
        transition_scores_dict[i // generation_args["num_return_sequences"]] = batch_transition_scores
        perplexities_dict[i // generation_args["num_return_sequences"]] = batch_perplexities
        metrics_dict[i // generation_args["num_return_sequences"]] = batch_metrics

    return generated_texts, all_generated_texts, transition_scores_dict, perplexities_dict, metrics_dict

import random
from datasets import Dataset
import pandas as pd

def select_samples(dataset, true_ratio=0.1, num_samples=100, seed=42):
    """
    Select a subset of samples from the dataset, ensuring a specific ratio for `original_wtd_friction_data`.

    Parameters:
    - dataset (Dataset): HuggingFace Dataset object
    - true_ratio (float): Ratio of samples where `original_wtd_friction_data` is True
    - num_samples (int): Total number of samples to select
    - seed (int): Random seed for reproducibility

    Returns:
    - Dataset: A subset of the original dataset with the desired ratio
    """
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Clean the data: Replace None with False
    df['original_wtd_friction_data'] = df['original_wtd_friction_data'].fillna(False)

    # Split into True and False subsets
    true_subset = df[df['original_wtd_friction_data'] == True]
    false_subset = df[df['original_wtd_friction_data'] == False]

    # Compute the number of samples to select from each group
    num_true = int(num_samples * true_ratio)
    num_false = num_samples - num_true

    # Check if there are enough samples in each subset
    if len(true_subset) < num_true:
        print(f"Not enough True samples. Requested: {num_true}, Available: {len(true_subset)}")
        num_true = len(true_subset)  # Adjust to available samples
        num_false = num_samples - num_true  # Adjust the other subset accordingly
    
    if len(false_subset) < num_false:
        print(f"Not enough False samples. Requested: {num_false}, Available: {len(false_subset)}")
        num_false = len(false_subset)  # Adjust to available samples
        num_true = num_samples - num_false  # Adjust the other subset accordingly

    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly sample from each group
    true_samples = true_subset.sample(n=num_true, random_state=seed)
    false_samples = false_subset.sample(n=num_false, random_state=seed)

    # Combine the two subsets
    selected_samples = pd.concat([true_samples, false_samples]).sample(frac=1, random_state=seed)  # Shuffle

    # Convert back to HuggingFace Dataset
    return Dataset.from_pandas(selected_samples), df

 
if __name__ == "__main__":
    # Define arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set seed for reproducibility
    set_seed(script_args.seed)
 
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

#     # Initialize the accelerator
#     accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    friction_data = load_from_disk("all_samples_friction_for_sft")
    train_dataset = friction_data['train']
    test_dataset = friction_data['test']
    print(f"Test samples: {len(friction_data['test'])}")
    selected_validation_dataset, _ = select_samples(test_dataset, true_ratio=0.1, num_samples=500, seed=42)
    print(f"selected_validation_dataset samples: {len(selected_validation_dataset)}")
    selected_validation_dataset.save_to_disk("selected_validation_dataset_friction_mixed_with_original")


    dummy_eval_dataset = selected_validation_dataset.map(
    process_data_template,
    remove_columns=selected_validation_dataset.column_names  # Removes all other columns except the returned ones
)

    print("size of WTD after processing and filtering: dummy_eval_dataset", dummy_eval_dataset)
    seed = 42

 

    models_list = # list of trained models to run inference on!!

    batch_size = script_args.per_device_eval_batch_size
    results = []
    results_dict = defaultdict(list)
    generation_args = {
            "max_new_tokens": 356,
            "temperature": 0.7,# for diversity of samples 
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9
            "num_beams": 5,
            "min_length": 100,
        'num_return_sequences':1
        }

    temperature_list = [0.7]
    batches = create_batches(dummy_eval_dataset['prompt'], batch_size)
    # Tags to search for
    tags_for_parsing = ["friction", "rationale", "t", "b"] # <t> task-state </t> and <b> belief-states </b> --> these demarcators are used to parse the task and belief states
    
    wandb_table = wandb.Table(columns=["Prompt", "Friction", "Task State", "Belief State", "Rationale"])
    parsed_data_list = []
    # run loop to get inferenc on batches of prompts using the list of models 
    for index, model in enumerate(tqdm(models_list)):
        #Load model 
        # model_name = model
        model_name = model.split("/")[-1]
        model_path = model
        # evalgenerations_directory = f"{model}_evalgenerations"
        evalgenerations_directory = script_args.output_dir
        os.makedirs(evalgenerations_directory, exist_ok=True)
        model_folder = model  # get the inference model names here 
        if not any(keyword in model for keyword in ['sft', 'dpo']):
            model_path = model
            print("Base model path:", model_path)
        else:

            model_path = os.path.join(current_dir, model_folder)
            print("SFT or DPO model path:", model_path)
        # model, tokenizer = load_model_and_tokenizer(model_path)
        model, tokenizer = load_model_and_tokenizer_2(model_path) # for those checkpoints witout merged models
        print("LOADING MODEL", model,)

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Re-create batches for each model to ensure fresh data each time
        batches = list(create_batches(dummy_eval_dataset['prompt'], batch_size))
        # get argmax and top-p and top-k sampling loop
        # Iterate through the generated batches
        for batch_index, batch in enumerate(tqdm(batches)):
            for temperature in temperature_list:
                generation_args['temperature'] = temperature
                print("getting responses at diversity temperature:", generation_args['temperature'])
                # outputs, prompts, transition_scores_dict, perplexities_dict = generate_multiple_sequences(model, tokenizer, batch, generation_args, device, strategy="top_p_sampling", batched=True)
             
                generated_texts, all_generated_texts, transition_scores_dict, perplexities_dict,\
                 metrics_dict = generate_multiple_sequences_with_intrinsic_metrics(model, tokenizer, batch, generation_args, device, strategy="top_p_sampling", batched=True)
                # print("transition_scores_dict", transition_scores_dict)
                print("perplexities_dict", perplexities_dict)
                print("metrics_dict", metrics_dict)
                            # Collect results
                for idx, (output, prompt) in enumerate(zip(generated_texts, all_generated_texts)):
                    # Extract transition scores, perplexities, and metrics from their respective dictionaries
                    batch_transition_scores = transition_scores_dict.get(idx, [])
                    batch_perplexities = perplexities_dict.get(idx, [])
                    batch_metrics = metrics_dict.get(idx, [])
                    print("output friction full generation excluding prompt-->",output )
                    parsed_frictive_states_and_friction = parse_tags(output[0], tags_for_parsing) 
                    friction_intervention = ' '.join(parsed_frictive_states_and_friction['friction'])
                    if not friction_intervention:
                        friction_intervention = handle_friction_logic(output[0]) # if friction tags are not present, parse with some logic (see function)
                    task_state = ' '.join(parsed_frictive_states_and_friction['t'])
                    belief_state = ' '.join(parsed_frictive_states_and_friction['b'])
                    rationale = ' '.join(parsed_frictive_states_and_friction['rationale'])
                    
                    print("task_state-->",task_state)
                    print("belief_state-->",belief_state)
                    print("rationale-->",rationale)
                    print("friction_intervention-->",friction_intervention)
                    
                    # Append results to the dictionary
                    results_dict[(model_name, temperature)].append({
                        "output": output,
                        "prompt": prompt,
                        # "transition_scores": batch_transition_scores,
                        # "perplexities": batch_perplexities,
                        "metrics": batch_metrics
                    })

                     # Append parsed data to the list
                    parsed_data_list.append({
                        "Prompt": prompt,
                        "Friction Intervention": friction_intervention,
                        "Task State": task_state,
                        "Belief State": belief_state,
                        "Rationale": rationale,
                       
                    })

                parsed_df = pd.DataFrame(parsed_data_list)
                wandb_table = wandb.Table(dataframe=parsed_df)
                wandb.log({"Friction Interventions Table": wandb_table})

           
            # Save intermediate results every 30 batches
            if batch_index % 30 == 0:
                print(f"saving batched filename:{batch_index}")
                batched_filename = f"friction_{batch_index}.pkl"
                pickle.dump(results_dict, open(f'{evalgenerations_directory}/{model_name}_evalgenerations_{batch_index}.pkl', 'wb'))

        # Save results for the current model
        pickle.dump(results_dict, open(f'{evalgenerations_directory}/{model_name}_evalgenerations.pkl', 'wb'))

    # Save all results from all models
    pickle.dump(results_dict, open(f'{evalgenerations_directory}/all_model_generations.pkl', 'wb'))
# Finish wandb logging
    wandb.finish()
 
