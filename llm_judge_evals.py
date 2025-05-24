import pickle
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from collections import defaultdict

# Initialize API client with secure credential handling
llm_client = OpenAI(api_key="##")  # Replace with actual API key

 
EVAL_PROMPT = """You are an expert evaluating friction interventions in collaborative problem-solving.

Context: Participants test the Wason Card task rule: "All cards with vowels have an even number on the other side." A common mistake is checking only one direction—ignoring whether odd-numbered cards might have vowels, leading to false validation.

Example friction:
"When we say 'all cards with vowels have even numbers,' does that tell us anything about what's on the other side of number cards?"

A friction intervention is an indirect prompt for self-reflection, like asking "Are we sure?" or suggesting a step review.

You must rate each intervention (between 1 to 5) along these **dimensions** given the json format below. 

1. Relevance: How well does the intervention address key issues or assumptions in the reasoning process?
2. Gold Alignment: How well does the friction intervention align with the golden friction sample? 
3. Actionability: Does the friction intervention provide actionable guidance or suggest concrete steps for participants to improve their reasoning?
4. Rationale Fit: How well does the provided rationale align with the preference for the friction intervention?
5. Thought-Provoking: Encourages self-reflection
6. Specificity: Does the intervention pinpoint specific flaws, assumptions, or gaps?
7. Impact: To what extent does the friction intervention have the potential to change the course of the participants' reasoning and lead them to a more accurate or justified conclusion?
 

Finally, you must a choice between which of two interventions is more preferable and provide two-sentence explanation at the end. 

<Dialogue>: {}
<Gold interventions>: {}
<Intervention A>: {}
<Rationale A>: {}
<Intervention B>: {}
<Rationale B>: {}

{{
  "A": {{
    "relevance": <1-5>,
    "gold_alignment": <1-5>,
    "actionability": <1-5>,
    "rationale_fit": <1-5>,
    "thought_provoking": <1-5>,
    "specificity": <1-5>,
    "impact": <1-5>,
 
  }},
  "B": {{
    "relevance": <1-5>,
    "gold_alignment": <1-5>,
    "actionability": <1-5>,
    "rationale_fit": <1-5>,
    "thought_provoking": <1-5>,
    "specificity": <1-5>,
    "impact": <1-5>,
 
  }},
  "winner": {{
    "friction_sample": <"A" or "B">,
    "rationale": <one sentence explanation>
  }}
}}"""

 # Configuration parameters
BATCH_SAVE_INTERVAL = 400
EVALUATION_MODES = ['direct', 'reverse']
RESULT_PREFIX = "pairwise_preference_evals/gpt_evaluations"

# Load input data
input_data = pd.read_csv("##")  # Replace with actual file path of sampled completions from baselines
datasets = [input_data, input_data]   

def format_evaluation_prompt(context, gold_standard, intervention_a, 
                           rationale_a, intervention_b, rationale_b):
    """Construct evaluation prompt with given parameters"""
    return EVALUATION_TEMPLATE.format(
        context, gold_standard, intervention_a, 
        rationale_a, intervention_b, rationale_b
    )

def save_results(data_store, batch_num, evaluation_mode):
    """Save evaluation results to disk"""
    filename = f"{RESULT_PREFIX}/batch_{batch_num}_{evaluation_mode}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(data_store, file)

def process_dataset(dataset, evaluation_mode):
    """Process dataset in specified evaluation mode"""
    evaluations = defaultdict(list)
    
    for idx, record in tqdm(dataset.iterrows(), total=len(dataset)):
        try:
            dialog = record['context']
            gold_intervention = record['gold_chosen_friction']
            
            # Handle missing data
            policy_intervention = record.get('friction_model1', "No intervention")
            policy_rationale = record.get('rationale_model1', "No rationale")
            reference_intervention = record.get('friction_model2', "No intervention")
            reference_rationale = record.get('rationale_model2', "No rationale")
            
            # Order interventions based on evaluation mode
            if evaluation_mode == 'direct':
                prompt = format_evaluation_prompt(dialog, gold_intervention,
                                                 policy_intervention, policy_rationale,
                                                 reference_intervention, reference_rationale)
            else:
                prompt = format_evaluation_prompt(dialog, gold_intervention,
                                                 reference_intervention, reference_rationale,
                                                 policy_intervention, policy_rationale)
            
            # Get LLM evaluation
            response = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Store response
            model_comparison = f"{record['model_model1']}_{record['model_model2']}_{record['temperature']}"
            evaluations[model_comparison].append(response.choices[0].message.content)
            
            # Periodic saving
            if idx % BATCH_SAVE_INTERVAL == 0 and idx > 0:
                save_results(evaluations, idx//BATCH_SAVE_INTERVAL, evaluation_mode)
                evaluations.clear()
                
        except Exception as error:
            print(f"Error processing record {idx}: {str(error)}")
            evaluations.setdefault('errors', []).append(str(error))
    
    # Save final results
    save_results(evaluations, 'final', evaluation_mode)

# Execute processing pipeline
for data_chunk, eval_mode in zip(datasets, EVALUATION_MODES):
    print(f"\nProcessing in {eval_mode} evaluation mode")
    process_dataset(data_chunk, eval_mode)