# FAAF Implementation Files

## Main Files
* `faaf_config.py` - Configuration and hyperparameter settings for FAAF training
  * Includes definitions for FAAF ablation losses

* `faaf_main_training_file.py` - Main training script for FAAF
  * Supports DPO (loss_type="sigmoid") and IPO (loss_type="ipo") baselines
  * Controls hyperparameter sweeps and logging

* `faaf_trainer.py` - Core FAAF trainer implementation
  * Handles loss computation 
  * Manages phi-unconditioned forward passes
  * Implements preference alignment

* `llm_judge_evals.py` - LLM evaluation pipeline
  * Runs pairwise position-swapped evaluations
  * Generates results for Table 1 comparisons

* `opt_reward_modeling.py` - Reward modeling implementation
  * Uses OPT models for reward computation
  * Supports PPO training and Table 2 evaluations

* `ppo_baseline_training.py` - PPO baseline implementation
* `friction_agent_inference.py` - FAAF model inference
  * Handles generation and parsing
  * Computes evaluation metrics

## Usage

## Requirements
Install dependencies: `pip install -r requirements.txt`
1. Run training through `faaf_main_training_file.py`
   * Uses configs from `faaf_config.py`
   * Implements FAAF trainer from `faaf_trainer.py`

2. For baselines:
   * Use `opt_reward_modeling.py` for reward modeling
   * Use `ppo_baseline_training.py` for PPO comparison

3. For evaluation:
   * Run `friction_agent_inference.py` for model generations
   * Use `llm_judge_evals.py` for preference scoring
