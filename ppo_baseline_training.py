# Standard library imports
import os
import sys
import re
import math
import json
import copy
import random
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from enum import Enum
from functools import partial, wraps

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import bitsandbytes as bnb
import wandb

# Hugging Face Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollator,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import EvalLoopOutput, TrainOutput, speed_metrics
from transformers.trainer_callback import TrainerCallback

# Dataset and data handling
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from trl.trainer.utils import RewardDataCollatorWithPadding

# TRL and PEFT
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
    RewardConfig,
    RewardTrainer,
)
from peft import (
    LoraConfig, 
    PeftModel, 
    prepare_model_for_kbit_training, 
    AutoPeftModelForCausalLM
)

# Accelerate
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object, is_deepspeed_available

# Evaluation and metrics
from evaluate import load as load_metric
from rouge_score import rouge_scorer

# Debug settings
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Initialize wandb
wandb.init(project="ppo_training")

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: Optional[str] = field(default="DELI_sft_weights/checkpoint-6000", metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default="llama3_8b_instruct", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    rm_adapter: Optional[str] = field(
        default="trl-lib/llama-7b-hh-rm-adapter", metadata={"help": "the rm adapter name"}
    )
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=False, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


 
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type you want to test your policy on"})

    per_device_train_batch_size: Optional[int] = field(default=12, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=5, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    dataset: Optional[str] = field(default="ultrafeedback_binarized", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=20, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    save_strategy: Optional[str] = field(default="no", metadata={"help": "whether to save intermediate steps during training"})
  
 
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )



class FrictionPPO_trainer(PPOTrainer):
    """
    The DensePPOTrainer uses Proximal Policy Optimization to optimize language models specifically for dense embeddings.

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head. Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face transformer model with a casual language modeling head. Check the documentation of `PreTrainedModelWrapper` for more details. If no reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the data. Check the documentation of `transformers.PreTrainedTokenizer` and `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be created outside the trainer. Users need to design their own dataloader and ensure the batch size used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and passed along the dataloader.
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[typing.Callable] = None,
    ):
        """
        Initialize DensePPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modeling head. Used for KL penalty.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer.
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset will be preprocessed by removing the columns that are not used by the model. If none is passed, a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function that is going to be used for `prepare_dataloader` method. Note this collator is different from the one we use for training. Pass a valid `training_data_collator` instead.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared. Used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
            training_data_collator (Optional[function]):
                Custom data collator used for training.
        """
        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            num_shared_layers=num_shared_layers,
            lr_scheduler=lr_scheduler,
            training_data_collator=training_data_collator,
        )
        # Assign the dataset to an instance variable if it's passed in
        if dataset is not None:
            self.dataset = dataset
            print("Dataset is initialized:", self.dataset)
        else:
            self.dataset = None
            print("No dataset provided.")
        
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """

        def custom_data_collator(batch):
            # Include 'dialogue_context' in the collated batch if needed
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'label': torch.stack([item['label'] for item in batch]),
                'query': [item['query'] for item in batch],
                'golden_friction': [item['golden_friction'] for item in batch],
                'dialogue_context': [item['dialogue_context'] for item in batch],  # Make sure this is included
            }

        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        print("Dataset columns before creating dataloader and collator", dataset.column_names)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += ["label", "query", "response", "dialogue_context", "golden_friction"]
            print("self signature cols for ppo", self._signature_columns)

    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        print("ignored cols",ignored_columns )

        columns = [k for k in signature_columns if k in dataset.column_names]
        print("dataset columns here", dataset.column_names)
        # if version.parse(datasets.__version__) < version.parse("1.4.0"):
        #     dataset.set_format(
        #         type=dataset.format["type"],
        #         columns=columns,
        #         format_kwargs=dataset.format["format_kwargs"],
        #     )
        #     return dataset
        # else:
        return dataset.remove_columns(ignored_columns) 
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size
        # self.current_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
 
@dataclass
class ScriptArguments_2:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    
 
    student_model_name_or_path: Optional[str] = field(
        default="facebook/opt-1.3b",
        metadata={"help": "the location of the SFT model name or the student model or path"},
    )
 

 
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type you want to test your policy on"})

    per_device_train_batch_size: Optional[int] = field(default=12, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=5, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    dataset: Optional[str] = field(default="ultrafeedback_binarized", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=20, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    save_strategy: Optional[str] = field(default="no", metadata={"help": "whether to save intermediate steps during training"})
  
 
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

        
        
def transform_and_assign_preferences(example):
    """Prepare the friction generation prompt and response for SFT training."""
    
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. "
    "Your task is to generate nuanced friction statements within a dialogue. "
    "Given the **dialogue history** involving three participants and the *game details*, "
    "generate a <friction> statement that acts as indirect persuasion. This statement should "
    "encourage the participants to reevaluate their beliefs and assumptions about the task. "
    "Additionally, provide a <rationale> or explanation for your friction statement. Base your reasoning "
    "on evidence from the dialogue, focusing on elements such as: "
    "- Incorrect assumptions "
    "- False beliefs "
    "- Rash decisions "
    "- Missing evidence ")

    friction_definition_game_definition_prompt_rm = (
        "*Game details and ground-truth*: The game is called 'Game of Weights.' The participants (P1, P2, and P3) are "
        "trying to determine the weight of various blocks. The blocks are of different colors and have specific weights in grams: "
        "the red block is 10 grams, the blue block is 10 grams, the green block is 20 grams, the purple block is 30 grams, and "
        "the yellow block is 50 grams. At the start of the game, participants are only allowed to weigh two blocks at a time, "
        "and they are told the weight of the red block. The participants must figure out how to determine the weight of each block. "
        "At the beginning of the game, they are unaware of the actual weights. Additionally, we inform the participants that they "
        "don’t need to use the scale's slider. The actual reason is that the blocks are in increments of 10 grams. "
        "The **dialogue history** is given below: "
    )
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication."
    "Your task is to generate nuanced friction statements within a dialogue."
   )

    
 
    prompt = (system_prompt_rm + example['context']).replace('\n', ' ')
 
    chosen_response_format = f"Answer: <friction> {example['chosen']}. <rationale>: {example['chosen_rationale']}"
    rejected_response_format = f"Answer: <friction> {example['rejected']}. <rationale>: {example['rejected_rationale']}"

    chosen_response = [
        {'content': prompt, 'role': 'user'},
        {'content': chosen_response_format, 'role': 'assistant'}
    ]

    # Format the rejected response
    rejected_response = [
        {'content': prompt, 'role': 'user'},
        {'content': rejected_response_format, 'role': 'assistant'}
    ]

    # Return the new structure with feedback weights
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response,
    }


def transform_and_assign_preferences_2(example):
    """Prepare the friction generation prompt and response for SFT training."""
 
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
    "Finally, generate a nuanced friction statement based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a friction statement is needed. This rationale must be enclosed between the "
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

    # "Be specific and ensure that your response clearly addresses the dynamics in the dialogue.")

    # text = f"Question: {system_prompt_rm} {friction_definition_game_definition_prompt_rm}. {example['context']}\n\nAnswer: <friction> {example['chosen']}. <rationale>: {example['chosen_rationale']}" # old sft prompt format
    # the below prompt is formatted acc to llama3 instruction format from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14
    # this excludes the game definition prompt since
    system_part = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|>"
    )

    user_part = (
    f"<|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['chosen_context']}\n\n" # context previously (silly change of naming)
    f"<|eot_id|>"
    )
    
    full_context_prompt = f"{system_part}\n{user_part}"
    # Chosen response format: rogue data processing currently appends chosen etc for the task and belief states
    chosen_response_format = (
    f"{system_part}\n{user_part}"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant: <t> {example['chosen_task_state']} </t>\n"
    f"        <b> {example['chosen_belief_state']} </b>\n"
    f"        <rationale>: {example['chosen_rationale']} </rationale>\n"
    f"        <friction> {example['chosen_friction_statement']} </friction>\n"
    f"<|eot_id|>"
    )

    # Rejected response format
    rejected_response_format = (
    f"{system_part}\n{user_part}"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant: <t> {example['rejected_task_state']} </t>\n"
    f"        <b> {example['rejected_belief_state']} </b>\n"
    f"        <rationale>: {example['rejected_rationale']} </rationale>\n"
    f"        <friction> {example['rejected_friction_statement']} </friction>\n"
    f"<|eot_id|>"
    )
 

    dialogue_context = example['chosen_context']
    return {
        'prompt': full_context_prompt,
        'chosen': chosen_response_format,
        'rejected': rejected_response_format,
        'dialogue_context':dialogue_context
    }


def transform_and_assign_preference_deli(example):
    """Prepare the friction generation prompt and response for SFT training."""
 
    # more intuitive friction generation system prompt: first predicts the task, then beliefs, then explian why friction is needed before generating the friction intervention
  
    system_prompt_rm = '''You are an expert in collaborative task analysis and reasoning.  
        Participants must test the rule: **"All cards with vowels have an even number on the other side."**  
        A common mistake is verifying only one direction—ignoring the need to check whether odd-numbered cards might have vowels. This incomplete reasoning risks false validation.  

        **Example friction:**  
        *"When we say 'all cards with vowels have even numbers,' does that tell us anything about what might be on the other side of number cards?"*  
        This prompts bidirectional reasoning to ensure both necessary and sufficient conditions are considered.  

        For each dialogue:  

        <belief_state>  
        Identify contradictions in understanding, reasoning, or assumptions.  
        </belief_state>  

        <rationale>  
        Explain why intervention is needed—what’s misaligned, its impact, and the expected resolution.  
        </rationale>  

        <friction>  
        Generate a friction intervention that fosters self-reflection, realigns understanding, and supports collaboration.  
        </friction>  
        '''

 
    system_part = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}\n\n"
    f"<|eot_id|>"
    )

    user_part = (
    f"<|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n" # context previously (silly change of naming)
    f"<|eot_id|>"
    )
    
    full_context_prompt = f"{system_part}\n{user_part}"
    # Chosen response format: rogue data processing currently appends chosen etc for the task and belief states
    chosen_response_format = (
    f"{system_part}\n{user_part}"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant: <b> {example['belief_state']}{example['contradiction_reason']} </b>\n"
    f"        <rationale>: {example['chosen_rationale']} </rationale>\n"
    f"        <friction> {example['chosen_friction']} </friction>\n"
    f"<|eot_id|>"
    )

    # Rejected response format
    rejected_response_format = (
    f"{system_part}\n{user_part}"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant: <b> {example['belief_state']}{example['contradiction_reason']} </b>\n"
    f"        <rationale>: {example['rejected_rationale']} </rationale>\n"
    f"        <friction> {example['rejected_friction']} </friction>\n"
    f"<|eot_id|>"
    )

    dialogue_context = example['context']
    return {
        'prompt': full_context_prompt,
        'chosen': chosen_response_format,
        'rejected': rejected_response_format,
        'dialogue_context':dialogue_context
    }
  
 

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

def build_friction_dataset_gpt2_full_context(train_data, config, max_length=1024, max_query_length=800):
    """
    Build dataset for training a GPT-2 model on friction classification using the full context up to "Answer: <friction>",
    and filter queries exceeding the max_query_length.

    Args:
        train_data (`list`): List of training samples containing context and friction statements.
        config (`object`): Configuration object containing model_name.
        max_length (`int`): Maximum token length for the input sequence.
        max_query_length (`int`): Maximum token length allowed for the query.

    Returns:
        dataset (`datasets.Dataset`): The filtered dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_friction_samples(sample):
        """
        Tokenizes and processes a sample into input, query, and label for training.

        Args:
            sample (`dict`): Dictionary containing context and friction statements.

        Returns:
            `dict`: Processed sample with tokenized input, query, and label.
        """
        # Extract context and response
        context = sample["chosen"][0]["content"]  # Dialogue context
        response = sample["chosen"][1]["content"]  # Friction response

        # Use full context up to "Answer: <friction>"
        input_sequence = context + " " + "Answer: <friction>" + response.split("Answer: <friction>")[0].strip()

        # Tokenize the input sequence
        input_ids = tokenizer.encode(input_sequence, truncation=True, max_length=max_length)

        # Decode the input back into a query for readability
        query = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Process label: "assistant" indicates relevant friction, "user" indicates less relevant
        role = sample["chosen"][1]["role"]
        label = 0 if role == "assistant" else 1

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "label": label,
            "query": query,
            "golden_friction": response
        }

    # Tokenize all samples
    processed_samples = [
        tokenize_friction_samples(sample) for sample in tqdm(train_data, desc="Tokenizing Samples")
    ]

    # Filter queries that are within the max_query_length
    filtered_samples = [
        sample for sample in processed_samples if len(sample["input_ids"]) <= max_query_length
    ]

    print(f"Filtered {len(processed_samples) - len(filtered_samples)} samples exceeding {max_query_length} tokens.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(filtered_samples)

    # Set format for PyTorch compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "query", "golden_friction"])

    return dataset



def build_friction_dataset_llama_full_context(train_data, config, max_length=1024, max_query_length=800):
    """
    Build dataset for training a Llama3  model on friction classification using the full context up to "Answer: <friction>",
    and filter queries exceeding the max_query_length.

    Args:
        train_data (`list`): List of training samples containing context and friction statements.
        config (`object`): Configuration object containing model_name.
        max_length (`int`): Maximum token length for the input sequence.
        max_query_length (`int`): Maximum token length allowed for the query.

    Returns:
        dataset (`datasets.Dataset`): The filtered dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token # for GPT 2
    
    # below processing is for llama 3
    tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
 

    def tokenize_friction_samples(sample):
        """
        Tokenizes and processes a sample into input, query, and label for training.

        Args:
            sample (`dict`): Dictionary containing context and friction statements.

        Returns:
            `dict`: Processed sample with tokenized input, query, and label.
        """
        # Extract context and response
        context = sample["prompt"]  # Dialogue context
        response = sample["chosen"]  # Friction response
        dialogue_context = sample['dialogue_context']
   
        # friction_response = response.split("<rationale>:")[1].strip()

        friction_response = "<rationale>:" + response.rsplit("<rationale>:", 1)[-1].strip()
       

        # print("friction parsed response in llama tokenization", friction_response)
        # print()  # Adds a new line
        # print("dialogue history in llama tokenization", dialogue_context)

        # Use full context up to "Answer: <friction>"
        input_sequence = context + " " + "### Assistant:"
        # Tokenize the input sequence
        input_ids = tokenizer.encode(input_sequence, truncation=True, max_length=max_length)
        # friction_input_ids = tokenizer.encode(friction_response, truncation=True, max_length=max_length)
        # Decode the input back into a query for readability
        query = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Process label: "assistant" indicates relevant friction, "user" indicates less relevant
        label = 1

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "label": label,
            "query": query,
            "golden_friction": friction_response, 
            "dialogue_context":dialogue_context
        }

    # Tokenize all samples
    processed_samples = [
        tokenize_friction_samples(sample) for sample in tqdm(train_data, desc="Tokenizing Samples")
    ]

    # Filter queries that are within the max_query_length
    filtered_samples = [
        sample for sample in processed_samples if len(sample["input_ids"]) <= max_query_length
    ]

    print(f"Filtered {len(processed_samples) - len(filtered_samples)} samples exceeding {max_query_length} tokens.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(filtered_samples)

    # Set format for PyTorch compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "query", "golden_friction"])

    return dataset

class Config:
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
 
 
 
def compute_probability(rewards_predicted_friction, rewards_golden_friction):
    # Compute the difference between the predicted and golden rewards
    reward_difference = rewards_predicted_friction - rewards_golden_friction
    
    # Normalize the difference to a range between -1 and 1
    # Here, we assume the difference is between 0 and 6 for normalization
    # normalized_score = 2 * (reward_difference - 0) / (5 - 0) - 1
    
    # # Clip the value to ensure it stays between -1 and 1
    # normalized_score = torch.clamp(normalized_score, -1.0, 1.0)
    
    return reward_difference



def compute_rewards_from_classifier(model, tokenizer, queries, responses, max_length=512, response_length=100, device="cuda", golden_friction =None):
    """
    Computes rewards for responses based on a trained friction classifier.

    Args:
        model: The trained BERT classification model.
        tokenizer: The tokenizer corresponding to the model.
        queries: List of query strings (prompts).
        responses: List of response strings (generated outputs).
        max_length: Maximum length for tokenization.
        response_length: Max tokens allocated to the response.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        rewards: List of rewards for each query-response pair.
    """
    model = model.to(device)
    model.eval()
    rewards = []
    system_prompt_rm = (
        "Please rate the following friction intervention in light of the **dialogue history** of a *game* provided below. "
        "A friction intervention is a statement that acts as indirect persuasion and prompts participants to "
        "reevaluate their beliefs and assumptions about the task, primarily—but not exclusively—in response to new evidence "
        "that challenges their preconceived notions about the state of the task or the block weights."
    )

    def extract_text(input_string):
        # Define the regular expression pattern to match text between 'user' and '### Assistant:'
        pattern = r'user\n(.*?)### Assistant:'
        
        # Use re.search to find the matching part
        match = re.search(pattern, input_string, re.DOTALL)  # re.DOTALL allows dot (.) to match newlines
        
        # If a match is found, return the extracted text; otherwise, return None
        if match:
            return match.group(1).strip()
        else:
            return None


    with torch.no_grad():
        for query, response, gold_friction in zip(queries, responses, golden_friction):
            # print("query  in reward fetching", query)
            dialogue_context = extract_text(query)
            # print("dialogue context in reward fetching", dialogue_context)
            prompt_and_dialogue_context = system_prompt_rm + dialogue_context
            # print("full prompt_and_dialogue_context context in reward fetching", len(golden_friction))

            predicted_response = f"</s> {response} </s>"
            gold_friction = f"</s> {gold_friction} </s>"
            # Tokenize response first to ensure it is fully included
            encoded_response = tokenizer(
                predicted_response,
                truncation=True,
                max_length=response_length,  # Cap response length
                padding=False,
                return_tensors="pt"
            )

            encoded_golden_response = tokenizer(
                gold_friction,
                truncation=True,
                max_length=response_length,  # Cap response length
                padding=False,
                return_tensors="pt"
            )


            # Calculate the remaining space for the query
            remaining_length = max_length - encoded_response["input_ids"].size(1) - 1  # Space for [SEP]
            remaining_length_golden = max_length - encoded_golden_response["input_ids"].size(1) - 1  # Space for [SEP]
            # Tokenize query with the remaining length
            # encoded_query = tokenizer(
            #     query,
            #     truncation=True,
            #     max_length=remaining_length,  # Truncate query to remaining space
            #     padding=False,
            #     return_tensors="pt"
            # )

            encoded_query = tokenizer(
                prompt_and_dialogue_context,
                truncation=True,
                max_length=remaining_length,  # Truncate query to remaining space: this one is with the change of prompts of rthe trained rewards model OPT 1.3
                padding=False,
                return_tensors="pt"
            )


            # Move query and response tensors to the same device
            encoded_query = {key: value.to(device) for key, value in encoded_query.items()}
            encoded_response = {key: value.to(device) for key, value in encoded_response.items()}
            encoded_golden_response = {key: value.to(device) for key, value in encoded_golden_response.items()}
            # Combine query and response with [SEP] token
            # sep_token = torch.tensor([[tokenizer.sep_token_id]], device=device)
            # input_ids = torch.cat([
            #     encoded_query["input_ids"],
            #     sep_token,  # [SEP] token
            #     encoded_response["input_ids"]
            # ], dim=1)

            # attention_mask = torch.cat([
            #     encoded_query["attention_mask"],
            #     torch.tensor([[1]], device=device),  # Attention for [SEP]
            #     encoded_response["attention_mask"]
            # ], dim=1)

            #Avoid the SEP token for the OPT RM (probably does not make much difference)
            # sep_token = torch.tensor([[tokenizer.sep_token_id]], device=device)
            # for predicted friction encoding
            input_ids = torch.cat([
                encoded_query["input_ids"],
   
                encoded_response["input_ids"]
            ], dim=1)

            attention_mask = torch.cat([
                encoded_query["attention_mask"],
   
                encoded_response["attention_mask"]
            ], dim=1)
            

            # for golden friction encoding
            input_ids_golden = torch.cat([
                encoded_query["input_ids"],
       
                encoded_golden_response["input_ids"]
            ], dim=1)

            attention_mask_golden = torch.cat([
                encoded_query["attention_mask"],
 
                encoded_golden_response["attention_mask"]
            ], dim=1)



            # Ensure the final length does not exceed max_length
            if input_ids.size(1) > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            if input_ids_golden.size(1) > max_length:
                input_ids_golden = input_ids_golden[:, :max_length]
                attention_mask_golden = attention_mask_golden[:, :max_length]   

            # Get model outputs
            # rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
            rewards_predicted_friction = model(input_ids=input_ids, attention_mask = attention_mask)[0]
            rewards_golden_friction = model(input_ids=input_ids_golden, attention_mask = attention_mask_golden)[0]
            # Compute the probability
            probability = compute_probability(rewards_predicted_friction, rewards_golden_friction)
 

            print("Probability based on reward difference:", probability)
            rewards.append(probability)
            # #previous code below for BERT RM 
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits

            # # Compute reward as scaled "GOOD" score
            # good_score = torch.softmax(logits, dim=1)[0][0].item()  # "GOOD" class probability
            # reward = 2 * good_score - 1  # Scale to [-1, 1]
            # rewards.append(reward)

    return rewards, rewards_predicted_friction, rewards_golden_friction

def process_wtd_simulated_dataset(dataset, split, 
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24):

    if sanity_check:
        dataset = dataset.shuffle().select(range(1000))
    print(dataset)
 
    original_columns = dataset.column_names
    selected_columns = ['prompt', 'chosen', 'rejected', 'dialogue_context']
    removable_columns = [ x for x in original_columns if x not in selected_columns]
 
    processed_dataset = dataset.map(
        # return_prompt_and_responses,
  
        num_proc= num_proc,
        # remove_columns=removable_columns,
    )
    # Filter the dataset based on prompt and response length constraints
    print(f"Filtered dataset before length constraint (train): {len(processed_dataset)}")
    print(f"Getting prompt-length stats BEFORE filtering:")
    # analyze_lengths(processed_dataset, split = split)

    filtered_dataset = processed_dataset.filter(
        lambda x: (len(x["chosen"]) <= script_args.max_length) and 
                (len(x["rejected"]) <= script_args.max_length)
    )
    print(f"Script args max length: {script_args.max_length}")
    print(f"Filtered dataset after length constraint (train): {len(filtered_dataset)}")
    print(f"Getting prompt-length stats AFTER filtering:")
    # analyze_lengths(filtered_dataset,  split = split)
 
    print("Sample of filtered dataset (train):")
    print(filtered_dataset[:2])  # Show sample from filtered dataset
    return filtered_dataset


def sample_df_for_logging(df, n=5):
    if len(df) > n:
        return df.sample(n)
    return df    

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
 
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)
 
    set_seed(script_args.seed)
 
    friction_data_test = load_from_disk("##")
    friction_data = load_from_disk("##")  
    train_data = friction_data['train'].map(transform_and_assign_preference_deli)
 
    print(f"Size of the train set: {len(train_data)}")

 
    # FOR DELI
    train_dataset = process_wtd_simulated_dataset(train_data, split ='train',  sanity_check=False)
    train_dataset = train_dataset.select(range(20000))
    train_dataset = train_dataset.shuffle(seed=42) 
    print("size of DELI after processing and filtering: train_dataset", train_dataset)
 
    config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=3e-6,
    log_with="wandb",
    batch_size=16,  # Smaller batch size
    mini_batch_size=2,
    gradient_accumulation_steps=8,  # Ensure batch_size = mini_batch_size * gradient_accumulation_steps
)

    friction_dataset = build_friction_dataset_llama_full_context(train_dataset, config, max_length=900, max_query_length=512) # this keep queries (or friction dialogue context < 800 so that ~64 tokens for friction statement (q + f <= 1024); else, you get tensor error

    # Check an example from the dataset
    example = friction_dataset[0]
    print("Input IDs:", example["input_ids"])
    print("Query:", example["query"])
    print("Label:", example["label"])

    #load the lora and peft configs 


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    ref_model = None
    device_map = {"": Accelerator().local_process_index}
    #load the policy and reference model with value heads
 
    config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=8e-7,
    log_with="wandb",
    batch_size=8,  # Smaller batch size
    mini_batch_size=2,
    gradient_accumulation_steps=4,  # Ensure batch_size = mini_batch_size * gradient_accumulation_steps
 

)

    # Load the policy and reference model

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name,
        device_map={"": Accelerator().local_process_index},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=script_args.load_in_4bit,  # Enable 4-bit quantization
     
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = "<|reserved_special_token_0|>" # don't use the eos token for padding during finetuning since it can stop the model from learning when to stop
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    print("Loading LoRA adapter...")

    lora_merged_model = PeftModel.from_pretrained(
        model,
        script_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("script args model name", lora_merged_model)
 


    print("script args model name", lora_merged_model)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
    lora_merged_model,
    device_map=device_map,
    peft_config=lora_config,
    quantization_config=nf4_config,
    # reward_adapter=script_args.rm_adapter,
    use_safetensors=script_args.use_safetensors,
)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
 
    # Initialize ppo trainer class
    friction_ppo_trainer = FrictionPPO_trainer(config, model, ref_model, tokenizer, dataset=friction_dataset, data_collator=collator)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8}

    #now load the bert friction_classifier trained to assign rewards to good and rogue friction samples
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    CHECKPOINT_DIR = "friction_rm_DELI_output_dir_all_samples/checkpoint-3000" # DELI RM checkpoints
 

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load the trained model
    friction_classifier = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR,
    
    num_labels=1,  # Set to 1 for reward model classification
            trust_remote_code=True,
            torch_dtype=torch.float16,
    )
    friction_classifier = friction_classifier.to(DEVICE)
    print("loaded trained reward model", friction_classifier)
    # Load the tokenizer
    friction_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    # ppo needs response to be generated from policy, so configure generation below
    NUM_EPOCHS = 3  # Define the total number of epochs
 

    output_min_length = 160
    output_max_length = 256

    output_length_sampler = LengthSampler(output_min_length, output_max_length)
 
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,           # Add top_k filtering
        "top_p": 0.85,        # Slightly more conservative
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.8,    # Reduce for more stable outputs
    }

#run the ppo training loop finally, get policy generation, compute rewards and take the ppo step 
# Initialize variables for tracking rewards

all_rewards = []
average_rewards = []
steps = []

# Directory to save the final aggregated plot
PLOT_DIR = "./ppo_reward_trajectory_deli"
CHKPT_DIR = "./ppo_friction_checkpoints_deli"
os.makedirs(PLOT_DIR, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(CHKPT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

for epoch in range(NUM_EPOCHS):
    print(f"\nStarting Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch_idx, batch in tqdm(enumerate(friction_ppo_trainer.dataloader), desc=f"Training Epoch {epoch + 1}"):
        query_tensors = batch["input_ids"]
        golden_friction = batch["golden_friction"]
        attention_masks = batch["attention_mask"]  # Ensure attention_mask is available
        print("keys in batch of dataloader", batch.keys())
        print("Dialogue Context in batch:", batch.get('dialogue_context'))

        #### Generate Responses from GPT-2
        response_tensors = []
        print(f"Batch {batch_idx + 1}: Generating responses for queries...")

        for query_idx, query in enumerate(query_tensors):
            query_tensors = [q.to("cuda") for q in query_tensors]
            gen_len = output_length_sampler()
            remaining_space = 1024 - query.size(0)  # Calculate remaining space
            generation_kwargs["max_new_tokens"] = max(1, min(gen_len, remaining_space)) #to ensure max new tokens do not become zero as that will fail model.generate

            print("max_new_tokens", min(gen_len, remaining_space))
            response = friction_ppo_trainer.generate(
                query,
 
                **generation_kwargs,
            )
 
            total_length = query.size(0) + response.size(1)
            print("query and response size", query.size(0), response.size(1))
            if total_length > 1024:
                # Truncate the query instead of the response
                truncate_length = total_length - 1024
                query = query[-(query.size(0) - truncate_length):]  # Keep only the last tokens
                print(f"Truncated query for Query {query_idx} to fit within max_length.")

            response_tensors.append(response.squeeze()[-generation_kwargs["max_new_tokens"]:])
            if query_idx % 5 == 0:  # Log every 5 queries
                print(f"Query {query_idx}: Response generated.")

        # Decode responses into text for reward computation
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        #### Compute Rewards Using Friction Classifier
        queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        responses = batch["response"]
 
        rewards, rewards_predicted_friction, rewards_golden_friction = compute_rewards_from_classifier(
            friction_classifier, friction_tokenizer, queries, responses, max_length=1024, device="cuda", golden_friction = golden_friction
        )
 
        rewards = [torch.tensor(reward, device="cuda").squeeze() for reward in rewards]  # Convert rewards to 1D tensors
 
        data = {
                "predicted_friction": [],
                "Reward Probability": [],
                "Predicted Reward": [],
                "Golden Friction Sample": [],
                "Golden Reward": []
            }
        # Define conditions for logging responses
        reward_lower_threshold = 0.25
        reward_upper_threshold = 4
        # Display each response with its reward
        print("Responses and Rewards in this batch:")
        for idx, (response, reward, gold_friction, pred_reward, gold_reward) in enumerate(zip(responses, rewards, golden_friction, rewards_predicted_friction, rewards_golden_friction)):
            # print(f"Response {idx + 1}:\n{response}\nReward: {reward.item():.4f}\n")
            if reward.item() < reward_lower_threshold or reward.item() > reward_upper_threshold:
                data["predicted_friction"].append(response)
                data["Reward Probability"].append(reward.item())  # Assuming reward is a tensor
                data["Predicted Reward"].append(pred_reward.item())
                data["Golden Friction Sample"].append(gold_friction)
                data["Golden Reward"].append(gold_reward.item())

            print(f"Response {idx + 1}:")
            print(f"Predicted friction sample: {response}")
            print(f"Reward Probability: {reward}")
            print(f"Predicted Reward: {pred_reward.item():.4f}")
            print(f"Golden Friction Sample: {gold_friction}")
            print(f"Golden Reward: {gold_reward.item():.4f}")
            print("-" * 50)  # Divider between responses

            
        # Append rewards for tracking
        all_rewards.extend([r.item() for r in rewards])

        #log a df for fields outputted for logging 
      
        df_rewards = pd.DataFrame(data)
        # Log the DataFrame into WandB as a rich table
        wandb.log({"rewards_df": wandb.Table(dataframe=df_rewards)})
        # Optionally, display the DataFrame to the user
  
        #### Track Rewards
        avg_reward = sum([r.item() for r in rewards]) / len(rewards)
        steps.append(len(steps) + 1)  # Incremental step count
        average_rewards.append(avg_reward)
        
        
        #### Run PPO Step
        print(f"Number of query tensors before PPO step: {len(query_tensors)}")
        print(f"Number of response tensors before PPO step: {len(response_tensors)}")
        print(f"Number of rewards before PPO step: {len(rewards)}")

        # If you want to print the shape of individual tensors within these lists:
        # for idx, query in enumerate(query_tensors):  # Limit to first 5 for readability
        #     print(f"Query Tensor {idx + 1} Shape: {query.size()}")
        # for idx, response in enumerate(response_tensors):  # Limit to first 5 for readability
        #     print(f"Response Tensor {idx + 1} Shape: {response.size()}")

        print(len(rewards))
        stats = friction_ppo_trainer.step(
            queries=query_tensors,  # Pass query tensors directly
            responses=response_tensors,
            scores=rewards,
        )

        friction_ppo_trainer.log_stats(stats, batch, rewards)

        #### Intermediate Logging
        if batch_idx % 10 == 0:  # Log every 10 batches
            avg_reward = sum([r.item() for r in rewards]) / len(rewards)
            print(f"  [Epoch {epoch + 1}, Batch {batch_idx + 1}] Avg Reward: {avg_reward:.4f}")
#             print(f"  PPO Stats: {stats}")



    # After each epoch, save an aggregated plot
        if batch_idx % 50 == 0: 
            plt.figure(figsize=(10, 6))
            plt.plot(steps, average_rewards, label="Average Reward", marker="o", color="b")
            plt.xlabel("Steps")
            plt.ylabel("Average Reward")
            plt.title("Trajectory of Rewards During Training")
            plt.legend()
            plt.grid()

            # Save the plot to file
            plot_path = os.path.join(PLOT_DIR, "aggregated_reward_plot.png")
            plt.savefig(plot_path)
            print(f"Aggregated plot saved to {plot_path}")
            plt.close()

 
    # Save intermediate checkpoint every 50 batches
        if batch_idx % 200 == 0 and batch_idx > 0:
            save_path = f"{CHKPT_DIR}/ppo_checkpoint_epoch_{epoch + 1}_batch_{batch_idx}"
            os.makedirs(save_path, exist_ok=True)
            friction_ppo_trainer.model.module.save_pretrained(save_path)  # Use .module to access the underlying model
            friction_ppo_trainer.tokenizer.save_pretrained(save_path)
            print(f"Checkpoint saved for Epoch {epoch + 1}, Batch {batch_idx}")

 

    # Save epoch checkpoint
    save_path = f"{CHKPT_DIR}/ppo_checkpoint_epoch_{epoch + 1}"
    os.makedirs(save_path, exist_ok=True)
    friction_ppo_trainer.model.module.save_pretrained(save_path)  # Use .module to access the underlying model
    friction_ppo_trainer.tokenizer.save_pretrained(save_path)
    print(f"Checkpoint saved for Epoch {epoch + 1}")

    print(f"Epoch {epoch + 1} completed.")
