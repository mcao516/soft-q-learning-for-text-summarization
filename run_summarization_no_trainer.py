#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version
from rouge_score import rouge_scorer

from soft_q_loss import SoftQLearningCriterion
from data_utils import get_raw_dataset, process_raw_dataset, postprocess_text

_has_wandb = False
try:
    import wandb

    _has_wandb = True
except:
    logger.warning(
        "W&B logger is not installed, \
        for advanced logging please install using pip install wandb"
    )

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Cache directory for datasets."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="kd_experiment",
        help="W&B job name."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="summarization-kd",
        help="W&B project name."
    )

    # soft-Q loss arguments
    parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                        help='epsilon for label smoothing, 0 means no label smoothing')
    parser.add_argument('--reward-shaping', action='store_true',
                        help='Whether use reward shaping')
    parser.add_argument('--old-r-min', default=0., type=float,
                        help='Original minimum reward value')
    parser.add_argument('--old-r-max', default=1.0, type=float,
                        help='Original maximum reward value')
    parser.add_argument('--new-r-min', default=-0.5, type=float,
                        help='Minimum reward value after reshaping')
    parser.add_argument('--new-r-max', default=0.5, type=float,
                        help='Maximum reward value after reshaping')
    parser.add_argument('--gamma-pcl', default=1.0, type=float,
                        help='Reward discount factor')
    parser.add_argument('--tau-pcl', default=1.0, type=float,
                        help='Shannon entropy coefficient in PCL')

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def setup_wandb(args, model, resume_id=None):
    if _has_wandb:
        if resume_id is not None:
            wandb.init(
                project=args.project_name,
                group=args.job_name,
                dir="./",
                resume="allow",
                id=resume_id,
            )
        else:
            wandb.init(project=args.project_name, group=args.job_name, dir="./")
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
    else:
        logger.info("W&B library not installed. Using only CLI logging.")


def report_metrics(lr, loss, step):
    current_lr = lr[0] if type(lr) == list else lr

    if _has_wandb:
        log_info = {
            f"train/lr": current_lr,
            f"train/train_loss": loss,
        }
        wandb.log(log_info, step=step)


def load_pretrained_model_and_tokenizer(
    model_name_or_path,
    config_name,
    tokenizer_name,
    model_type=None,
    use_slow_tokenizer=False
):
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=not use_slow_tokenizer)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    return config, tokenizer, model


def setup_optimizer(args, model):
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def eval(args, accelerator, model, tokenizer, eval_dataloader, metric):
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)


def generate(
    args,
    accelerator,
    model,
    tokenizer,
    batch,
    do_sample=False
):
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": 1,
    }

    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
            do_sample=do_sample,
        )

    return generated_tokens


def decode_ids_to_strs(
    args,
    labels,
    accelerator,
    tokenizer
):
    if not args.pad_to_max_length:
        # If we did not pad to max length, we need to pad the labels too
        labels = accelerator.pad_across_processes(
            labels, dim=1, pad_index=tokenizer.pad_token_id
        )
    
    labels = labels.cpu().numpy()

    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_labels


def calculate_rouge(args, decoded_preds, decoded_labels):
    assert len(decoded_preds) == len(decoded_labels), \
        "predicts: {}; references: {}".format(len(decoded_preds), len(decoded_labels))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rouge_scores, rewards, length_rewards = [], [], []
    for r, p in zip(decoded_labels, decoded_preds):
        rouge = scorer.score(r, p)
        rouge_scores.append(rouge)

        reward = rouge['rouge1'].fmeasure
        # if args.length_reward:
        #     length_reward = 1.0 if len(p) <= len(r) else 0.0
        #     reward += length_reward
        #     length_rewards.append(length_reward)

        rewards.append(reward)

    return rewards


def main():
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.gradient_accumulation_steps)
    # accelerator = Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    config, tokenizer, model = load_pretrained_model_and_tokenizer(
        args.model_name_or_path,
        args.config_name,
        args.tokenizer_name,
        model_type=args.model_type,
        use_slow_tokenizer=args.use_slow_tokenizer
    )

    _, _, tgt_model = load_pretrained_model_and_tokenizer(
        args.model_name_or_path,
        args.config_name,
        args.tokenizer_name,
        model_type=args.model_type,
        use_slow_tokenizer=args.use_slow_tokenizer
    )

    model.resize_token_embeddings(len(tokenizer))
    tgt_model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # setup W&B logging
    if accelerator.is_main_process:
        setup_wandb(args, model, resume_id=None)

    # Get the raw dataset
    raw_datasets = get_raw_dataset(args)

    # Preprocessing the datasets.
    processed_datasets = process_raw_dataset(args, accelerator, raw_datasets, tokenizer)
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Prepare loss function
    criterion = SoftQLearningCriterion(1, args.label_smoothing)

    # Prepare optimizer
    optimizer = setup_optimizer(args, model)

    # Prepare everything with our `accelerator`.
    model, tgt_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, tgt_model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    def polyak_update(model, model_, target_lr=0.001):
        for param_, param in zip(model_.parameters(), model.parameters()):
            param_.data.copy_((1 - target_lr) * param_ + target_lr * param)

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            """
            batch: {
                attention_mask: [batch_size, src_length]
                input_ids: [batch_size, src_length]
                labels: [batch_size, tgt_length]
                decoder_input_ids: [batch_size, tgt_length]
            }
            """
            if step % 2 == 0: 
                generated_tokens = generate(
                    args,
                    accelerator,
                    model,
                    tokenizer,
                    batch,
                    do_sample=True
                )

                # [batch_size, tgt_length]
                labels = torch.zeros_like(generated_tokens)
                labels[:, 1:] = generated_tokens[:, 1:]

                decoder_input_ids = torch.zeros_like(generated_tokens)
                decoder_input_ids[:, 1:] = labels[:, :-1]
                decoder_input_ids[:, 0] = 2

            else: # learning from demonstration
                pad_mask = batch.labels.eq(-100) # replace -100 with 1
                labels = batch.labels.masked_fill(pad_mask, 1)
                decoder_input_ids = batch.decoder_input_ids

            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            with torch.no_grad():
                tgt_outputs = tgt_model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )

            # reward calculation
            decoded_labels = decode_ids_to_strs(args, batch.labels, accelerator, tokenizer)
            decoded_preds = decode_ids_to_strs(args, labels, accelerator, tokenizer)
            rewards = calculate_rouge(args, decoded_preds, decoded_labels)
            rewards = torch.tensor(rewards).to(outputs[0])

            sample = {'target': labels, 'rewards': rewards}
            loss = criterion(outputs[0], tgt_outputs[0], sample)[0]
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                # update target model
                polyak_update(model, tgt_model)

                # W&B logging
                if accelerator.is_main_process:
                    report_metrics(
                        lr_scheduler.get_last_lr(),
                        loss.item(),
                        completed_steps
                    )

            if completed_steps >= args.max_train_steps:
                break

        # Run evaluation
        logger.info("***** Running evaluation *****")
        eval(args, accelerator, model, tokenizer, eval_dataloader, metric)

        # Extract a few results from ROUGE
        result = metric.compute(use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)

        if accelerator.is_main_process and _has_wandb:
            log_info = {"Validation/" + k: v for k, v in result.items()}
            wandb.log(log_info, completed_steps)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()