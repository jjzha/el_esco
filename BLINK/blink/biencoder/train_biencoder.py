# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import WEIGHTS_NAME, get_linear_schedule_with_warmup
from datasets import load_dataset

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
import numpy as np
import torch
from tqdm import tqdm

def evaluate(reranker, eval_dataloader, params, device, logger):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    r_precisions = []  # Store individual R-Precision values
    reciprocal_ranks = []

    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, _ = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
            torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        # Calculate R-Precision for the batch
        batch_r_precision = calculate_r_precision(logits, label_ids)
        r_precisions.append(batch_r_precision)  # Append the single R-Precision value

        # Calculate Reciprocal Rank for each example in the batch
        batch_reciprocal_ranks = calculate_reciprocal_ranks(logits, label_ids)
        reciprocal_ranks.extend(batch_reciprocal_ranks)

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    normalized_r_precision = np.mean(r_precisions)  # Calculate mean R-Precision
    mrr = calculate_mean_reciprocal_rank(reciprocal_ranks)

    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    logger.info("Eval R-Precision: %.5f" % normalized_r_precision)
    logger.info("Eval MRR: %.5f" % mrr)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["normalized_r_precision"] = normalized_r_precision
    results["mrr"] = mrr

    return results


def calculate_r_precision(logits, label_ids, position=1):
    # Calculate R-Precision at a specific position (default is @1)
    r_precisions = []
    for i in range(len(label_ids)-1):
        sorted_indices = np.argsort(logits[i])[::-1][:position]
        correct_predictions = np.isin(sorted_indices, label_ids[i])
        r_precision = np.sum(correct_predictions) / position
        r_precisions.append(r_precision)

    return np.mean(r_precisions)


def calculate_reciprocal_ranks(logits, label_ids):
    # Calculate Reciprocal Rank for each example in the batch
    reciprocal_ranks = []
    for i in range(len(label_ids)-1):
        sorted_indices = np.argsort(logits[i])[::-1]
        rank = np.where(sorted_indices == label_ids[i])[0]
        if rank.size > 0:
            reciprocal_rank = 1.0 / (rank[0] + 1)
        else:
            # Handle the case where the label is not in the sorted indices
            reciprocal_rank = 0.0
        reciprocal_ranks.append(reciprocal_rank)
    return reciprocal_ranks


def calculate_mean_reciprocal_rank(reciprocal_ranks):
    # Calculate Mean Reciprocal Rank (MRR) for the batch
    if len(reciprocal_ranks) == 0:
        return 0.0
    mrr = np.mean(reciprocal_ranks)
    return mrr


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_samples = load_dataset(
        "json",
        data_files={"train": os.path.join(params["data_path"], "train.jsonl")},
        streaming=False,
        cache_dir=params["cache_dir"],
    )["train"]
    logger.info("Read %d train samples." % len(train_samples))

    train_tensor_data = data.process_mention_data_for_hf(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        logger=logger,
        debug=params["debug"],
    )
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size, collate_fn=data.data_collator_for_hf
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid", params["data_path"])
    logger.info("Read %d valid samples." % len(valid_samples))

    valid_data, valid_tensor_data = data.process_mention_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )
    # evaluate before training
    results = evaluate(
        reranker, 
        eval_dataloader=valid_dataloader, 
        params=params, 
        device=device, 
        logger=logger,
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.json"), json.dumps(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, _ = batch
            loss, _ = reranker(context_input, candidate_input)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        results = evaluate(
            reranker, 
            eval_dataloader=valid_dataloader, 
            params=params, 
            device=device, 
            logger=logger,
        )


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
