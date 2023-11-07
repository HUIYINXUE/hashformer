from typing import List

from datasets import concatenate_datasets, load_dataset
import datasets
from transformers import RobertaTokenizerFast

from free_tokenizer import LSHTokenizer, Md5Tokenizer, LSHValueTokenizer, Md5ValueTokenizer

from tqdm.contrib import tenumerate
from tqdm import tqdm

from pathlib import Path

from sentencizer import Sentencizer

import numpy as np
import torch

from argparse import ArgumentParser

parser = ArgumentParser(description="Preprocess glue datasets for all tokenizers")
parser.add_argument("-p", "--path", help="(str) Where to save?",
                    default=None)
parser.add_argument("--disable_tqdm", help="(bool) If `True`, a progress bar will be disabled.",
                    default=False)
parser.add_argument("--mode", help="(str) Which tokenizer to use?",
                    default=None)
parser.add_argument("--task", help="(str) Which task in GLUE to preprocess?",
                    default=None)
parser.add_argument("--n_worker", help="(int) How many processor for preprocessing?",
                    default=1)
parser.add_argument("--max_length", help="(int) Max_length of the tokenizer.",
                    default=128)
args = parser.parse_args()


def tokenize_example(task_name, tokenizer, example):
    if task_name in ['cola', 'sst2']:
        tokenized_example = tokenizer(example['sentence'], max_length=args.max_length,
                                      padding="max_length", truncation="longest_first")
    elif task_name=='mnli':
        batched = list(map(lambda x, y: (x, y), example['premise'], example['hypothesis']))
        tokenized_example = tokenizer(batched, max_length=args.max_length,
                                      padding="max_length", truncation="longest_first")
    elif task_name=='qnli':
        batched = list(map(lambda x, y: (x, y), example['question'], example['sentence']))
        tokenized_example = tokenizer(batched, max_length=args.max_length,
                                      padding="max_length", truncation="longest_first")
    elif task_name=='qqp':
        batched = list(map(lambda x, y: (x, y), example['question1'], example['question2']))
        tokenized_example = tokenizer(batched, max_length=args.max_length,
                                      padding="max_length", truncation="longest_first")
    else:
        batched = list(map(lambda x, y: (x, y), example['sentence1'], example['sentence2']))
        tokenized_example = tokenizer(batched, max_length=args.max_length,
                                      padding="max_length", truncation="longest_first")
    return tokenized_example


def preproc_glue(tokenizer, save_on_disk=True):
    if args.path is None:
        raise ValueError("Please give an appropriate path!")
    token_save_path = Path(args.path) / str(args.mode)
    if token_save_path.exists() is False:
        token_save_path.mkdir()

    token_save_path = token_save_path / str(args.task)
    if token_save_path.exists() is False:
        token_save_path.mkdir()

    if args.task == 'mnli':
        print("Preproc train set...")
        dataset = load_dataset("glue", "mnli")['train']
        dataset = dataset.map(lambda example: tokenize_example(args.task, tokenizer, example),
                              batched=True, batch_size=100, num_proc=int(args.n_worker),
                              load_from_cache_file=False)
        if save_on_disk:
            print("Saving the processed train set to disk...")
            task_save_path = token_save_path / str('train')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            dataset.save_to_disk(str(task_save_path))
            print("Done!")
        print("Preproc test set...")
        test_dataset = load_dataset("glue", "mnli")['validation_matched']
        test_dataset = test_dataset.map(lambda example: tokenize_example(args.task, tokenizer, example),
                                        batched=True, batch_size=100, num_proc=int(args.n_worker),
                                        load_from_cache_file=False)
        if save_on_disk:
            print("Saving the processed test set to disk...")
            task_save_path = token_save_path / str('test')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            test_dataset.save_to_disk(str(task_save_path))
            print("Done!")
        print("Preproc dev set...")
        dev_dataset = load_dataset("glue", "mnli")['validation_mismatched']
        dev_dataset = dev_dataset.map(lambda example: tokenize_example(args.task, tokenizer, example),
                                      batched=True, batch_size=100, num_proc=int(args.n_worker),
                                      load_from_cache_file = False)
        dataset = concatenate_datasets([dev_dataset, test_dataset])
        if save_on_disk:
            print("Saving the processed dev set to disk...")
            task_save_path = token_save_path / str('dev')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            dataset.save_to_disk(str(task_save_path))
            print("Done!")

    else:
        print("Preproc train set...")
        dataset = load_dataset("glue", args.task)['train']
        dataset = dataset.map(lambda example: tokenize_example(args.task, tokenizer, example),
                              batched=True, batch_size=100, num_proc=int(args.n_worker),
                              load_from_cache_file=False)
        if save_on_disk:
            print("Saving the processed train set to disk...")
            task_save_path = token_save_path / str('train')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            dataset.save_to_disk(str(task_save_path))
            print("Done!")
        print("Preproc dev set...")
        dataset = load_dataset("glue", args.task)['validation']
        dataset = dataset.map(lambda example: tokenize_example(args.task, tokenizer, example),
                              batched=True, batch_size=100, num_proc=int(args.n_worker),
                              load_from_cache_file=False)
        if save_on_disk:
            print("Saving the processed dev set to disk...")
            task_save_path = token_save_path / str('dev')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            dataset.save_to_disk(str(task_save_path))
            print("Done!")
        if save_on_disk:
            print("Saving the processed test set to disk...")
            task_save_path = token_save_path / str('test')
            if task_save_path.exists() is False:
                task_save_path.mkdir()
            dataset.save_to_disk(str(task_save_path))
            print("Done!")


if __name__ == "__main__":
    if args.mode == 'vanilla':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.mode == 'md5val':
        tokenizer = Md5ValueTokenizer()
    elif args.mode == 'lshval':
        tokenizer = LSHValueTokenizer()
    elif args.mode == 'md5':
        tokenizer = Md5Tokenizer()
    elif args.mode == 'lsh':
        tokenizer = LSHTokenizer()
    else:
        raise ValueError("`mode` must in `vanilla`, `md5val`, `lshval`, `md5`, `lsh`")
    preproc_glue(tokenizer, save_on_disk=True)









