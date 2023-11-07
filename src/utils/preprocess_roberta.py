"""Preprocessing code: DOC-SENTENCES & FULL-SENTENCES"""
from typing import List

from datasets import concatenate_datasets, load_dataset
import datasets
from transformers import RobertaTokenizerFast

from tqdm.contrib import tenumerate
from tqdm import tqdm

from pathlib import Path

from sentencizer import Sentencizer

import numpy as np
import torch

from argparse import ArgumentParser

parser = ArgumentParser(description="Preprocess datasets for RoBERTa")
parser.add_argument("-p", "--path", help="(str) Where to save or where to load from?",
                    default=None)
parser.add_argument("--disable_tqdm", help="(bool) If `True`, a progress bar will be disabled.",
                    default=False)
parser.add_argument("--mask_prob", help="(float) Ratio of masked tokens",
                    default=0.15)
args = parser.parse_args()



def concatenate_sentences(sentences: list, max_length=512):
    """Utility function to concatenate sentences into str, while ensuring
    the number of tokens should be less than 512."""
    chunk = ""
    total_num_tokens = 0

    for sentence in sentences:
        senlen = len(sentence.split(' '))
        if (total_num_tokens + senlen + 1) < max_length:
            if chunk != "":
                chunk += " " + sentence
                total_num_tokens += senlen + 1  # blank
            else:
                chunk = sentence
                total_num_tokens += senlen
        else:
            break

    return chunk, total_num_tokens


def preprocess_book(lines: List[str]):
    """Preprocess book corpus by each line.
    Args:
        articles (List[str]): str list of lines.

    Returns:
        dict.

    References:
        https://github.com/huggingface/transformers/blob/9bdce3a4f91c6d53873582b0210e61c92bba8fd3/src/transformers/data/datasets/language_modeling.py#L19
        https://github.com/huggingface/transformers/blob/9bdce3a4f91c6d53873582b0210e61c92bba8fd3/src/transformers/data/data_collator.py#L120
        https://github.com/dhlee347/pytorchic-bert/blob/master/tokenization.py
        https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
    """
    text_list = []
    overlen_lines = []
    num_stop_words = []
    ratio_stop_words = []
    seq_len_list = []

    chunk = ""
    total_num_tokens = 0
    for index, line in tenumerate(lines, disable=args.disable_tqdm):
        # We try to fill up the chunk such that the number of tokens in it is
        # close to 512. Then, we use the chunk as an input sequence.

        num_tokens = len(line.split(' '))

        if (total_num_tokens + num_tokens) < 512:
            # add to chunk
            if chunk != "":
                chunk += " " + line
                total_num_tokens += num_tokens + 1
            else:
                chunk = line
                total_num_tokens += num_tokens

        else:
            # remove blanks
            text = chunk.strip()
            text = text.replace("\n", "")

            # add to lists
            text_list.append(text)

            if num_tokens < 512:
                # initialise again
                total_num_tokens = num_tokens
                chunk = line
            else:
                # over-length sample
                # put lists -> sentencize & use as a sample later
                overlen_lines.append(line)
                chunk = ""
                total_num_tokens = 0

    if overlen_lines != []:
        print("Preprocessing over-length samples in BookCorpus...")
        # split each str line into sentences
        sentencizer = Sentencizer()
        sentencised_lines = sentencizer(overlen_lines)

        for index, line in tenumerate(sentencised_lines, disable=args.disable_tqdm):
            # concatenate sentences with their maximum length 512
            text, total_num_tokens = concatenate_sentences(line, max_length=512)

            # remove blanks
            text = text.strip()
            text = text.replace("\n", "")
            if text == "" or len(text.split(" ")) <= 1:
                continue

            # add to lists
            text_list.append(text)

    return {"text": text_list}


def remove_wiki_info(example):
    """Remove unnecessary texts in the wikipedia corpus."""
    keywords = ("See also", "References", "Category")
    for keyword in keywords:
        index = example["text"].find(keyword)
        if index != -1:
            example["text"] = example["text"][:index]
    return example


def preprocess_wiki(articles: List[str]):
    """Preprocess wikipedia corpus by each article.
    Args:
        articles (List[str]): str list of articles.

    Returns:
        dict.
    """
    text_list = []

    # split each str article into sentencess
    print("Split wiki articles into sentences...")
    sentencizer = Sentencizer()
    sentencised_articles = sentencizer(articles, n_jobs=40)

    print("Generate Roberta samples...")
    for index, article in tenumerate(sentencised_articles, disable=args.disable_tqdm):
        # concatenate sentences with their maximum length 512
        text, total_num_tokens = concatenate_sentences(article, max_length=512)

        # remove blanks
        text = text.strip()
        text = text.replace("\n", "")
        if text == "" or len(text.split(" ")) <= 1:
            continue

        # add to lists
        text_list.append(text)

    return {"text": text_list}


def tokenize_example(tokenizer, example):
    """Tokenise batched examples using a pre-trained RoBERTa tokeniser."""
    tokenized_example = tokenizer(example['text'], max_length=512, padding="max_length", truncation="longest_first")

    return tokenized_example


def preprocess_dataset_for_roberta_pretraining(seed: int = 1234,
                                               data_path=None,
                                               save_on_disk: bool = True,
                                               save_path=None):
    """Preprocess a dataset for pre-training
    Args:
        seed (int): random seed for shuffling
        data_path (str): the location of the processed dataset. This must be given
                         when `load_from_disk` is `True`.
        save_on_disk (bool): whether to save a processed dataset on the disk.
        save_path (str): where to save the processed dataset. This must be given
                         when `save_on_disk` is `True`.

    References:
        https://huggingface.co/docs/datasets/splits.html
    """

    # setting up the path to save datasets
    if Path(save_path).exists() is False:
        Path(save_path).mkdir()

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    for ratio in range(0, 100, 1):
        # To avoid memory error, we preprocess datasets 10% each, then save it.

        # preprocess bookcorpus data
        book_dataset = load_dataset("bookcorpus", split=f"train[{ratio}%:{ratio + 1}%]")
        print("Preprocessing BookCorpus...")
        book_dataset = preprocess_book(book_dataset["text"])

        # preprocess wiki data
        wiki_dataset = load_dataset("wikipedia", "20200501.en", split=f"train[{ratio}%:{ratio + 1}%]")
        wiki_dataset.remove_columns_("title")  # only keep the text based on the original BERT paper
        print("Removing unnecessary wiki data...")
        wiki_dataset = wiki_dataset.map(remove_wiki_info)  # remove references etc.
        print("Preprocessing Wikipedia dataset...")
        wiki_dataset = preprocess_wiki(wiki_dataset["text"])  # make a sentence pair & labels

        # tokenisation
        print("Tokenising datasets...")
        book_dataset = datasets.Dataset.from_dict(book_dataset)
        wiki_dataset = datasets.Dataset.from_dict(wiki_dataset)
        bert_dataset = concatenate_datasets([book_dataset, wiki_dataset])
        bert_dataset = bert_dataset.shuffle(seed=seed)
        dataset = bert_dataset.map(lambda example: tokenize_example(tokenizer, example),
                                   batched=True, batch_size=100, num_proc=40)

        """
        # this is needed to use `input_ids` in generating stop word masks
        # type should be ndarray: a `dict` of types like `(<class 'list'>, <class 'numpy.ndarray'>)`.
        dataset.set_format(type='np', columns=['input_ids', 'attention_mask', 
                                               'num_stop_word', 'ratio_stop_word'])

        # generate stop word masks
        print("Generating stop word masks...")
        dataset = dataset.map(lambda example: add_stop_word_masking_example(example),
                              batched=True, batch_size=64)

        # generate masked input sequences
        print("Generating masked input sequences...")
        dataset = dataset.map(lambda example: mask_example(example, tokenizer, args.mask_prob),
                              batched=True, batch_size=32)
        """
        # reset format
        dataset.set_format()

        if save_on_disk:
            print("Saving the processed data to disk...")
            if save_path is None:
                raise ValueError("Please give an appropriate path!")
            temp_save_path = Path(save_path) / str(ratio)
            if temp_save_path.exists() is False:
                temp_save_path.mkdir()
            dataset.save_to_disk(str(temp_save_path))
            print("Done!")


def main():
    # preprocess datasets
    preprocess_dataset_for_roberta_pretraining(save_path=args.path)


if __name__ == "__main__":
    main()