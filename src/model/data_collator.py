from transformers import BatchEncoding, PreTrainedTokenizerBase
import torch
import re
import pickle
import numpy as np

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


########################################################################################################################

class DataCollatorForShuffleRandomThreeWayClassification:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, manipulate_prob: float):
        self.tokenizer = tokenizer
        self.manipulate_prob = manipulate_prob

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # Shuffle words and create word masks
        manipulated_input_ids, shuffle_random_mask = self.manipulate_tokens(batch["input_ids"])

        return {"input_ids": manipulated_input_ids, "attention_mask": batch["attention_mask"],
                "shuffle_random_mask": shuffle_random_mask}

    def manipulate_tokens(self, input_ids):
        """Prepare shuffled tokens inputs/masks."""
        # init
        manipulated_input_ids = input_ids.clone()
        shuffle_random_mask = torch.zeros_like(manipulated_input_ids)

        # create shuffled input_ids matrices
        shuffled_words = manipulated_input_ids[:, torch.randperm(manipulated_input_ids.size()[1])]  # row-wise shuffle
        # We need to care about special tokens: start, end, pad, mask.
        # If shuffled words fall in these, they must be put back to their original tokens.
        # This might cause the case where a shuffled token is not actually a shuffled one,
        # but because the number of special tokens is small, it does not matter and might contribute to robustness.
        special_tokens_indices = [
            self.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            shuffled_words.tolist()
        ]
        special_tokens_indices = torch.tensor(special_tokens_indices, dtype=torch.bool)  # -> boolean indices
        shuffled_words[special_tokens_indices] = manipulated_input_ids[special_tokens_indices]

        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full(manipulated_input_ids.shape, self.manipulate_prob)
        special_tokens_mask = [
            self.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            manipulated_input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool()  # -> boolean indices
        manipulated_input_ids[shuffled_indices] = shuffled_words[shuffled_indices]
        shuffle_random_mask[shuffled_indices] = 1

        # replace some tokens with random ones
        # this should not override shuffled tokens.
        random_indices = torch.bernoulli(torch.full(manipulated_input_ids.shape,
                                                    self.manipulate_prob)).bool() & ~shuffled_indices & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), manipulated_input_ids.shape, dtype=torch.long)
        manipulated_input_ids[random_indices] = random_words[random_indices]
        shuffle_random_mask[random_indices] = 2

        # We only compute loss on active tokens
        shuffle_random_mask[special_tokens_mask] = -100

        return manipulated_input_ids, shuffle_random_mask

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 4 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

########################################################################################################################


class DataCollatorForShuffleRandomThreeWayClassification_Proj:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self, manipulate_prob: float):
        #self.tokenizer = tokenizer
        self.manipulate_prob = manipulate_prob

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        tmp_tuple = list(map(lambda x: self.deep_proc(x), examples))
        inputs_embeds, input_ids = list(list(zip(*tmp_tuple)))
        # Handling of all keys.
        batch = {}
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in examples])
        batch['inputs_embeds'] = torch.tensor(list(inputs_embeds))
        batch['input_ids'] = torch.tensor(list(input_ids))

        # Shuffle words and create word masks
        manipulated_input_ids, manipulated_input_embeds, shuffle_random_mask = self.manipulate_tokens(batch["input_ids"],
                                                                                                      batch["inputs_embeds"])

        return {"input_ids": manipulated_input_ids, "inputs_embeds": manipulated_input_embeds,
                "attention_mask": batch["attention_mask"], "shuffle_random_mask": shuffle_random_mask}

    def manipulate_tokens(self, input_ids, input_embeds):
        """Prepare shuffled tokens inputs/masks."""
        # init
        manipulated_input_embeds = input_embeds.clone()
        ini_size = input_ids.size()
        shuffle_random_mask = torch.zeros((ini_size[0], ini_size[2]), dtype=torch.long)
        special_tokens_mask = self.get_special_tokens_mask(input_ids)
        # create shuffled input_ids and input_embeds matrices
        ## get shuffled index sequences
        shuffled_index_seq = torch.randperm(ini_size[2])
        ## row-wise shuffle tokens
        shuffled_input_embeds = manipulated_input_embeds[:, shuffled_index_seq, :]
        ## manipulate specical_tokens_mask
        shuffled_special_tokens_mask = special_tokens_mask[:, shuffled_index_seq]
        ## If shuffled words fall in special tokens, they must be put back to their original tokens.
        shuffled_input_embeds[shuffled_special_tokens_mask] = manipulated_input_embeds[shuffled_special_tokens_mask]
        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full((ini_size[0], ini_size[2]), self.manipulate_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool()  # -> boolean indices
        ### no need to shuffle input_ids as it only indicates special tokens
        manipulated_input_embeds[shuffled_indices] = shuffled_input_embeds[shuffled_indices]
        shuffle_random_mask[shuffled_indices] = 1
        # replace some tokens with random ones
        # this should not override shuffled tokens.
        random_indices = torch.bernoulli(torch.full((ini_size[0], ini_size[2]), self.manipulate_prob)).bool() & \
                         ~shuffled_indices & ~special_tokens_mask
        manipulated_input_embeds[random_indices] = self.random_bitarrays(manipulated_input_embeds[random_indices])
        shuffle_random_mask[random_indices] = 2
        # We only compute loss on active tokens
        shuffle_random_mask[special_tokens_mask] = -100
        return input_ids, manipulated_input_embeds, shuffle_random_mask

    def get_special_tokens_mask(self, input_ids):
        ind = (1 - input_ids[:, 1, :]).sum(dim=1) - 2
        max_len = input_ids[:, 1, :].size(1)
        new_mat = torch.tensor([[1] + [0] * i + [1] * (max_len - 1 - i) for i in ind], dtype=torch.bool)
        return new_mat

    def random_bitarrays(self, x):
        input_size = x.size()
        prob = torch.rand(input_size[0], 1).expand(input_size)
        return torch.bernoulli(prob)

    def proc_pos(self, x, y):
        base10 = int(x, base=36)
        if base10 >= 5:
            spec_ind = 5
            base2 = format(base10 - 5, '0>128b')
            bitarray = np.array(list(base2), dtype=np.float32).tolist()
        else:
            spec_ind = base10
            bitarray = [0.0] * 128
        return bitarray, spec_ind, 1-y

    def deep_proc(self, example):
        # base36 to base10
        temp_list = list(map(lambda x, y: self.proc_pos(x, y),
                             example['input_ids'].split(' '), example['attention_mask']))
        bitarray, spec_ind, padding_mask = list(list(zip(*temp_list)))
        return list(bitarray), [list(spec_ind), list(padding_mask)]

########################################################################################################################

class DataCollatorForShuffleRandomThreeWayClassification_Conj:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self, manipulate_prob: float):
        #self.tokenizer = tokenizer
        self.manipulate_prob = manipulate_prob

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        tmp_tuple = list(map(lambda x: self.deep_proc(x), examples))
        inputs_embeds, input_ids = list(list(zip(*tmp_tuple)))
        # Handling of all keys.
        batch = {}
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in examples])
        batch['inputs_embeds'] = torch.tensor(list(inputs_embeds))
        batch['input_ids'] = torch.tensor(list(input_ids))

        # Shuffle words and create word masks
        manipulated_input_ids, manipulated_input_embeds, shuffle_random_mask = self.manipulate_tokens(batch["input_ids"],
                                                                                                      batch["inputs_embeds"])

        return {"input_ids": manipulated_input_ids, "inputs_embeds": manipulated_input_embeds,
                "attention_mask": batch["attention_mask"], "shuffle_random_mask": shuffle_random_mask}

    def manipulate_tokens(self, input_ids, input_embeds):
        """Prepare shuffled tokens inputs/masks."""
        # init
        manipulated_input_embeds = input_embeds.clone()
        ini_size = input_ids.size()
        shuffle_random_mask = torch.zeros((ini_size[0], ini_size[2]), dtype=torch.long)
        special_tokens_mask = self.get_special_tokens_mask(input_ids)
        # create shuffled input_ids and input_embeds matrices
        ## get shuffled index sequences
        shuffled_index_seq = torch.randperm(ini_size[2])
        ## row-wise shuffle tokens
        shuffled_input_embeds = manipulated_input_embeds[:, shuffled_index_seq, :]
        ## manipulate specical_tokens_mask
        shuffled_special_tokens_mask = special_tokens_mask[:, shuffled_index_seq]
        ## If shuffled words fall in special tokens, they must be put back to their original tokens.
        shuffled_input_embeds[shuffled_special_tokens_mask] = manipulated_input_embeds[shuffled_special_tokens_mask]
        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full((ini_size[0], ini_size[2]), self.manipulate_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool()  # -> boolean indices
        ### no need to shuffle input_ids as it only indicates special tokens
        manipulated_input_embeds[shuffled_indices] = shuffled_input_embeds[shuffled_indices]
        shuffle_random_mask[shuffled_indices] = 1
        # replace some tokens with random ones
        # this should not override shuffled tokens.
        random_indices = torch.bernoulli(torch.full((ini_size[0], ini_size[2]), self.manipulate_prob)).bool() & \
                         ~shuffled_indices & ~special_tokens_mask
        manipulated_input_embeds[random_indices] = self.random_indarrays(manipulated_input_embeds[random_indices])
        shuffle_random_mask[random_indices] = 2
        # We only compute loss on active tokens
        shuffle_random_mask[special_tokens_mask] = -100
        return input_ids, manipulated_input_embeds, shuffle_random_mask

    def get_special_tokens_mask(self, input_ids):
        ind = (1 - input_ids[:, 1, :]).sum(dim=1) - 2
        max_len = input_ids[:, 1, :].size(1)
        new_mat = torch.tensor([[1] + [0] * i + [1] * (max_len - 1 - i) for i in ind], dtype=torch.bool)
        return new_mat

    def random_indarrays(self, x):
        input_size = x.size()
        a = torch.randint(1, 1025, (input_size[0], input_size[1] - 1))
        b = torch.randint(1, 257, (input_size[0], 1))
        return torch.cat((a, b), dim=1)

    def proc_pos(self, x, y):
        base10 = int(x, base=36)
        if base10 >= 5:
            spec_ind = 5
            base2 = format(base10 - 5, '0>128b')
            ind_array = list(map(lambda ind: int(base2[ind: ind + 10], 2) + 1, [i for i in range(0, 128, 10)]))
        else:
            spec_ind = base10
            ind_array = [0] * 13
        return ind_array, spec_ind, 1-y

    def deep_proc(self, example):
        # base36 to base10
        temp_list = list(map(lambda x, y: self.proc_pos(x, y),
                             example['input_ids'].split(' '), example['attention_mask']))
        ind_array, spec_ind, padding_mask = list(list(zip(*temp_list)))
        return list(ind_array), [list(spec_ind), list(padding_mask)]

