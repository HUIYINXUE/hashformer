from transformers import BatchEncoding
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

class DataCollatorForGlue:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self):
        self.collator_name = 'vanilla'

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

        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                "labels": batch["label"]}


########################################################################################################################


class DataCollatorForGlue_Proj:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self):
        self.collator_name = 'projection'

    def __call__(self, examples):
        tmp_tuple = list(map(lambda x: self.deep_proc(x), examples))
        inputs_embeds, input_ids = list(list(zip(*tmp_tuple)))
        # Handling of all keys.
        batch = {}
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in examples])
        batch['inputs_embeds'] = torch.tensor(list(inputs_embeds))
        batch['input_ids'] = torch.tensor(list(input_ids))
        batch['labels'] = torch.tensor([f['label'] for f in examples])
        return batch

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

class DataCollatorForGlue_Conj:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.
    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
        https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
    """

    def __init__(self):
        self.collator_name = 'conjunction'

    def __call__(self, examples):
        tmp_tuple = list(map(lambda x: self.deep_proc(x), examples))
        inputs_embeds, input_ids = list(list(zip(*tmp_tuple)))
        # Handling of all keys.
        batch = {}
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in examples])
        batch['inputs_embeds'] = torch.tensor(list(inputs_embeds))
        batch['input_ids'] = torch.tensor(list(input_ids))
        batch['labels'] = torch.tensor([f['label'] for f in examples])
        return batch

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


########################################################################################################################

def mc_collator_1(base, num_choices=2):
    class DataCollatorForMC(base):
        def __init__(self, n_choices=2):
            super().__init__()
            self.n_choices = n_choices

        def __call__(self, examples):
            if self.collator_name == 'vanilla':
                # vanilla collator
                if not isinstance(examples[0], (dict, BatchEncoding)):
                    examples = [vars(f) for f in examples]
                first = examples[0]
                examples = self.unpack_choices(examples)
                # Handling of all possible keys.
                # Again, we will use the first element to figure out which key/values are not None for this model.
                batch = {}
                for k, v in first.items():
                    if v is not None and not isinstance(v, str):
                        if isinstance(v, torch.Tensor):
                            batch[k] = torch.stack([f[k] for f in examples])
                        else:
                            batch[k] = torch.tensor([f[k] for f in examples])
            else:
                # compressive collator
                examples = self.unpack_choices(examples)
                tmp_tuple = list(map(lambda x: self.deep_proc(x), examples))
                inputs_embeds, input_ids = list(list(zip(*tmp_tuple)))
                # Handling of all keys.
                batch = {}
                batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in examples])
                batch['inputs_embeds'] = torch.tensor(list(inputs_embeds))
                batch['input_ids'] = torch.tensor(list(input_ids))
                batch['labels'] = torch.tensor([f['label'] for f in examples])
            return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                    "labels": batch["label"]}

        def unpack_choices(self, examples):
            unpacked_examples = []
            for item in examples:
                for j in range(self.n_choices):
                    each = {}
                    each['input_ids'] = item['input_ids'][j]
                    each['attention_mask'] = item['attention_mask'][j]
                    if j == item['label']:
                        each['label'] = 1
                    else:
                        each['label'] = 0
                    unpacked_examples.append(each)
            return unpacked_examples

    return DataCollatorForMC(num_choices)


########################################################################################################################
def mc_collator(base):
    class DataCollatorForMC(base):
        def __init__(self):
            super().__init__()

        def deep_proc(self, example):
            # base36 to base10
            n_choices = len(example['attention_mask'])
            embeds = []
            ids = []
            for i in range(n_choices):
                temp_list = list(map(lambda x, y: self.proc_pos(x, y),
                                 example['input_ids'][i].split(' '), example['attention_mask'][i]))
                ind_array, spec_ind, padding_mask = list(list(zip(*temp_list)))
                embeds.append(list(ind_array))
                ids.append([list(spec_ind), list(padding_mask)])
            return embeds, ids

    return DataCollatorForMC()

