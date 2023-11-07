import numpy as np
import hmac
import hashlib
import re
from transformers import RobertaTokenizerFast
from numpy import random
from collections import Counter
#from nltk.corpus import words
#from numba import jit



from numba import jit, njit
from numba.experimental import jitclass
import numba


"""
# two-way rotator
@njit(fastmath=True)
def compute_ind(rotator_vec, pair_mat):
    max_val = -1000000000000000000
    flag = 0
    max_idx = 0
    for i in range(25130):
        val = 0
        for ind, count in pair_mat:
            ind = int(ind)
            idx = ind + i
            # (idx * ((-1) ** (ind//25130)) + 25130) % 25130
            val += rotator_vec[idx % 25130]*count if ind < 25130 else rotator_vec[(-idx) % 25130]*count
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
            flag = 25130 if val < 0 else 0
            max_idx = i
    max_idx = max_idx + flag
    return max_idx
"""

"""
# one-way rotator
@njit(fastmath=True)
def compute_ind(rotator_vec, pair_mat):
    max_val = -10000000000000000
    max_idx = 0
    # np.arange(ind, 25130 + ind) % 50260
    for i in range(50260):
        val = 0
        for ind, count in pair_mat:
            val += rotator_vec[(int(ind)+i) % 50260] * count
        #val += np.sum(rotator_vec[(ind+i) % 50260] * count)
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx






"""
# one-way rotator
@njit(fastmath=True)
def compute_ind(rotator_vec, pair_mat):
    max_val = -10000000000000000
    max_idx = 0
    # np.arange(ind, 25130 + ind) % 50260
    for i in range(25130):
        val = 0
        for ind, count in pair_mat:
            val += rotator_vec[(int(ind)+i) % 50260] * count
        #val += np.sum(rotator_vec[(ind+i) % 50260] * count)
        if val > max_val:
            max_val = val
            max_idx = i
    # -self.rotator[np.arange(50259+ind, 25129+ind, -1) % 50260]
    for i in range(25130, 50260):
        val = 0
        for ind, count in pair_mat:
            val -= rotator_vec[(int(ind)+i) % 50260] * count
        #val -= np.sum(rotator_vec[(ind+i) % 50260] * count)
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx


"""

@jit(nopython=True, fastmath=False)
def compute_ind(rotator_vec, pair_mat):
    out_mat_1 = np.zeros(25130)
    out_mat_2 = np.zeros(25130)
    # np.arange(ind, 25130 + ind) % 50260
    for ind, count in pair_mat:
        out_mat_1 += rotator_vec[np.arange(int(ind), 25130 + int(ind)) % 50260] * count
        out_mat_2 -= rotator_vec[np.arange(50259+int(ind), 25129+int(ind), -1) % 50260] * count
    max_idx = np.argmax(np.concatenate((out_mat_1, out_mat_1), axis=0))
    return max_idx
"""





########################################################################################################################


class LSHValueTokenizer(object):
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.name_or_path = "LSH Value Tokenizer"
        self.CRYPT_KEY = str.encode('en')
        self.spec_token_dict = {
            '<s>': 0,
            '</s>': 2,
            '<mask>': 4,
            '<pad>': 1,
            '<unk>': 3
        }
        self.BPETokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        random.seed(42)
        self.rotator = np.random.normal(0, 1, 50260)
        #self.rotator = np.random.normal(0, 1, 25130)
        #self.hash_bag = HashBag()

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            sent_2 = self.tokenize(text[1])
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def convert_tokens_to_ids(self, token_list):
        return list(map(lambda token: self.spec_token_dict.get(token, self._get_lsh(token)), token_list))


    """
    def _get_lsh(self, word):
        variants_list = [word] + list(map(lambda i:
                                          ''.join(word[:i] + '-' + word[i:]),
                                          [k for k in range(1, len(word))]))
        variants_list = '-'.join(variants_list)
        c = Counter()
        c.update(self.BPETokenizer.tokenize(variants_list))
        if len(c.keys()) > 1:
            c.pop('-', None)  # not count `-`
        vec = np.sum(list(map(lambda item: self._projection(item, len(word) + 1), c.items())), axis=0)
        return np.base_repr(np.argmax(vec) + 5, base=36)
    """

    def _get_lsh(self, word):
        variants_list = [word] + list(map(lambda i:
                                          ''.join(word[:i] + '-' + word[i:]),
                                          [k for k in range(1, len(word))]))
        variants_list = '-'.join(variants_list)
        c = Counter()
        c.update(self.BPETokenizer.tokenize(variants_list))
        if len(c.keys()) > 1:
            c.pop('-', None)  # not count `-`
        word_len = len(word)+1
        count_mat = np.array(list(map(lambda x: self._get_weight(x, word_len), c.items())))
        #out = compute_ind(self.rotator, count_mat)
        #out = compute_ind(self.rotator, count_mat)
        out = compute_ind(self.rotator, count_mat)
        return out + 5



    def _get_weight(self, count_pair, word_len):
        w = count_pair[1] / max(word_len - len(count_pair[0]), 1)
        ind = self.BPETokenizer.convert_tokens_to_ids(count_pair[0])
        return [ind, w]



    def _projection(self, count_pair, word_len):
        ind = self.BPETokenizer.convert_tokens_to_ids(count_pair[0])
        r = np.concatenate((self.rotator[np.arange(ind, 25130+ind) % 50260],
                            -self.rotator[np.arange(50259+ind, 25129+ind, -1) % 50260]), axis=0)
        #### regularize by divide the rest length of word
        w = count_pair[1] / max(word_len-len(count_pair[0]), 1)
        return r * w




########################################################################################################################

class Md5ValueTokenizer(object):
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.name_or_path = "Md5 Value Tokenizer"
        self.CRYPT_KEY = str.encode('en')
        self.spec_token_dict = {
            '<s>': 0,
            '</s>': 2,
            '<mask>': 4,
            '<pad>': 1,
            '<unk>': 3
        }

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            sent_2 = self.tokenize(text[1])
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def convert_tokens_to_ids(self, token_list):
        return list(map(lambda token: self.spec_token_dict.get(token, self._get_md5_value(token)), token_list))

    def _get_md5_value(self, token):
        base10 = (int(hmac.new(self.CRYPT_KEY, str.encode(token), hashlib.md5).hexdigest(), 16) % 50260) + 5
        return base10

########################################################################################################################

class Md5Tokenizer(object):
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.name_or_path = "Md5 Tokenizer"
        self.CRYPT_KEY = str.encode('en')
        self.spec_token_dict = {
            '<s>': '0',
            '</s>': '2',
            '<mask>': '4',
            '<pad>': '1',
            '<unk>': '3'
        }

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            sent_2 = self.tokenize(text[1])
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = ' '.join(self.convert_tokens_to_ids(inputs))
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def convert_tokens_to_ids(self, token_list):
        return list(map(lambda token: self.spec_token_dict.get(token, self._get_md5(token)), token_list))

    def _get_md5(self, token):
        base10 = int(hmac.new(self.CRYPT_KEY, str.encode(token), hashlib.md5).hexdigest(), 16)
        return np.base_repr(base10 + 5, base=36)

########################################################################################################################

class LSHTokenizer(object):
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.name_or_path = "LSH Tokenizer"
        self.CRYPT_KEY = str.encode('en')
        self.spec_token_dict = {
            '<s>': '0',
            '</s>': '2',
            '<mask>': '4',
            '<pad>': '1',
            '<unk>': '3'
        }
        self.BPETokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            sent_2 = self.tokenize(text[1])
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = ' '.join(self.convert_tokens_to_ids(inputs))
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def convert_tokens_to_ids(self, token_list):
        return list(map(lambda token: self.spec_token_dict.get(token, self._get_proj(token)), token_list))

    def _get_proj(self, word):
        variants_list = [word] + list(map(lambda i:
                                          ''.join(word[:i] + '-' + word[i:]),
                                          [k for k in range(1, len(word))]))
        variants_list = '-'.join(variants_list)
        c = Counter()
        c.update(self.BPETokenizer.tokenize(variants_list))
        if len(c.keys()) > 1:
            c.pop('-', None)  # not count `-`
        vec = np.sum(list(map(lambda item: self._projection(item, len(word) + 1), c.items())), axis=0)
        vec = (vec > 0).astype(int).astype(str)
        base10 = int(''.join(vec), base=2)
        return np.base_repr(base10 + 5, base=36)

    def _projection(self, count_pair, word_len):
        random.seed(self.BPETokenizer.convert_tokens_to_ids(count_pair[0]))
        r = np.random.normal(0, 1, 128)
        #### regularize by divide the rest length of word
        w = count_pair[1] / max(word_len-len(count_pair[0]), 1)
        return r * w

########################################################################################################################

if __name__ == "__main__":
    text_a = 'hello world! My dog is cute. Natural Language Processing is really hard for me to learn.'
    text_b = 'hello my old friend'
    import time
    tokenizer = LSHValueTokenizer()
    time_1 = time.time()
    #a = torch.tensor(tokenizer([text_a] * 3, max_length=6, padding="max_length", truncation="longest_first")['input_embeds'])

    #print(a.size())
    #print((1-torch.tensor(tokenizer([text_a], max_length=10)['input_ids'])[:, 1, :]).sum(dim=1))

    ## get index
    #ind = tokenizer([text_a * 30]*100, max_length=512)['input_ids']
    #print(ind)

    print(tokenizer([text_a, text_b], max_length=10)['input_ids'])
    #print(tokenizer([text_a, text_b], max_length=6))

    #print(np.array(tokenizer([text_a, text_b], max_length=6)['input_embeds']).shape)
    #print(tokenizer([(text_a, text_b)], max_length=6))
    time_2 = time.time()
    print(time_2-time_1)