# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            #この下のコードでエラー発生する(ValueError: could not convert string to float: '.')
            #2連続以上の語がうまく扱えていないことによるバグ？
            #先頭から調べて文字と数字を区別する処理を加える必要がある
            judge = True
            arg_word = ''
            index_num = 0
            while judge:
                try:
                    judge_word = float(tokens[index_num])
                    arg_words = arg_word
                    arg_index = index_num
                    judge = False
                except ValueError:
                    if index_num == 0:
                        arg_word += tokens[index_num]
                    else:
                        arg_word += ' ' + tokens[index_num]
                    index_num += 1
            word_vec[arg_words] = np.asarray(tokens[arg_index:], dtype='float32')
    return word_vec

#change function
def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        #ここが初期設定かな？
        #embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        embedding_matrix = np.random.uniform(-0.25,0.25,(len(word2idx) + 2, embed_dim))
        #変更点
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.840B.300d.txt' #ここ！
        #ここからエラー
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        #word_vec:{単語：単語ベクトル}  /  word2idx:{単語：単語ID}
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                #ここが多分未知語処理している部分
                #変更点
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    #注意
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name,deptype2id=None,dep_order="first"):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class DepInstanceParser():
    def __init__(self, basicDependencies, tokens):
        self.basicDependencies = basicDependencies
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()


    def dep_parsing(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                token['word'] = token
                words.append(self.change_word(token['word']))
            dep_governed_info = [
                {"word": word}
                for i,word in enumerate(words)
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1
            governed_index = dep['governor'] - 1
            dep_governed_info[dependent_index] = {
                "governor": governed_index,
                "dep": dep['dep']
            }
        self.dep_governed_info = dep_governed_info

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_first_order(self, direct=False):
        dep_adj_matrix  = [[0] * len(self.dep_governed_info) for _ in range(len(self.dep_governed_info))]
        dep_type_matrix = [["none"] * len(self.dep_governed_info) for _ in range(len(self.dep_governed_info))]
        # for i in range(len(self.dep_governed_info)):
        #     dep_adj_matrix[i][i]  = 1
        #     dep_type_matrix[i][i] = "self_loop"
        for i, dep_info in enumerate(self.dep_governed_info):
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)
        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_index, end_index, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        _, dep_path = self.search_dep_path(start_index, end_index, dep_type_matrix, [start_index])
        return dep_path

#['text_raw_indices', 'aspect_indices', 'aspect_in_text']
class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()

        self.datafile = fname
        self.depfile = "{}.dep".format(fname)
        self.tokenizer = tokenizer
        self.dep_order = dep_order
        self.dep_type2id = deptype2id
        self.textdata = ABSADataset.load_datafile(self.datafile)
        self.depinfo = ABSADataset.load_depfile(self.depfile)
        self.polarity2id = self.get_polarity2id()
        self.feature = []
        for sentence,depinfo in zip(self.textdata,self.depinfo):
            self.feature.append(self.create_feature(sentence,dep_info))


        #polarity_list = []
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
                'input_ids':self.feature["input_id"][i],
                'valid_ids':self.feature["input_id"][i],
                'segment_ids':self.feature["input_id"][i],
                'mem_valid_ids':self.feature["input_id"][i],
                'dep_adj_matrix':self.feature["input_id"][i],
                'dep_value_matrix':self.feature["input_id"][i]
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def ws(self, text):
        tokens = []
        valid_ids = []
        for i, word in enumerate(text):
            if len(text) <= 0:
                continue
            token = self.tokenizer.tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid_ids.append(1)
                else:
                    valid_ids.append(0)
        token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids, valid_ids

    def create_feature(self, sentence, depinfo):
        text_left, text_right, aspect, polarity = sentence

        cls_id = self.tokenizer.tokenizer.vocab["[CLS]"]
        sep_id = self.tokenizer.tokenizer.vocab["[SEP]"]

        doc = text_left + " " + aspect + " " + text_right

        left_tokens, left_token_ids, left_valid_ids = self.ws(text_left.split(" "))
        right_tokens, right_token_ids, right_valid_ids = self.ws(text_right.split(" "))
        aspect_tokens, aspect_token_ids, aspect_valid_ids = self.ws(aspect.split(" "))
        tokens = left_tokens + aspect_tokens + right_tokens
        input_ids = [cls_id] + left_token_ids + aspect_token_ids + right_token_ids + [sep_id] + aspect_token_ids + [sep_id]
        valid_ids = [1] + left_valid_ids + aspect_valid_ids + right_valid_ids + [1] + aspect_valid_ids + [1]
        mem_valid_ids = [0] + [0] * len(left_tokens) + [1] * len(aspect_tokens) + [0] * len(right_tokens)
        segment_ids = [0] * (len(tokens) + 2) + [1] * (len(aspect_tokens)+1)

        dep_instance_parser = DepInstanceParser(basicDependencies=depinfo, tokens=[])
        if self.dep_order == "first":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order()
        elif self.dep_order == "second":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_second_order()
        elif self.dep_order == "third":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_third_order()

        token_head_list = []
        for input_id, valid_id in zip(input_ids, valid_ids):
            if input_id == cls_id:
                continue
            if input_id == sep_id:
                break
            if valid_id == 1:
                token_head_list.append(input_id)

        final_dep_adj_matrix = [[0]*self.max_key_len for _ in range(self.tokenizer.max_seq_len)]
        final_dep_value_matrix = [[0]*self.max_key_len for _ in range(self.tokenizer.max_seq_len)]
        for i in range(len(token_head_list)):
            for j in range(len(dep_adj_matrix[i])):
                if j >= self.max_key_len:
                    break
                final_dep_adj_matrix[i+1][j] = dep_adj_matrix[i][j]
                final_dep_value_matrix[i+1][j] = self.deptype2id[dep_type_matrix[i][j]]

        input_ids = self.tokenizer.id_to_sequence(input_ids)
        valid_ids = self.tokenizer.id_to_sequence(valid_ids)
        segment_ids = self.tokenizer.id_to_sequence(segment_ids)
        mem_valid_ids = self.tokenizer.id_to_sequence(mem_valid_ids)

        return {
            "input_ids":torch.tensor(input_ids),
            "valid_ids":torch.tensor(valid_ids),
            "segment_ids":torch.tensor(segment_ids),
            "mem_valid_ids":torch.tensor(mem_valid_ids),
            "dep_adj_matrix":torch.tensor(final_dep_adj_matrix),
            "dep_value_matrix":torch.tensor(final_dep_value_matrix),
            "polarity": self.polarity2id[polarity],
            "raw_text": doc,
            "aspect": aspect
        }


    @staticmethod
    def load_depfile(filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data

    @staticmethod
    def load_datafile(filename):
        data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_right = text_right.replace("$T$", aspect)
                polarity = lines[i + 2].strip()
                data.append([text_left, text_right, aspect, polarity])

        return data

    @staticmethod
    def load_deptype_map(opt):
        deptype_set = set()
        for filename in [opt.train_file, opt.test_file, opt.val_file]:
            filename = "{}.dep".format(filename)
            if os.path.exists(filename) is False:
                continue
            data = ABSADataset.load_depfile(filename)
            for dep_info in data:
                for item in dep_info:
                    deptype_set.add(item['dep'])
        deptype_map = {"none": 0}
        for deptype in sorted(deptype_set, key=lambda x:x):
            deptype_map[deptype] = len(deptype_map)
        return deptype_map

    @staticmethod
    def get_polarity2id():
        polarity_label = ["-1","0","1"]
        return dict([(label, idx) for idx,label in enumerate(polarity_label)])