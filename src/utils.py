import itertools
import os
from collections import defaultdict

from torch.utils.data import TensorDataset
from tqdm import trange
import torch
import numpy as np
from model_utils import get_tokenizer_type


def read_lines_txt(filepath, split=True):
    with open(filepath) as f:
        if split:
            return [x[:-1].split(' ') for x in f]
        return [x[:-1] for x in f]


def write_lines_txt(lines, filepath, join=True):
    with open(filepath, 'w') as f:
        if join:
            f.writelines((f'{" ".join(x)}\n' for x in lines))
        else:
            f.writelines((f'{x}\n' for x in lines))


class DataUtils(object):

    def __init__(self, data_dir, tokenizer, tag_scheme):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.tag_scheme = tag_scheme
        self.read_types(self.data_dir)
        self.labels = None
        self.label_map = None
        self.inv_label_map = None
        self.get_label_map(tag_scheme)

    def read_file(self, file_dir="conll", dataset_name="train", supervision="dist"):
        text_file = os.path.join(file_dir, f"{dataset_name}_text.txt")
        f_text = open(text_file)
        text_contents = f_text.readlines()
        label_file = os.path.join(file_dir, f"{dataset_name}_label_{supervision}.txt")
        f_label = open(label_file)
        label_contents = f_label.readlines()
        sentences = []
        labels = []
        for text_line, label_line in zip(text_contents, label_contents):
            sentence = text_line.strip().split()
            label = label_line.strip().split()
            assert len(sentence) == len(label)
            sentences.append(sentence)
            labels.append(label)
        return sentences, labels

    def read_dict(self, dict_dir):
        _, _, dict_files = next(os.walk(dict_dir))
        self.entity_dict = defaultdict(list)
        self.entity_types = []
        for dict_file in dict_files:
            contents = open(os.path.join(dict_dir, dict_file)).readlines()
            entities = [content.strip() for content in contents]
            entity_type = dict_file.split('.')[0]
            self.entity_types.append(entity_type)
            for entity in entities:
                self.entity_dict[entity].append(entity_type)
            print(f"{entity_type} type has {len(entities)} entities")

        for entity in self.entity_dict:
            if len(self.entity_dict[entity]) > 1:
                print(self.entity_dict[entity])
                exit()

    def read_types(self, file_path):
        type_file = open(os.path.join(file_path, 'types.txt'))
        types = [line.strip() for line in type_file.readlines()]
        self.entity_types = []
        for entity_type in types:
            if entity_type != "O":
                self.entity_types.append(entity_type)

    def get_label_map(self, tag_scheme):
        if self.label_map is not None:
            return self.label_map, self.inv_label_map
        self.labels = ['O'] + [f'I-{t}' for t in self.entity_types]
        if tag_scheme == 'iob':
            self.labels += [f'B-{t}' for t in self.entity_types]
        label_map = {v: i for i, v in enumerate(self.labels)}
        label_map['UNK'] = -100
        inv_label_map = {k: v for v, k in label_map.items()}
        if tag_scheme == 'io':
            label_map.update({f'B-{t}': label_map[f'I-{t}'] for t in self.entity_types})
        self.label_map = label_map
        self.inv_label_map = inv_label_map
        return label_map, inv_label_map

    def get_data(self, dataset_name, supervision='true'):
        sentences, labels = self.read_file(self.data_dir, dataset_name, supervision)
        sent_len = [len(sent) for sent in sentences]
        print(f"{dataset_name} # words / #sents : {np.average(sent_len)} (avg) / {np.max(sent_len)} (max)")
        data = []
        for sentence, label in zip(sentences, labels):
            text = ' '.join(sentence)
            label = label
            data.append((text, label))
        return data

    def get_tensor(self, dataset_name, max_seq_length, supervision="true", drop_o_ratio=0):
        bio = get_tokenizer_type(self.tokenizer)
        data_file = os.path.join(self.data_dir, f"{dataset_name}_{supervision}.pt")
        if self.tag_scheme != "io":
            data_file = os.path.join(self.data_dir, f"{dataset_name}_{supervision}_{self.tag_scheme}.pt")
        if os.path.exists(data_file):
            # print(f"Loading data from {data_file}")
            tensor_data = torch.load(data_file)
        else:
            all_data = self.get_data(dataset_name=dataset_name, supervision=supervision)
            encodings = self.tokenizer.batch_encode_plus([x[0] for x in all_data], add_special_tokens=True,
                                                         max_length=max_seq_length,
                                                         padding='max_length', return_attention_mask=True,
                                                         truncation=True)
            raw_labels = []
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_valid_pos = []
            for i_text in trange(len(all_data), desc="Converting to tensors", ncols=0):
                labels = all_data[i_text][1]
                input_ids = encodings.input_ids[i_text]
                attention_mask = encodings.attention_mask[i_text]
                label_idx = -100 * torch.ones(max_seq_length, dtype=torch.long)
                valid_pos = torch.zeros(max_seq_length, dtype=torch.long)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                j = 0
                for i, token in enumerate(tokens[1:], start=1):  # skip [CLS]
                    if token == self.tokenizer.sep_token:
                        break
                    if ((not bio) and (i == 1 or token.startswith('Ġ'))) or (bio and not token.startswith('##')):
                        if j == len(labels):
                            break
                        label = labels[j]
                        label_idx[i] = self.label_map[label]
                        valid_pos[i] = 1
                        j += 1
                if i < max_seq_length - 1 and tokens[i] != self.tokenizer.sep_token:
                    continue
                if not (j == len(labels) or i == max_seq_length - 1):
                    continue
                assert j == len(labels) or i == max_seq_length - 1
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(label_idx.unsqueeze(0))
                all_valid_pos.append(valid_pos.unsqueeze(0))
                raw_labels.append(labels)

            all_input_ids = torch.LongTensor(all_input_ids)
            all_attention_mask = torch.LongTensor(all_attention_mask)
            all_labels = torch.cat(all_labels, dim=0)
            all_valid_pos = torch.cat(all_valid_pos, dim=0)
            all_idx = torch.arange(all_input_ids.size(0))
            tensor_data = {"all_idx": all_idx, "all_input_ids": all_input_ids, "all_attention_mask": all_attention_mask,
                           "all_labels": all_labels, "all_valid_pos": all_valid_pos, "raw_labels": raw_labels}
            print(f"Saving data to {data_file}")
            torch.save(tensor_data, data_file)
        return self.drop_o(tensor_data, drop_o_ratio)

    def drop_o(self, tensor_data, drop_o_ratio=0):
        if drop_o_ratio == 0:
            return tensor_data
        labels = tensor_data["all_labels"]
        rand_num = torch.rand(labels.size())
        drop_pos = (labels == 0) & (rand_num < drop_o_ratio)
        labels[drop_pos] = -100
        tensor_data["all_labels"] = labels
        return tensor_data

    def drop_entity_labels(self, tensor_data):
        tensor_data["all_labels"][tensor_data["all_labels"] > 0] = -100
        return tensor_data


def drop_empty_examples(tensor_data):
    non_empty_idxs = (tensor_data['all_labels'] > 0).any(-1).numpy().nonzero()[0]
    return {k: sublist(v, non_empty_idxs) for k, v in tensor_data.items()}


def concat_tensor_data(*ts):
    all_idx = torch.arange(sum(t['all_idx'].size(0) for t in ts))
    all_input_ids = torch.cat([t['all_input_ids'] for t in ts])
    all_attention_mask = torch.cat([t['all_attention_mask'] for t in ts])
    all_labels = torch.cat([t['all_labels'] for t in ts])
    all_valid_pos = torch.cat([t['all_valid_pos'] for t in ts])
    raw_labels = list(itertools.chain.from_iterable((t['raw_labels'] for t in ts)))
    tensor_data = {"all_idx": all_idx, "all_input_ids": all_input_ids, "all_attention_mask": all_attention_mask,
                   "all_labels": all_labels, "all_valid_pos": all_valid_pos, "raw_labels": raw_labels}
    return tensor_data


def shuffled_idxs(size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.permutation(size)


def sublist(vs, idxs):
    if isinstance(vs, list):
        return [vs[i] for i in idxs]
    return vs[idxs]


def subtensor(tensor_data, size, seed=None, remainder=False, by_token=False):
    if size < 0 or size == 1:
        return tensor_data
    if seed is None or seed < 0:
        if 0 < size < 1:
            size = int(len(tensor_data['all_idx']) * size)
        if remainder:
            return {k: v[size:] for k, v in tensor_data.items()}
        return {k: v[:size] for k, v in tensor_data.items()}
    idxs = shuffled_idxs(len(tensor_data['all_idx']), seed)
    if by_token:
        lv_arr = tensor_data['all_valid_pos'].sum(-1).numpy()
        token_size = lv_arr.sum() * size
        size = lv_arr[idxs].cumsum().searchsorted(token_size, side='right') + 1
    else:
        if size < 1:
            size = int(len(tensor_data['all_idx']) * size)
    sub_idxs = (idxs[size:] if remainder else idxs[:size]).copy()
    sub_idxs.sort()
    return {k: sublist(v, sub_idxs) for k, v in tensor_data.items()}


def get_selected_all(tensor_data, selected_tensor):
    len_valids = tensor_data['all_valid_pos'].sum(-1).tolist()
    offsets = torch.LongTensor([0] + len_valids).cumsum(-1)
    selected_all = torch.zeros(offsets[-1], dtype=torch.bool)
    for i in selected_tensor['all_idx']:
        selected_all[offsets[i]:offsets[i + 1]] = True
    return selected_all


def to_dataset(tensor_data):
    all_idx = tensor_data["all_idx"]
    all_input_ids = tensor_data["all_input_ids"]
    all_attention_mask = tensor_data["all_attention_mask"]
    all_valid_pos = tensor_data["all_valid_pos"]
    all_labels = tensor_data["all_labels"]
    return TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)
