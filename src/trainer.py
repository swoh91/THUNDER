import json
import math
import os
import pickle
import random
import socket
import time
from functools import partial
from glob import glob

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (AdamW, get_linear_schedule_with_warmup, logging)
from tqdm import tqdm

from eval_utils import performance_report
from model import split_probs
from model_utils import get_tokenizer, get_thunder_model
from utils import DataUtils, to_dataset, subtensor, concat_tensor_data, write_lines_txt, read_lines_txt, \
    get_selected_all, drop_empty_examples, sublist

tqdm = partial(tqdm, ncols=0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_train_loader(train_data, batch_size):
    train_sampler = RandomSampler(train_data)
    drop_last = len(train_data) % batch_size / batch_size < 0.25
    return DataLoader(train_data, batch_size, sampler=train_sampler, drop_last=drop_last)


def roundrobin_longest(*iterables, n_iter=None):
    if n_iter is None:
        n_iter = max(len(it) for it in iterables)
    from itertools import cycle
    cycles = [cycle(it) for it in iterables]
    for i in range(n_iter):
        for it in cycles:
            yield it.__next__()


def ensemble_probs(probs_all):
    return [torch.stack(x).mean(0) for x in zip(*probs_all)]


def load_ensemble_probs(paths):
    return ensemble_probs([torch.load(p)['pred_probs'] for p in paths])


def probs_to_tags(probs, labels):
    return [[labels[i] for i in x.argmax(-1)] for x in probs]


def get_mean_std(x, s=0):
    return x.mean() + s * x.std()


def get_classwise_th(probs, train_labels, s=0, clf_type=1, pos=False):
    if clf_type == 1:
        counts = torch.bincount(train_labels)
        th = torch.zeros(probs.shape[-1])
        for i, c in enumerate(counts):
            th[i] = get_mean_std(probs[train_labels == i, i], s)
        return th
    if clf_type == 2:
        probs_type, probs2 = split_probs(probs, clf_type, pos)
    else:
        probs_type, probs2, probs3 = split_probs(probs, clf_type, pos)
    num_types = probs_type.shape[-1]
    th = torch.zeros(probs_type.shape[-1])
    for i in range(num_types):
        th[i] = get_mean_std(probs_type[(train_labels == 1 + i) | (train_labels == 1 + num_types + i), i], s)
    th2 = torch.zeros(probs2.shape[-1])
    th2[0] = get_mean_std(probs2[train_labels == 0, 0], s)
    th2[1] = get_mean_std(probs2[(train_labels > 0) & (train_labels < 1 + num_types), 1], s)
    if clf_type == 2:
        if pos:
            th2[2] = get_mean_std(probs2[train_labels > num_types, 2], s)
        return th, th2
    th3 = torch.zeros(2)
    th3[0] = get_mean_std(probs3[(train_labels > 0) & (train_labels < 1 + num_types), 0], s)
    th3[1] = get_mean_std(probs3[train_labels > num_types, 1], s)
    return th, th2, th3


class BaseTrainer(object):
    def __init__(self, args, tokenizer=None, processor=None):
        self.args = args
        self.seed = args.seed
        set_seed(args.seed)
        logging.set_verbosity_error()
        self.output_dir = args.output_dir
        self.data_dir = args.data_dir
        self.temp_dir = args.temp_dir

        self.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        self.eval_batch_size = args.eval_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_seq_length = args.max_seq_length

        self.warmup_proportion = args.warmup_proportion
        self.weight_decay = args.weight_decay

        if tokenizer is None:
            self.tokenizer = get_tokenizer(args)
        else:
            self.tokenizer = tokenizer
        if processor is None:
            self.processor = DataUtils(self.data_dir, self.tokenizer, args.tag_scheme)
        else:
            self.processor = processor
        self.label_map, self.inv_label_map = self.processor.get_label_map(args.tag_scheme)
        self.labels = self.processor.labels
        self.num_labels = len(self.inv_label_map) - 1  # exclude UNK type
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.mask_id = self.tokenizer.mask_token_id
        self.num_types = len(self.processor.entity_types)

        self.optimizer = None
        self.scheduler = None
        self.tb_writer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log_init(self, log_name):
        self.tb_writer = SummaryWriter(f'{self.output_dir}/{log_name}')
        configs = self.output_dir.split('/')[-3:]
        wandb.init(name=f'{socket.gethostname()}/{"/".join(configs)}/{log_name}')
        wandb.config.update({k: v for k, v in zip(('name', 'data', 'seed'), configs)})
        wandb.config['stage'] = log_name

    def log_close(self):
        self.tb_writer.close()
        self.tb_writer = None
        wandb.finish(quiet=True)

    def load_tensors(self, args):
        if args.do_train:
            self.true_tensor = self.processor.get_tensor(args.weak_data, self.max_seq_length)
            supervision = args.supervision
            if supervision in ('dist_0', 'fet_0.0'):
                supervision = 'true'
            elif supervision in ('dist_1', 'flip_0'):
                supervision = 'dist'
            self.dist_tensor = self.processor.get_tensor(args.weak_data, self.max_seq_length, supervision)
            self.n_dist = len(self.dist_tensor['all_idx'])
            if hasattr(args, 'drop_o_ratio'):
                self.processor.drop_o(self.dist_tensor, args.drop_o_ratio)
            self.dist_data = to_dataset(self.dist_tensor)
            self.tensor_data = self.dist_tensor
            self.ss_teacher = getattr(args, 'is_teacher', False) and getattr(args, 'ss_teacher', False)
            self.ss_only = getattr(args, 'ss_only', False) or self.ss_teacher
            self.ss_upscale = self.ss_only and getattr(args, 'ss_upscale', False)
            self.selected_all = None
            if hasattr(args, 'valid_size'):
                args.strong_data = 'valid'
                args.strong_supervision = 'true'
                args.strong_seed = -1
                args.strong_size = args.valid_size
                args.strong_train = args.valid_train
            if args.strong_size != 0:
                str_tensor = self.processor.get_tensor(args.strong_data, self.max_seq_length, args.strong_supervision)
                self.str_tensor = subtensor(str_tensor, args.strong_size, args.strong_seed)
                self.str_data = to_dataset(self.str_tensor)
                self.str_dataloader = DataLoader(self.str_data, args.eval_batch_size)
                self.ss_scale = len(self.true_tensor['all_idx']) / len(self.str_tensor['all_idx'])
                self.selected_all = get_selected_all(str_tensor, self.str_tensor)
                self.save_selected_all()
                if self.ss_only:
                    self.true_tensor = self.str_tensor
                    self.tensor_data = self.str_tensor
                if args.strong_train:
                    self.true_tensor = concat_tensor_data(self.true_tensor, self.str_tensor)
                    self.tensor_data = concat_tensor_data(self.tensor_data, self.str_tensor)
            self.shuffle_labels(args)

            self.aug_semi_teacher = getattr(args, 'is_teacher', False) and getattr(args, 'aug_semi', False)
            if getattr(args, 'aug_data', None) and not self.aug_semi_teacher:
                self.add_aug_tensor(args)
            all_input_ids = self.tensor_data["all_input_ids"]
            self.train_data = to_dataset(self.tensor_data)
            self.train_dataloader = DataLoader(self.train_data, batch_size=self.eval_batch_size)

            self.train_labels = self.tensor_data['all_labels']
            self.train_all_labels = self.train_labels[self.train_labels >= 0]
            self.train_bin_labels = (self.train_labels[self.train_labels >= 0] > 0).long()
            self.train_type_labels = self.train_labels[self.train_labels > 0]
            self.true_labels = self.true_tensor['all_labels']
            self.true_all_labels = self.true_labels[self.train_labels >= 0]
            self.true_bin_labels = (self.true_labels[self.train_labels >= 0] > 0).long()
            self.true_type_labels = self.true_labels[self.train_labels > 0]
            # print(f"***** Training - Num data = {all_input_ids.size(0)}, Batch size = {args.train_batch_size} *****")
        if args.do_eval:
            self.valid_tensor = self.processor.get_tensor('valid', self.max_seq_length)
            self.valid_tensor = subtensor(self.valid_tensor, 0.8, args.strong_seed, True)
            self.valid_data = to_dataset(self.valid_tensor)
            self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.eval_batch_size)
            #
            tensor_data = self.processor.get_tensor(args.eval_on, self.max_seq_length)
            all_input_ids = tensor_data["all_input_ids"]
            self.y_true = tensor_data["raw_labels"]
            self.y_true2 = None
            if os.path.exists(f'{args.data_dir}/test_label_true2.txt'):
                self.y_true2 = read_lines_txt(f'{args.data_dir}/test_label_true2.txt')
            eval_data = to_dataset(tensor_data)
            self.test_tensor = tensor_data
            self.test_data = eval_data
            self.eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            # print(f"***** Evaluation - Num data = {all_input_ids.size(0)}, Batch size = {args.eval_batch_size} *****")

    def add_aug_tensor(self, args):
        splits, supervisions = args.aug_data.split(':'), args.aug_supervision.split(':')
        assert len(splits) == len(supervisions)
        self.aug_tensor = None
        for aug_split, aug_supervision in zip(splits, supervisions):
            aug_tensor = self.processor.get_tensor(aug_split, self.max_seq_length, aug_supervision)
            sources_file = f'{args.data_dir}/{aug_split}_sources.txt'
            if args.strong_size < 1 and os.path.exists(sources_file):
                str_idxs = set(self.str_tensor['all_idx'].tolist())
                sources = read_lines_txt(sources_file)
                sub_idxs = np.array([bool(str_idxs.intersection(set(int(i) for i in s))) for s in sources]).nonzero()[0]
                aug_tensor = {k: sublist(v, sub_idxs) for k, v in aug_tensor.items()}
            if self.aug_tensor is None:
                self.aug_tensor = aug_tensor
            else:
                self.aug_tensor = concat_tensor_data(self.aug_tensor, aug_tensor)
        if getattr(args, 'aug_non_empty'):
            self.aug_tensor = drop_empty_examples(self.aug_tensor)
        if getattr(args, 'aug_drop_ent'):
            self.aug_tensor = self.processor.drop_entity_labels(self.aug_tensor)
        if getattr(args, 'aug_only'):
            self.true_tensor = self.tensor_data = self.aug_tensor
        else:
            self.true_tensor = concat_tensor_data(self.true_tensor, self.aug_tensor)
            self.tensor_data = concat_tensor_data(self.tensor_data, self.aug_tensor)

    def save_selected_all(self):
        selected_all_path = f'{self.output_dir}/selected_all.pt'
        if not os.path.exists(selected_all_path):
            torch.save(self.selected_all, selected_all_path)

    def shuffle_labels(self, args):
        all_valid_pos = self.tensor_data["all_valid_pos"]
        all_labels = self.tensor_data['all_labels']
        if getattr(args, 'weak_random', False):
            rng = np.random.default_rng(args.seed)
            random_labels = rng.integers(self.num_labels, size=all_valid_pos.sum().item())
            all_labels[all_valid_pos == 1] = torch.from_numpy(random_labels)
        elif getattr(args, 'weak_random_type', False):
            ent_labels = all_labels >= 1
            rng = np.random.default_rng(args.seed)
            random_labels = rng.integers(self.num_labels - 1, size=ent_labels.sum().item()) + 1
            all_labels[ent_labels] = torch.from_numpy(random_labels)
        elif getattr(args, 'weak_shift', False):
            valid_labels = all_labels >= 0
            shift_labels = (all_labels[valid_labels] + 1) % self.num_labels
            all_labels[valid_labels] = shift_labels
        elif getattr(args, 'weak_shift_type', False):
            ent_labels = all_labels >= 1
            shift_labels = (all_labels[ent_labels] % (self.num_labels - 1)) + 1
            all_labels[ent_labels] = shift_labels

    def step(self, loss, step):
        if loss.isnan():
            raise ValueError('loss: nan')
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        if (step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

    def is_max_step(self):
        return self.max_steps is not None and self.max_steps == self.optimizer._step_count

    # obtain model predictions on a given dataset
    @torch.no_grad()
    def eval(self, model, eval_dataloader, is_teacher=False):
        model = model.to(self.device)
        model.eval()
        y_pred = []
        pred_probs = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            _, input_ids, attention_mask, valid_pos, _ = tuple(t.to(self.device) for t in batch)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in \
                                                         (input_ids, attention_mask, valid_pos))

            mc_dropout = getattr(self.args, 'mc_dropout', 0)
            if is_teacher and mc_dropout > 0:
                if getattr(self.args, 'mc_seed', False):
                    torch.manual_seed(self.args.seed)
                model.train()
                type_prob = 0
                for m in range(mc_dropout):
                    type_prob += model.forward_prob(input_ids, attention_mask, valid_pos)
                type_prob /= mc_dropout
                model.eval()
            else:
                type_prob = model.forward_prob(input_ids, attention_mask, valid_pos)
            pred_prob = type_prob.cpu()
            preds = type_prob.argmax(-1).cpu().numpy()

            num_valid_tokens = valid_pos.sum(-1)
            i = 0
            for j in range(len(num_valid_tokens)):
                pred_probs.append(pred_prob[i:i + num_valid_tokens[j]])
                y_pred.append([self.inv_label_map[pred] for pred in preds[i:i + num_valid_tokens[j]]])
                i += num_valid_tokens[j]
        model.train()
        return y_pred, pred_probs

    # print out ner performance given ground truth and model prediction
    def performance_report(self, y_true, y_pred, suffix, verbose=True, split='test'):
        report, report_token = performance_report(y_true, y_pred, self.output_dir, suffix, verbose)
        if self.tb_writer is not None:
            prefix = f'{split}_' if split in ['train', 'valid'] else ''
            global_step = 0 if self.optimizer is None else self.optimizer._step_count
            for name in report:
                for k, v in report[name].items():
                    self.tb_writer.add_scalar(f'{prefix}{name}/{k}', v, global_step)
                    wandb.log({f'{prefix}{name}/{k}': v}, step=global_step)
            for name in report_token:
                for k, v in report_token[name].items():
                    self.tb_writer.add_scalar(f'token/{prefix}{name}/{k}', v, global_step)
                    wandb.log({f'token/{prefix}{name}/{k}': v}, step=global_step)

    def report_eval(self, suffix, verbose=True, report_time=False):
        print(f"****** Evaluating on {self.args.eval_on} set: {self.output_dir}/report{suffix} ******")
        start = time.time()
        y_pred, pred_probs = self.eval(self.model, self.eval_dataloader)
        eval_time = time.time() - start
        self.performance_report(self.y_true, y_pred, suffix, verbose)
        if self.y_true2 is not None:
            self.performance_report(self.y_true2, y_pred, f'{suffix}2', verbose)
        if verbose:
            pickle.dump(y_pred, open(f'{self.output_dir}/y_pred{suffix}.pkl', 'wb'))
            write_lines_txt(y_pred, f'{self.output_dir}/y_pred{suffix}.txt')
            torch.save({"pred_probs": pred_probs}, os.path.join(self.output_dir, f"y_pred{suffix}.pt"))
        if report_time:
            print('eval_time:', eval_time)
            with open(f'{self.output_dir}/eval_time.txt', 'w') as f:
                f.write(f'eval_time: {eval_time}\n')

    def report_eval_split(self, split='test', suffix='', verbose=True):
        suffix = f'_{split}{suffix}'
        if split == 'test':
            self.report_eval(suffix, verbose)
            return
        elif split == 'train':
            eval_dataloader = self.train_dataloader
            y_true = self.true_tensor['raw_labels']
        else:
            eval_dataloader = self.valid_dataloader
            y_true = self.valid_tensor['raw_labels']
        y_pred, pred_probs = self.eval(self.model, eval_dataloader)
        self.performance_report(y_true, y_pred, suffix, verbose, split)
        if verbose:
            pickle.dump(y_pred, open(f'{self.output_dir}/y_pred{suffix}.pkl', 'wb'))
            write_lines_txt(y_pred, f'{self.output_dir}/y_pred{suffix}.txt')
            torch.save({"pred_probs": pred_probs}, os.path.join(self.output_dir, f"y_pred{suffix}.pt"))

    def get_valid_probs(self, suffix=None, data=None):
        if data is None:
            data = self.train_data
        eval_dataloader = DataLoader(data, batch_size=self.args.eval_batch_size)
        y_pred, pred_probs = self.eval(self.model, eval_dataloader)
        performance_report(self.true_tensor['raw_labels'], y_pred, self.output_dir, suffix, suffix is not None)
        return torch.cat(pred_probs)

    # save model, tokenizer, and configs to directory
    def save_model(self, model, model_name, save_dir):
        print(f"Saving {model_name} ...")
        torch.save(self.args, os.path.join(save_dir, 'args.pt'))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, model_name))
        self.tokenizer.save_pretrained(save_dir)
        model_config = {"max_seq_length": self.max_seq_length,
                        "num_labels": self.num_labels,
                        "label_map": self.label_map}
        json.dump(model_config, open(os.path.join(save_dir, "model_config.json"), "w"))

    # load model from directory
    def load_model(self, model_name, load_dir):
        self.model.load_state_dict(torch.load(os.path.join(load_dir, model_name)), strict=False)


class THUNDERTrainer(BaseTrainer):
    def __init__(self, args, tokenizer=None, processor=None):
        super().__init__(args, tokenizer, processor)

        self.dual_train_lr = getattr(args, 'dual_train_lr', 1e-5)
        self.dual_train_epochs = args.dual_train_epochs
        self.self_train_epochs = args.self_train_epochs
        self.aug_clf = getattr(args, 'aug_clf', False)
        self.token_conf = getattr(args, 'token_conf', False)
        self.token_unconf = getattr(args, 'token_unconf', False)
        self.w_conf = getattr(args, 'w_conf', 1)
        self.w_cw = getattr(args, 'w_cw', 0)
        self.w_nc = getattr(args, 'w_nc', 0.1)
        self.w_aux = getattr(args, 'w_aux', 1e-6)
        self.no_dual = getattr(args, 'no_dual', False)
        self.gce_weak = getattr(args, 'gce_weak', False)
        self.weak_first = getattr(args, 'weak_first', False)

        # Prepare model
        self.model = get_thunder_model(self, args)
        self.model.no_dual = self.no_dual

        self.load_tensors(args)
        if self.ss_upscale:
            self.dual_train_epochs = int(self.dual_train_epochs * self.ss_scale)
            self.self_train_epochs = int(self.self_train_epochs * self.ss_scale)
        self.max_steps = None
        if getattr(args, 'aug_data', None) and getattr(args, 'aug_downscale', False):
            num_batches = round(self.n_dist / self.train_batch_size)
            self.max_steps = 2 * num_batches * (self.dual_train_epochs if args.is_teacher else self.self_train_epochs)
            aug_scale = self.n_dist / len(self.tensor_data['all_idx'])
            self.dual_train_epochs = math.ceil(self.dual_train_epochs * aug_scale)
            self.self_train_epochs = math.ceil(self.self_train_epochs * aug_scale)

    # prepare model, optimizer and scheduler for training
    def prepare_train(self, epochs, num_batches, lr=None):
        if lr is None:
            lr = self.dual_train_lr
        model = self.model.to(self.device)
        num_train_steps = int(num_batches / self.gradient_accumulation_steps) * epochs
        if self.max_steps and self.max_steps < num_train_steps:
            num_train_steps = self.max_steps
        params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(self.warmup_proportion * num_train_steps)
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters, lr, eps=1e-8)
        self.scheduler = scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
        model.train()
        return model, optimizer, scheduler

    def dual_train(self, log_name, data=None, th=None, skip_exist=True):
        if skip_exist and os.path.exists(f"{self.temp_dir}/y_pred_{log_name}.pt"):
            print(f"******* Model {log_name} predictions found; skip training *******\n")
            return
        self.log_init(log_name)
        loader_s = get_train_loader(self.str_data, self.train_batch_size)
        if data is None:
            data = self.train_data
            epochs = self.dual_train_epochs
        else:
            epochs = self.self_train_epochs
        loader_w = get_train_loader(data, self.train_batch_size)
        if th is not None:
            if isinstance(th, tuple):
                th = tuple(t.to(self.device) for t in th)
            else:
                th = th.to(self.device)
        num_batches = round(len(self.tensor_data['all_input_ids']) / self.train_batch_size)
        self.prepare_train(epochs, 2 * num_batches, self.dual_train_lr)
        for epoch in range(epochs):
            if self.weak_first:
                loader = roundrobin_longest(loader_w, loader_s, n_iter=num_batches)
            else:
                loader = roundrobin_longest(loader_s, loader_w, n_iter=num_batches)
            for step, batch in enumerate(tqdm(loader, f"[{log_name}] {epoch}/{epochs}", 2 * num_batches)):
                weak = step % 2 == (0 if self.weak_first else 1)
                self.train_step(batch, step, weak, th)
                if num_batches >= 2000 and (step + 1) % 1000 == 0:
                    self.report_eval_split('valid', f'_{log_name}', False)
                    self.report_eval(f'_{log_name}', False)
                if self.is_max_step():
                    break
            if self.args.do_eval:
                last_epoch = self.is_max_step() or (epoch + 1 == epochs)
                if last_epoch:
                    self.report_eval_split('train', f'_{log_name}')
                self.report_eval_split('valid', f'_{log_name}', last_epoch)
                self.report_eval(f'_{log_name}', last_epoch)
        self.log_close()
        if self.aug_semi_teacher:
            self.add_aug_tensor(self.args)
            self.train_data = to_dataset(self.tensor_data)
        if self.ss_teacher:
            eval_dataloader = DataLoader(self.dist_data, batch_size=self.eval_batch_size)
        else:
            eval_dataloader = DataLoader(self.train_data, batch_size=self.eval_batch_size)
        y_pred, pred_probs = self.eval(self.model, eval_dataloader, self.args.is_teacher)
        torch.save({"pred_probs": pred_probs}, f"{self.temp_dir}/y_pred_{log_name}.pt")
        # self.save_model(self.model, f"model_{log_name}.pt", self.temp_dir)
        return self.model

    def train_step(self, batch, step, weak=False, th=None):
        self.model.train()
        idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
        max_len = attention_mask.sum(-1).max().item()
        input_ids, attention_mask, valid_pos, labels = tuple(
            t[:, :max_len] for t in (input_ids, attention_mask, valid_pos, labels))
        valid_labels = labels[valid_pos > 0]
        gce = weak and self.gce_weak
        kwargs = {'gce': gce, 'w_aux': self.w_aux, 'w_conf': self.w_conf, 'w_cw': self.w_cw, 'w_nc': self.w_nc,
                  'token_conf': self.token_conf, 'token_unconf': self.token_unconf}
        if self.aug_clf:
            kwargs['aug_mask'] = (idx >= self.n_dist).unsqueeze(-1).expand_as(valid_pos)[valid_pos.bool()]
        loss, log_probs = self.model.dual(input_ids, attention_mask, valid_pos, valid_labels, weak, th, **kwargs)
        self.step(loss, step)

    def ensemble_pl(self):
        if self.args.num_models == 0:
            train_valid_labels = self.tensor_data['all_labels'][self.tensor_data['all_valid_pos'] == 1]
            return torch.nn.functional.one_hot(train_valid_labels, self.num_labels).float()
        ens_probs = load_ensemble_probs(sorted(glob(f'{self.temp_dir}/y_pred_dual_*.pt')))
        y_pred = probs_to_tags(ens_probs, self.labels)
        performance_report(self.true_tensor['raw_labels'], y_pred, self.output_dir, '_ens_pl')
        return torch.cat(ens_probs, 0)

    def get_labeled_dataset(self, valid_probs, data=None):
        if data is None:
            all_valid_pos = self.tensor_data["all_valid_pos"]
        else:
            all_valid_pos = data.tensors[-2]
        labels = torch.zeros(*all_valid_pos.shape, valid_probs.shape[-1])
        labels[all_valid_pos > 0] = valid_probs
        data = self.train_data
        tensors = list(data.tensors)
        tensors[-1] = labels
        return TensorDataset(*tensors)
