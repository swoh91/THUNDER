import argparse
import os
import time

import torch

from model_utils import get_tokenizer
from utils import DataUtils
from trainer import THUNDERTrainer, get_classwise_th


def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--data_dir", default="conll", type=str)
    parser.add_argument("--pretrained_model", default='./roberta-base/', type=str)
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--temp_dir', type=str, default="tmp")
    parser.add_argument("--output_dir", default='out', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--tag_scheme", default='io', type=str, choices=['iob', 'io'])
    parser.add_argument("--bio", action='store_true')

    # training setting parameters
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--eval_on", default="test", choices=['valid', 'test'])
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--num_models", default=3, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weak_data', default='train', type=str)
    parser.add_argument('--supervision', default='dist', type=str)
    parser.add_argument('--train_size', default=1, type=float)
    parser.add_argument('--aug_data')
    parser.add_argument('--aug_supervision', default='true', type=str)
    parser.add_argument('--aug_clf', action='store_true')
    parser.add_argument('--aug_downscale', action='store_true')
    parser.add_argument('--aug_drop_ent', action='store_true')
    parser.add_argument('--aug_non_empty', action='store_true')
    parser.add_argument('--aug_only', action='store_true')
    parser.add_argument('--aug_semi', action='store_true')
    parser.add_argument('--strong_data', default='valid', type=str)
    parser.add_argument('--strong_supervision', default='true', type=str)
    parser.add_argument('--strong_seed', type=int, default=0)
    parser.add_argument('--strong_size', default=1, type=float)
    parser.add_argument('--strong_train', action='store_true')
    parser.add_argument('--ss_only', action='store_true')
    parser.add_argument('--ss_upscale', action='store_true')
    parser.add_argument('--ss_teacher', action='store_true')
    #
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--dual_train_lr", default=1e-5, type=float)
    parser.add_argument("--dual_train_epochs", default=3, type=int)
    parser.add_argument("--self_train_epochs", default=3, type=int)
    parser.add_argument("--token_conf", action='store_true')
    parser.add_argument("--token_unconf", action='store_true')
    parser.add_argument("--w_conf", default=1, type=float)
    parser.add_argument("--w_cw", default=0, type=float)
    parser.add_argument("--w_nc", default=0.1, type=float)
    parser.add_argument("--w_aux", default=1e-6, type=float)
    parser.add_argument("--gce_weak", action='store_true')
    parser.add_argument("--weak_first", action='store_true')
    parser.add_argument("--no_dual", action='store_true')
    parser.add_argument("--no_dual2", action='store_true')
    parser.add_argument("--retrain", default=1, type=int)
    parser.add_argument("--clf_type", default=2, type=int)
    parser.add_argument("--use_crf", action='store_true')
    parser.add_argument("--split_th", action='store_true')
    parser.add_argument("--th", default=-1, type=float)
    parser.add_argument("--th_s", default=0, type=float)
    parser.add_argument("--mc_dropout", default=0, type=int)
    parser.add_argument("--mc_seed", action='store_true')
    parser.add_argument("--weak_random", action='store_true')
    parser.add_argument("--weak_random_type", action='store_true')
    parser.add_argument("--weak_shift", action='store_true')
    parser.add_argument("--weak_shift_type", action='store_true')
    args = parser.parse_args()
    if args.clf_type == 3:
        args.tag_scheme = 'iob'

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    print(args)
    torch.save(args, f'{args.output_dir}/args.pt')
    torch.save(args, f'{args.temp_dir}/args.pt')

    tokenizer = get_tokenizer(args)
    processor = DataUtils(args.data_dir, tokenizer, args.tag_scheme)

    if args.do_train and not os.path.exists(f"{args.output_dir}/final_model.pt"):
        start = time.time()
        trainer = None
        # Training teachers
        args.is_teacher = True
        for i in range(args.num_models):
            trainer = THUNDERTrainer(args, tokenizer, processor)
            model = trainer.dual_train(f'dual_{i}')
            if model is not None:
                trainer.save_model(trainer.model, f'dual_{i}_model.pt', args.output_dir)
            args.seed = args.seed + 1
        # Training a student
        args.is_teacher = False
        if not os.path.exists(f"{args.output_dir}/final_model.pt"):
            if trainer is None or args.ss_teacher:
                trainer = THUNDERTrainer(args, tokenizer, processor)
            valid_probs = trainer.ensemble_pl()
            for r in range(args.retrain):
                trainer = THUNDERTrainer(args, tokenizer, processor)
                trainer.no_dual = trainer.no_dual or args.no_dual2
                trainer.model.no_dual = trainer.no_dual
                # Confident
                if args.w_conf > 0:
                    label_mask = trainer.train_labels[trainer.tensor_data['all_valid_pos'] > 0] >= 0
                    valid_labeled_probs = valid_probs[label_mask]
                    if args.split_th:
                        th = get_classwise_th(valid_labeled_probs, trainer.train_all_labels, args.th_s, args.clf_type,
                                              args.tag_scheme == 'iob')
                    else:
                        th = get_classwise_th(valid_labeled_probs, trainer.train_all_labels, args.th_s)
                        if args.th >= 0:
                            th[:] = args.th
                else:
                    th = None
                trainer.dual_train(f'self_{r}', trainer.get_labeled_dataset(valid_probs), th)
                valid_probs = trainer.get_valid_probs()
            trainer.save_model(trainer.model, f"final_model.pt", args.output_dir)
        train_time = time.time() - start
        print('train_time:', train_time)
        with open(f'{args.output_dir}/train_time.txt', 'w') as f:
            f.write(f'train_time: {train_time}\n')

    if args.do_eval:
        trainer = THUNDERTrainer(args, tokenizer, processor)
        trainer.load_model("final_model.pt", args.output_dir)
        trainer.log_init('final')
        trainer.report_eval_split('train', '_final')
        trainer.report_eval_split('valid', '_final')
        trainer.report_eval('_final', report_time=True)
        trainer.log_close()


if __name__ == "__main__":
    main()
