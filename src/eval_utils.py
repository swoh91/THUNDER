from collections import defaultdict, Counter
from itertools import zip_longest
import numpy as np
import pandas as pd


def prf(pred, tp, true):
    p = tp / pred.clip(1)
    r = tp / true.clip(1)
    f = 2 * tp / (pred + true).clip(1)
    return p, r, f


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False
    if (prev_tag == 'B' or prev_tag == 'I') and (tag == 'B' or tag == 'O'):
        chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True
    return chunk_start


def get_entities(seq):
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_
    return chunks


def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for i in range(len(y_true)):
        for type_name, start, end in get_entities(y_true[i]):
            entities_true[type_name].add((i, start, end))
        for type_name, start, end in get_entities(y_pred[i]):
            entities_pred[type_name].add((i, start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return target_names, pred_sum, tp_sum, true_sum


def eval_tokens(y_true, y_pred, ignore_pos=True):
    pred = Counter()
    tp = Counter()
    true = Counter()
    for i in range(len(y_true)):
        for t, p in zip_longest(y_true[i], y_pred[i], fillvalue='O'):
            if t != 'O':
                if ignore_pos:
                    t = t[2:]
                true[t] += 1
            if p != 'O':
                if ignore_pos:
                    p = p[2:]
                pred[p] += 1
                if t == p:
                    tp[p] += 1
    return pred, tp, true


def expand_table(table):
    table.loc['ALL'] = table.sum()
    table['fn'] = table['true'] - table['tp']
    table['fp'] = table['pred'] - table['tp']
    table['precision'], table['recall'], table['f1'] = prf(table['pred'], table['tp'], table['true'])
    return table


def remove_pos(y_true):
    return [[tag.split('-', 1)[-1] for tag in s] for s in y_true]


def table_token(y_true, y_pred, ignore_pos=True):
    pred, tp, true = eval_tokens(y_true, y_pred, ignore_pos)
    table = pd.DataFrame({'pred': pred, 'tp': tp, 'true': true}).fillna(0).astype(int).sort_index()
    return expand_table(table)


def report_table(y_true, y_pred):
    target_names, pred, tp, true = extract_tp_actual_correct(y_true, y_pred)
    table = pd.DataFrame({'pred': pred, 'tp': tp, 'true': true}, target_names)
    return expand_table(table)


def performance_report(y_true, y_pred, output_dir, suffix=None, verbose=True):
    report = report_table(y_true, y_pred)
    report_token = table_token(y_true, y_pred)
    if verbose:
        print(f'{output_dir}/report{suffix}.txt')
        print(report.to_string(float_format='%.4f'))
    if suffix is not None:
        report.to_pickle(f'{output_dir}/report{suffix}.pkl')
        report.to_string(f'{output_dir}/report{suffix}.txt', float_format='%.4f')
        report_token.to_pickle(f'{output_dir}/report_token{suffix}.pkl')
        report_token.to_string(f'{output_dir}/report_token{suffix}.txt', float_format='%.4f')
    return report, report_token
