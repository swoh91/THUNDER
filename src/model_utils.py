from transformers import AutoTokenizer, RobertaTokenizer

from model import THUNDERModel


def get_default_model_name(args):
    if getattr(args, 'bio', False):
        return 'dmis-lab/biobert-v1.1'
    else:
        return './roberta-base/'


def get_tokenizer(args):
    if getattr(args, 'bio', False):
        return AutoTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False, cache_dir=args.cache_dir)
    else:
        return RobertaTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False, cache_dir=args.cache_dir)


def get_tokenizer_type(tokenizer):
    return not isinstance(tokenizer, RobertaTokenizer)


def get_thunder_model(trainer, args):
    kwargs = {'hidden_dropout_prob': args.dropout, 'attention_probs_dropout_prob': args.dropout,
              'cache_dir': args.cache_dir}

    clf_type = getattr(args, 'clf_type', 2)
    clf_size = trainer.num_labels if clf_type == 1 else trainer.num_types
    clf2_size = 3 if clf_type == 2 and args.tag_scheme == 'iob' else 2

    THUNDERModel.aug_clf = getattr(args, 'aug_clf', False)
    THUNDERModel.clf2_size = clf2_size
    model = THUNDERModel.from_pretrained(args.pretrained_model, num_labels=clf_size, **kwargs)
    model.clf_type = clf_type
    return model
