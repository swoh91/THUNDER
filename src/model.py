from transformers import RobertaForTokenClassification
from transformers.models.roberta.modeling_roberta import RobertaLMHead
import torch
from torch import nn
import torch.nn.functional as F


def ce_loss(log_probs, targets, weights=1, reduction='mean'):
    if log_probs.shape == targets.shape:
        losses = -(log_probs * targets).sum(-1) * weights
    else:
        losses = F.nll_loss(log_probs, targets, reduction='none') * weights
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()


def gce_loss(log_probs, targets, weights=1, reduction='mean', q=0.7):
    if log_probs.shape == targets.shape:
        losses = (targets * (1 - log_probs.exp() ** q) / q).sum(-1) * weights
    else:
        pred = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).exp()
        losses = (1 - pred ** q) / q * weights
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()


def split_probs(probs, clf_type=2, pos=False):
    if clf_type == 1:
        return probs,
    num_types = (probs.shape[1] - 1) // 2 if pos else (probs.shape[1] - 1)
    probs_o, probs_pos = probs[..., :1], 1 - probs[..., :1]
    probs_i, probs_b = probs[..., 1:1 + num_types], probs[..., 1 + num_types:]
    probs_type = ((probs_i + probs_b) if pos else probs_i) / probs_pos
    if clf_type == 2:
        if pos:
            probs2 = torch.cat([probs_o, probs_i.sum(-1, keepdim=True), probs_b.sum(-1, keepdim=True)], -1)
        else:
            probs2 = torch.cat([probs_o, probs_pos], -1)
        return probs_type, probs2
    probs3 = torch.stack([probs_i.sum(-1), probs_b.sum(-1)], -1) / probs_pos
    return probs_type, torch.cat([probs_o, probs_pos], -1), probs3


def combine_log_probs(lp1, lp2):
    return (lp1.unsqueeze(-1) + lp2.unsqueeze(-2)).flatten(-2)


def combine_probs(p1, p2):
    return (p1.unsqueeze(-1) * p2.unsqueeze(-2)).flatten(-2)


def pred_log_prob(logits):
    if not isinstance(logits, tuple):
        if logits.shape[-1] == 1:
            return torch.cat([F.logsigmoid(-logits), F.logsigmoid(logits)], -1)
        return logits.log_softmax(-1)
    lp_type, lp_pos = logits[0].log_softmax(-1), pred_log_prob(logits[1])
    if len(logits) == 2:
        return torch.cat([lp_pos[..., [0]], combine_log_probs(lp_pos[..., 1:], lp_type)], -1)
    return torch.cat([lp_pos[..., [0]], lp_pos[..., [1]] + combine_log_probs(pred_log_prob(logits[2]), lp_type)], -1)


def pred_prob(logits):
    if not isinstance(logits, tuple):
        if logits.shape[-1] == 1:
            return torch.cat([(-logits).sigmoid(), logits.sigmoid()], -1)
        return logits.softmax(-1)
    p_type, p_pos = logits[0].softmax(-1), pred_prob(logits[1])
    if len(logits) == 2:
        return torch.cat([p_pos[..., [0]], combine_probs(p_pos[..., 1:], p_type)], -1)
    return torch.cat([p_pos[..., [0]], p_pos[..., [1]] * combine_probs(pred_prob(logits[2]), p_type)], -1)


def conf_loss(log_probs_c, log_probs_w, valid_labels, th, w_conf, w_cw, w_nc, token_conf=False, token_unconf=False):
    conf_mask = valid_labels >= th
    conf_labels = valid_labels.masked_fill(~conf_mask, 0)
    is_conf = conf_mask.any(-1)
    if token_conf:
        conf_loss = ce_loss(log_probs_c[is_conf], valid_labels[is_conf])
        cw_loss = ce_loss(log_probs_w[is_conf], valid_labels[is_conf])
    else:
        conf_loss = ce_loss(log_probs_c[is_conf], conf_labels[is_conf])
        cw_loss = ce_loss(log_probs_w[is_conf], conf_labels[is_conf])
    if not is_conf.any().item():
        conf_loss = cw_loss = 0
    if token_unconf:
        non_conf_loss = 0 if is_conf.all() else ce_loss(log_probs_w[~is_conf], valid_labels[~is_conf])
    else:
        non_conf_loss = ce_loss(log_probs_w, valid_labels.masked_fill(conf_mask, 0))
    return w_conf * conf_loss + w_cw * cw_loss + w_nc * non_conf_loss


def dual_loss_prob(logits_c, logits_w, valid_labels, weak=False, th=None, **kwargs):
    log_probs_c, log_probs_w = pred_log_prob(logits_c), pred_log_prob(logits_w)
    if weak:
        log_probs = log_probs_w
    else:
        log_probs = log_probs_c
    if kwargs.get('gce', False):
        loss = gce_loss(log_probs, valid_labels)
    else:
        loss = ce_loss(log_probs, valid_labels)
    w_aux = kwargs.get('w_aux', 0)
    if w_aux > 0:
        if weak:
            loss = loss + w_aux * ce_loss(log_probs_c, valid_labels)
        else:
            loss = loss + w_aux * ce_loss(log_probs_w, valid_labels)
    if weak and th is not None:
        ws = kwargs.get('w_conf', 0), kwargs.get('w_cw', 0), kwargs.get('w_nc', 0), kwargs.get('token_conf', False), \
            kwargs.get('token_unconf', False)
        if isinstance(th, tuple):
            targets = split_probs(valid_labels, len(th), (th[1].shape[-1] == 3) or (len(th) == 3))
            loss = 0
            for i in range(len(targets)):
                loss = loss + conf_loss(pred_log_prob(logits_c[i]), pred_log_prob(logits_w[i]), targets[i], th[i], *ws)
        else:
            loss = conf_loss(log_probs_c, log_probs_w, valid_labels, th, *ws)
    return loss, log_probs_c


def l2_norm(x: torch.Tensor):
    return 0.5 * (x ** 2)


class DualNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mu_net = nn.Linear(in_dim, out_dim)
        self.sigma_net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.tanh(self.mu_net(x)), F.softplus(self.sigma_net(x)) + 1e-5

    def loss(self, x, log_probs, log_probs_w, labels=None):
        mu, sigma = self.forward(x)
        log_diff = log_probs_w - log_probs
        dist = l2_norm((mu - log_diff) / sigma)
        losses = log_diff + dist + sigma.log() + l2_norm(mu) + l2_norm(sigma)
        if labels is None:
            return losses.mean()
        if losses.shape == labels.shape:
            return (losses * labels).sum(-1).mean()
        return losses.gather(-1, labels.unsqueeze(-1)).mean()


class THUNDERModel(RobertaForTokenClassification):
    aug_clf = False
    clf2_size = 1
    combine_attention = False

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = RobertaLMHead(config)
        self.bin_classifier = nn.Linear(config.hidden_size, self.clf2_size)
        self.clf_weak = nn.Linear(config.hidden_size, config.num_labels)
        self.bin_clf_weak = nn.Linear(config.hidden_size, self.clf2_size)
        if self.aug_clf:
            self.clf_aug = nn.Linear(config.hidden_size, config.num_labels)
            self.bin_clf_aug = nn.Linear(config.hidden_size, self.clf2_size)
        self.dual_net = DualNet(config.hidden_size, config.num_labels + self.clf2_size)
        self.no_dual = False
        self.clf_type = 2
        self.init_weights()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward_h(self, input_ids, attention_mask, valid_pos):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        valid_output = sequence_output[valid_pos > 0]
        return self.dropout(valid_output)

    def forward_logits(self, sequence_output, weak=False, aug=False):
        if self.clf_type == 1:
            if aug and not self.no_dual:
                return self.clf_aug(sequence_output)
            if weak and not self.no_dual:
                return self.clf_weak(sequence_output)
            return self.classifier(sequence_output)
        if aug and not self.no_dual:
            return self.clf_aug(sequence_output), self.bin_clf_aug(sequence_output)
        if weak and not self.no_dual:
            return self.clf_weak(sequence_output), self.bin_clf_weak(sequence_output)
        return self.classifier(sequence_output), self.bin_classifier(sequence_output)

    def forward(self, input_ids, attention_mask, valid_pos, weak=False, aug=False):
        return self.forward_logits(self.forward_h(input_ids, attention_mask, valid_pos), weak, aug)

    def forward_log_prob(self, input_ids, attention_mask, valid_pos, weak=False, aug=False):
        return pred_log_prob(self.forward(input_ids, attention_mask, valid_pos, weak, aug))

    def forward_prob(self, input_ids, attention_mask, valid_pos, weak=False, aug=False):
        return pred_prob(self.forward(input_ids, attention_mask, valid_pos, weak, aug))

    def forward_hp(self, input_ids, attention_mask, valid_pos, log_prob=False):
        sequence_output = self.forward_h(input_ids, attention_mask, valid_pos)
        if log_prob:
            return sequence_output, pred_log_prob(self.forward_logits(sequence_output))
        return sequence_output, pred_prob(self.forward_logits(sequence_output))

    def dual(self, input_ids, attention_mask, valid_pos, valid_labels, weak=False, th=None, **kwargs):
        sequence_output = self.forward_h(input_ids, attention_mask, valid_pos)
        logits_c, logits_w = self.forward_logits(sequence_output), self.forward_logits(sequence_output, True)
        if 'aug_mask' in kwargs:
            aug_mask = kwargs['aug_mask'].unsqueeze(-1)
            logits_a = self.forward_logits(sequence_output, aug=True)
            if isinstance(logits_a, tuple):
                logits_w = tuple(torch.where(aug_mask, la, lw) for la, lw in zip(logits_a, logits_w))
            else:
                logits_w = torch.where(aug_mask, logits_a, logits_w)
        return dual_loss_prob(logits_c, logits_w, valid_labels, weak, th, **kwargs)
