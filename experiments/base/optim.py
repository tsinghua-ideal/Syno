import torch
from torch import nn, optim
from timm.optim import create_optimizer_v2, optimizer_kwargs


def get_optimizer(model: nn.Module, args):
    return create_optimizer_v2(model, **optimizer_kwargs(cfg=args))


def get_gpt_optimizer(model: nn.Module, args):
    decay = set()
    no_decay = set()
    if "gpt" in args.model:
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif (pn.endswith("weight") and isinstance(m, whitelist_weight_modules)) or ("kernel" in pn and "weights" in pn):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
    else: # "rwkv" in args.model
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if p.dim() >= 2 and 'time_' not in mn:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "Parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "Parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer


def get_gnn_optimizer(model: nn.Module, args):
    return optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
