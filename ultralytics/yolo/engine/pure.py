import torch


import torch_pruning as tp
from functools import partial
from ultralytics.nn.modules import Detect
def get_pruner(model, example_inputs,methods):
    if methods == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=True)
    elif methods == "group_sl":
        #sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=0.0005, global_pruning=True)
    else:
        raise NotImplementedError

    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    for m in model.modules():
        if isinstance(m, (Detect)):
            ignored_layers.append(m)
        # ignore output layers
    # for  in list[model.model][-1]:
    #     if isinstance(m, torch.nn.Linear) and m.out_features == 10:
    #         ignored_layers.append(m)
    #     elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == 10:
    #         ignored_layers.append(m)

    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        ch_sparsity=0.5,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=1.0,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner