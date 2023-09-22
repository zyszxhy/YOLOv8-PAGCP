import os
import torch
from prune import *
from ultralytics import YOLO
import torch_pruning as tp
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
import argparse
from ultralytics.yolo.utils import DEFAULT_CFG_DICT
from ultralytics.yolo.utils import IterableSimpleNamespace
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.files import increment_path
from pathlib import Path
import time

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)

def initialize_weights(model):
    # Initialize model weights to random values
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def initialize_weights_conv(model):
    # Initialize model weights to random values
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
           nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def compress(model, dataloader, args):
    compress_savedir = args.save_dir + '/compression'
    model_name = args.model.lower()  # yolov8n
    dataset_name = args.dataset.lower()
    compression_body = args.compression.lower()  # global
    compresion_method = args.prunemethod.lower()  # l1
    round_ = args.round  # 0
    topk = args.topk  # 1.0
    exp = args.exp  # True
    imgsz = args.imgsz  # 640

    if not os.path.exists(compress_savedir):
        os.makedirs(compress_savedir)
    model_path = os.path.join(compress_savedir,
                              f'pruned_{model_name}_{imgsz}_{compression_body}_{dataset_name}_r{round_}.pt')

    LOGGER.info('Start Pruning...')
    sens = Sensitivity(.05, .95, 19, compresion_method, round_, exp, topk, args, LOGGER)
    sen_dict = sens(model, dataloader, args.part)
    LOGGER.info('sensitivity:' + str(sen_dict))
    rate = sens.get_ratio(sen_dict)
    LOGGER.info('rate: ' + str(rate))
    pruned_model = deepcopy(model)
    strategy = tp.strategy.L1Strategy() if compresion_method == 'l1' else tp.strategy.L2Strategy()
    DG = DependencyGraph()
    DG.build_dependency(pruned_model, example_inputs=torch.randn(1, 3, imgsz, imgsz))

    start_time = time.time()
    layers1 = None
    for i, k in enumerate(rate.keys()):
        if 'group' in k:
            group_id = int(k[5:])
            group = sens.groups[group_id - 1]
            to_prune_list = group_l1prune(pruned_model, group, rate[k], round_to=1)
            layers = eval(
                f'pruned_model.module.{group[0]} if hasattr(pruned_model, "module") else pruned_model.{group[0]}')
            layers1 = eval(
                f'pruned_model.module.{group[-1]} if hasattr(pruned_model, "module") else pruned_model.{group[-1]}')
        else:
            layers = eval(f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
            to_prune_list = strategy(layers.weight, amount=rate[k], round_to=1)
        if isinstance(layers, torch.nn.Conv2d):
            prune_m = tp.prune_conv
        if layers1:
            pruning_plan1 = DG.get_pruning_plan(layers1,prune_m , idxs=to_prune_list)
            pruning_plan1.exec()
            layers1 = None
        pruning_plan = DG.get_pruning_plan(layers, prune_m, idxs=to_prune_list)
        pruning_plan.exec()

    LOGGER.info(f'prune duration: {time.time() - start_time}')
    torch.save(pruned_model, os.path.join(compress_savedir,
                                          f'pruned_{model_name}_{imgsz}_{compression_body}_{dataset_name}_r{round}.pt'))
    del sens, sen_dict
    gc.collect()

    return pruned_model


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n', type=str, help='The model to be compressed.')
    parser.add_argument('--dataset', default='NWPU', type=str, choices=['NWPU', 'COCO'], help='On which dataset the model is trained. VOC or COCO?')
    parser.add_argument('--compression', default='global', type=str, choices=['backbone', 'global'], help='To compress which part? backbone or all layers?')
    parser.add_argument('--prunemethod', default='L1', type=str, choices=['L1', 'L2'], help='The pruning algorithm for convolution layer.')
    parser.add_argument('--pruned', action='store_true', help='whether the checkpoint model have been pruned?')
    parser.add_argument('--round', default=0, type=int, help='the compression iteration of the network.')
    parser.add_argument('--topk', default=1.0, type=float, help='the filtering ratio P of target layers.')
    parser.add_argument('--exp', action='store_true', help='whether to compute the sensitivity in a sequential fashion')
    parser.add_argument('--initial_rate', default=0.07, type=float, help='the initial performance drop threshold for the first pruning layer')
    parser.add_argument('--initial_thres', default=30.0, type=float, help='the global performance drop threshold for the first pruning layer')
    parser.add_argument('--rate_slope', default=0., type=float, help='the adjustment slope of the initial masking ratio at each pruning iteration')
    parser.add_argument('--thres_slope', default=0., type=float, help='the adjustment slope of the initial performance drop threshold at each pruning iteration')
    parser.add_argument('--weights', type=str, default='runs/detect/train23/weights/best.pt', help='initial weights path')
    parser.add_argument('--pruned-model', type=str, default='', help='the path of the pruned model')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/NWPU.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args(args=[])
    return opt

def main(opt):
    model = YOLO('train23/weights/best.pt')
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)
    overrides = {}
    # overrides['model'] = 'yolov8n.yaml'
    # model.args = {**DEFAULT_CFG_DICT, **overrides}
    kwargs = {'data': 'NWPU.yaml', 'epochs': 120, 'imgsz': 640, 'batch': 2, 'lr0': 0.01}
    overrides.update(kwargs)
    # overrides['mode'] = 'train'
    model.model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **overrides})
    data = check_det_dataset(model.model.args.data)
    trainset, testset = data['train'], data['val']

    mode = 'val'
    val_loader = build_dataloader(model.model.args, 2, img_path=testset, stride=32, rank=-1, mode='val', rect=(mode == 'val'),
                     names=data['names'])[0]
    LOGGER.info('Compressing the model...')
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    pruned_model = compress(model.model, val_loader, opt)
    model.model = deepcopy(pruned_model)
    print(model)
    print(model.model)
    # initialize_weights_conv(pruned_model.model)
    model.train(data='NWPU.yaml', epochs=120, imgsz=640, batch=16,lr0=0.01,pruning=True)




if __name__ == "__main__":
    opt = parse_opt()
    opt.exp = True
    if opt.compression.lower() == 'backbone':
        opt.part = [f'model.{i}.' for i in range(10)]
    else:
        opt.part = [f'model.{i}.' for i in range(22)]
    main(opt)