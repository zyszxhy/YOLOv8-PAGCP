import torch
import torch_pruning as tp
import numpy as np
from ultralytics.yolo.v8.detect.train import Loss
from ultralytics.yolo.utils import TQDM_BAR_FORMAT
from tqdm import tqdm
from ultralytics.yolo.utils.torch_utils import model_info
from scipy.optimize import fsolve
from functools import reduce
from copy import deepcopy
import gc
from .dependency import DependencyGraph
from .prune_zoo import *

def ratio_compute(initial_rate, layer_num, thres):
    def f(x, arg):
        it = [1 + arg[0] * pow(x, i) for i in range(arg[1])]  # it = [1+0.05*x^0,...]列表
        return reduce(lambda x, y: x * y, it) - arg[2]  # 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算

    return fsolve(f, 1, [initial_rate, layer_num, thres])[0]

class Sensitivity(object):
    """
    the core implementation of PAGCP.
    :param min_ratio: the initial masking ratio of each layer
    :param max_ratio: the maximal masking ratio of each layer
    :param num: the interval number between the initial and maximal masking ratio
    :param metric: filter saliency criterion
    :param round: pruning round
    :param exp: whether to scale the local performance drop of each layer
    :param topk: the filtering ratio
    :return: the pruned model
    """

    def __init__(self, min_ratio, max_ratio, num, metric, round_, exp, topk, *args):
        self.args = args[0]
        self.ratio = np.linspace(min_ratio, max_ratio, num)  # 0.05-0.95 19
        self.metric = tp.strategy.L1Strategy() if metric.lower() == 'l1' else tp.strategy.L2Strategy()
        self.exp = exp
        self.topk = topk
        self.inputsize = args[0].imgsz
        self.logger = args[1]
        self.round_ = round_
        self.func_rate = lambda x: args[0].initial_rate + args[
            0].rate_slope * x  # the computation of initial performance
        self.func_thres = lambda x: args[0].initial_thres + args[0].thres_slope * x
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_group(self, model):
        bottleneck_index = [2, 4, 6, 8]
        self.groups = [[f'model[{i}].m[{n}].cv2.conv' for n in
                        range(len(model.module[i].m if hasattr(model, 'module') else model[i].m))] + [
                           f'model[{i}].cv1.conv'] + [f'model[{i}].cv0.conv'] for i in bottleneck_index]
        return self.groups

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
        return batch

    def __call__(self, model, dataloader, part, sensitivity=None):
        if hasattr(self.metric, 'dataloader'): self.metric.dataloader = dataloader
        self.model = model
        self.set_group(model.model)
        self.criterion = Loss(model)
        self.temp_m = model.cuda()
        self.temp_m.eval()
        example_inputs = torch.randn(1, 3, self.inputsize, self.inputsize)
        base_b_total, base_o_total, base_c_total = 0., 0., 0.
        nb = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=nb, bar_format=TQDM_BAR_FORMAT)
        tloss = 0
        with torch.no_grad():
            for i, batch in pbar:
                batch = self.preprocess_batch(batch)
                preds = self.temp_m(batch['img'])
                imgs = batch['img']
                _, base_losses = self.criterion(preds, batch)
                #                 base_tlosses = (tloss * i + base_losses ) / (i + 1)
                base_b, base_o, base_c = base_losses[0] * imgs.shape[0], base_losses[1] * imgs.shape[0], base_losses[
                    2] * imgs.shape[0]
                base_b_total += base_b
                base_o_total += base_o
                base_c_total += base_c
        base_loss_total = base_b_total + base_o_total + base_c_total
        pruned_model = deepcopy(model).cuda()
        sensitivity = sensitivity if sensitivity is not None else {}
        #         del temp_m
        gc.collect()
        DG = DependencyGraph()
        thres = self.func_rate(self.round_)  # 0.05

        FLOPs_sens = {}
        # _, _, base_flops = self.model.cuda().info(verbose=True, self.inputsize)
        base_flops = model_info(model, imgsz=self.inputsize)
        # ----------layer sorting based on FLOPs---------- #
        for id, g in enumerate(self.groups):
            k = g[0]
            k1 = g[-1]
            DG.build_dependency(pruned_model, example_inputs=example_inputs)
            layers = eval(f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
            layers1 = eval(f'pruned_model.module.{k1} if hasattr(pruned_model, "module") else pruned_model.{k1}')
            prune_list = self.metric(layers.weight, amount=0.3, round_to=1)
            if len(prune_list) >= layers.weight.shape[0]:
                prune_list = prune_list[:-1]
            pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
            pruning_plan.exec()
            pruning_plan1 = DG.get_pruning_plan(layers1, tp.prune_conv, idxs=prune_list)
            pruning_plan1.exec()
            self.logger.info(f'model_id: {g}')
            temp_flops = model_info(pruned_model.cuda(), imgsz=self.inputsize)
            #             temp_flops =get_flops(pruned_model.cuda(),imgsz=self.inputsize)
            pruned_model = deepcopy(model).cuda()
            contrib_m = base_flops - temp_flops
            FLOPs_sens[f'group{id + 1}'] = contrib_m
        # _, _, base_flops = self.model.cuda().info(False, self.inputsize)
        #         base_flops = get_flops(model, imgsz=self.inputsize)
        base_flops = model_info(model, imgsz=self.inputsize)

        for k, v in model.named_modules():
            if hasattr(v, 'weight') and not isinstance(v, torch.nn.BatchNorm2d) and any(
                    [k.startswith(p) for p in part]) and k not in sensitivity.keys():
                DG.build_dependency(pruned_model, example_inputs=example_inputs)
                have_layers = [i.isdigit() for i in k.split('.')]  # 把数字取出来
                if any(have_layers):
                    model_id = []
                    for i, ele in enumerate(k.split('.')):
                        if have_layers[i]:
                            model_id[-1] = model_id[-1] + f'[{ele}]'
                        else:
                            model_id.append(ele)
                    model_id = '.'.join(model_id)
                else:
                    model_id = k
                if any([model_id in group for group in self.groups]):
                    continue
                else:
                    layers = eval(
                        f'pruned_model.module.{model_id} if hasattr(pruned_model, "module") else pruned_model.{model_id}')
                    prune_list = self.metric(layers.weight, amount=0.3, round_to=1)
                    if len(prune_list) >= layers.weight.shape[0]:
                        prune_list = prune_list[:-1]
                    if isinstance(v, torch.nn.Conv2d):
                        pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
                    else:
                        pruning_plan = None
                    pruning_plan.exec()
                self.logger.info(f'model_id: {model_id}')
                #                 _, _, temp_flops = pruned_model.cuda().info(False, self.inputsize)
                #                 temp_flops = get_flops(pruned_model.cuda(),imgsz=self.inputsize)
                temp_flops = model_info(pruned_model.cuda(), imgsz=self.inputsize)
                pruned_model = deepcopy(model).cuda()
                contrib_m = base_flops - temp_flops
                FLOPs_sens[model_id] = contrib_m

        exp = ratio_compute(thres, len(FLOPs_sens), self.func_thres(self.round_)) if self.exp else 1.0
        self.logger.info(f'lambda: {exp}')
        rank_modules = sorted(FLOPs_sens, key=lambda x: FLOPs_sens[x], reverse=exp <= 1)  # 按降序排列
        self.prune_sequence = rank_modules
        self.logger.info('prune_sequence:' + str(rank_modules))
        layers1 = None
        # 对于每一层遍历所有的剪枝率
        for num, k in enumerate(rank_modules):
            sensitivity[k] = {}
            sensitivity[k]['loss'] = []
            sensitivity[k]['base_loss'] = float(base_loss_total.data)
            for l, r in enumerate(self.ratio):
                self.logger.info(
                    f'pruning {num}/{len(rank_modules)}: {k}, base_loss:{base_loss_total:6f}, base_b:{base_b_total:6f}, base_o:{base_o_total:4f}, base_c:{base_c_total:6f}, ratio:{r}, thres:{thres}')
                temp_model = deepcopy(pruned_model)
                DG.build_dependency(temp_model, example_inputs=example_inputs)

                # get pruning set of each layer
                if 'group' in k:
                    group_id = int(k[5:])
                    group = self.groups[group_id - 1]
                    prune_list = group_l1prune(temp_model, group, r, round_to=1)
                    layers = eval(
                        f'temp_model.module.{group[0]} if hasattr(temp_model, "module") else temp_model.{group[0]}')
                    layers1 = eval(
                        f'temp_model.module.{group[-1]} if hasattr(temp_model, "module") else temp_model.{group[-1]}')
                else:
                    layers = eval(f'temp_model.module.{k} if hasattr(temp_model, "module") else temp_model.{k}')
                    prune_list = self.metric(layers.weight, amount=r, round_to=1)

                # execute the pruning
                if len(prune_list):
                    if len(prune_list) >= layers.weight.shape[0]:
                        prune_list = prune_list[:-1]
                    if isinstance(layers, torch.nn.Conv2d):
                        prune_m = tp.prune_conv
                    else:
                        prune_m = None
                    if layers1:
                        pruning_plan1 = DG.get_pruning_plan(layers1, prune_m, idxs=prune_list)
                        pruning_plan1.exec()
                        layers1 = None
                    pruning_plan = DG.get_pruning_plan(layers, prune_m, idxs=prune_list)
                    pruning_plan.exec()

                    temp_model = temp_model.cuda()  # 修剪后的模型
                    temp_b_total, temp_o_total, temp_c_total = 0., 0., 0.
                    pbar = tqdm(enumerate(dataloader), total=nb, bar_format=TQDM_BAR_FORMAT)
                    with torch.no_grad():
                        for i, batch in pbar:
                            batch = self.preprocess_batch(batch)
                            preds = temp_model(batch['img'])
                            _, temp_losses = self.criterion(preds, batch)
                            temp_b, temp_o, temp_c = temp_losses[0] * imgs.shape[0], temp_losses[1] * imgs.shape[0], \
                                                     temp_losses[2] * imgs.shape[0]

                            temp_b_total += temp_b
                            temp_o_total += temp_o
                            temp_c_total += temp_c

                    temp_loss_total = temp_b_total + temp_o_total + temp_c_total  # 计算损失
                    b_rel = temp_b_total / base_b_total  # 计算当前的损失与前一次模型损失的比值
                    o_rel = temp_o_total / base_o_total
                    c_rel = temp_c_total / base_c_total
                    self.logger.info(
                        f'temp_loss:{temp_loss_total:6f}, temp_b:{temp_b_total:6f}, temp_o:{temp_o_total:6f}, temp_c:{temp_c_total:6f}')

                    # ----------get the pruning ratio of each layer based on task choices---------- #
                    if max(b_rel, o_rel, c_rel) > (1 + thres):  # 说明该rate剪过头了，剪枝率rate[r-1]
                        idx = np.argmax([b_rel.cpu(), o_rel.cpu(), c_rel.cpu()])  # 选择剪枝后损失增大最多的任务
                        if 'group' in k:
                            group_id = int(k[5:])
                            group = self.groups[group_id - 1]
                            prune_list = group_l1prune(pruned_model, group, self.ratio[l - 1],
                                                       round_to=1) if l >= 1 else []
                            layers = eval(
                                f'pruned_model.module.{group[0]} if hasattr(pruned_model, "module") else pruned_model.{group[0]}')
                            layers1 = eval(
                                f'pruned_model.module.{group[-1]} if hasattr(pruned_model, "module") else pruned_model.{group[-1]}')
                        else:
                            layers = eval(
                                f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
                            prune_list = self.metric(layers.weight, amount=self.ratio[l - 1],
                                                     round_to=1) if l >= 1 else []
                        DG.build_dependency(pruned_model, example_inputs=example_inputs)
                        if len(prune_list):
                            if len(prune_list) >= layers.weight.shape[0]:
                                prune_list = prune_list[:-1]
                            if isinstance(layers, torch.nn.Conv2d):
                                prune_m = tp.prune_conv
                            else:
                                prune_m = None
                            pruning_plan = DG.get_pruning_plan(layers, prune_m, idxs=prune_list)
                            pruning_plan.exec()  # 根据当前剪枝率rate剪枝模型
                            if layers1:
                                pruning_plan1 = DG.get_pruning_plan(layers1, prune_m, idxs=prune_list)
                                pruning_plan1.exec()
                                layers1 = None


                            thres *= exp
                            base_loss_total, base_b_total, base_o_total, base_c_total = max(base_loss_total,
                                                                                            last_loss_total), max(
                                base_b_total, last_b_total), max(base_o_total, last_o_total), max(base_c_total,
                                                                                                  last_c_total)

                        sensitivity[k]['loss'].append(['box', 'object', 'class'][idx])
                        pruned_model = pruned_model.cuda()

                        break

                    sensitivity[k]['loss'].append(float(temp_loss_total))  # 最后的损失
                    last_loss_total, last_b_total, last_o_total, last_c_total = temp_loss_total, temp_b_total, temp_o_total, temp_c_total
                    del temp_model  # 由于这一参数不满足判断任务的阈值条件 ，所以删除该次的修剪模型
                else:
                    sensitivity[k]['loss'].append(float(base_b_total + base_o_total + base_c_total))
        del pruned_model
        gc.collect()
        return sensitivity

    def get_ratio(self, sensitivity):
        base_flops = model_info(self.model, imgsz=self.inputsize)

        pruned_model = deepcopy(self.model).cuda()
        DG = DependencyGraph()

        sens_keys = sensitivity.keys()
        flops = {}

        for k in sens_keys:
            if len(sensitivity[k]['loss']) > 1:
                DG.build_dependency(pruned_model, example_inputs=torch.randn(1, 3, self.inputsize, self.inputsize))
                r = self.ratio[len(sensitivity[k]['loss']) - 2]
                layers1 = None
                if 'group' in k:
                    group_id = int(k[5:])
                    group = self.groups[group_id - 1]
                    prune_list = group_l1prune(pruned_model, group, r, round_to=1)
                    layers = eval(
                        f'pruned_model.module.{group[0]} if hasattr(pruned_model, "module") else pruned_model.{group[0]}')
                    layers1 = eval(
                        f'pruned_model.module.{group[-1]} if hasattr(pruned_model, "module") else pruned_model.{group[-1]}')
                else:
                    layers = eval(f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
                    prune_list = self.metric(layers.weight, amount=r, round_to=1)
                if len(prune_list) >= layers.weight.shape[0]:
                    prune_list = prune_list[:-1]
                if layers1:
                    pruning_plan1 = DG.get_pruning_plan(layers1, tp.prune_conv, idxs=prune_list)
                    pruning_plan1.exec()
                    layers1 = None
                pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
                pruning_plan.exec()
                #                 _, _, fs = pruned_model.cuda().info(False, self.inputsize)
                fs = model_info(pruned_model, imgsz=self.inputsize)
                flops[k] = (sensitivity[k]['loss'][-2] - sensitivity[k]['base_loss']) / (base_flops - fs + 1e-20)
                base_flops = fs

        rank_keys = sorted(flops, key=lambda x: flops[x])  # 升序排列
        candidate_keys = sorted(rank_keys[:min(len(rank_keys), int(len(sens_keys) * self.topk))])
        sorted_index = sorted(
            [list(sens_keys).index(k) for k in candidate_keys])  # 根据candidate_keys判断sens_keys中有c_keys的索引
        ratio = {list(sens_keys)[i]: self.ratio[len(sensitivity[list(sens_keys)[i]]['loss']) - 2] for i in
                 sorted_index}  # if total, then - 1
        return ratio