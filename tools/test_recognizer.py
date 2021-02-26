import argparse

import torch
import mmcv
import tempfile
import os.path as osp
import torch.distributed as dist
import shutil
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict, get_dist_info
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel
from mmaction.apis import init_dist
from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy, non_mean_class_accuracy,
                                               mean_class_accuracy)
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np
import cv2
import copy
import os

def top_k_hit(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1

def top_k_accuracy(scores, labels, k=(1,)):
    res = []
    for kk in k:
        hits = []
        for x, y in zip(scores, labels):
            y = [y] if isinstance(y, int) else y
            hits.append(top_k_hit(x, set(y), k=kk)[0])
        res.append(np.mean(hits))
    return res

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def single_test(model, data_loader):
    args = parse_args()
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    count = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['get_logit'] = True
            v_img = data['img_group_0'].data[0]
            v_img = v_img.view(v_img.size(1), v_img.size(2), v_img.size(3), v_img.size(4), v_img.size(5))
            result = model(return_loss=False, **data)
            result, attention, t_attention = result
            if args.save_attention_map:
                c_att = attention.data.cpu()
                c_att = c_att.numpy()
                d_inputs = v_img.data.cpu()
                d_inputs = d_inputs.numpy()
                ct_att = t_attention.data.cpu()
                ct_att = ct_att.numpy()
                first = ct_att[0][0][0]
                last = ct_att[0][0][-1]               
                #ct_att[0][0][0] = np.amin(ct_att)
                #ct_att[0][0][-1] = np.amin(ct_att)
                
                ct_att[0][0][0] = np.mean(ct_att)
                ct_att[0][0][-1] = np.mean(ct_att)
                ct_att = min_max(ct_att)
                ct_att[0][0][0] = first
                ct_att[0][0][-1] = last
                
                in_b, in_c, in_f, in_y, in_x = v_img.shape

                for item_img, item_att, item_tatt in zip(d_inputs, c_att, ct_att):
                    for f in range(in_f):
                        att = item_att[:, f, :, :]
                        inputs = item_img[:, f, :, :]
                        t_att = item_tatt[:, f, :, :]
                        t_att_vis = t_att.item()
                        t_att_vis = (Decimal(str(t_att_vis)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
                        t_att = np.ones((100, 100)) * t_att
                        t_att = t_att.transpose((1, 2, 0)) * 255
                        t_att = np.uint8(t_att)
                        jet_tmap = cv2.applyColorMap(t_att, cv2.COLORMAP_JET)
                        jet_tmap = cv2.resize(jet_tmap, (224, 25))
                        pixelValue = jet_tmap[0, 0]
                        #cv2.rectangle(jet_tmap, (160, 0), (224, 25), (0, 0, 0), -1)
                        #cv2.putText(jet_tmap, str(t_att_vis), (170, 17), cv2.FONT_HERSHEY_SIMPLEX,
                         #   0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        mean = [0.485, 0.456, 0.406]
                        std = [0.229, 0.224, 0.225]
                        v_img = ((inputs.transpose((1, 2, 0)) * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255
                        v_img = v_img[:, :, ::-1]
                        resize_att = cv2.resize(att[0], (224, 224))
                        #resize_att = min_max(resize_att)
                        resize_att *= 255.
                        v_img = np.uint8(v_img)
                        vis_map = np.uint8(resize_att)

                        jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                        jet_map = cv2.addWeighted(v_img, 0.6, jet_map, 0.4, 0)
                        t_map = cv2.vconcat([jet_tmap, v_img])
                        #t_map = cv2.vconcat([v_img, jet_tmap])
                        out_dir = os.path.join('output')
                        raw_dir = os.path.join('output/raw')
                        att_dir = os.path.join('output/attention')
                        satt_dir = os.path.join('output/attention/spatial')
                        tatt_dir = os.path.join('output/attention/temporal')
                        if not os.path.exists(out_dir):
                            os.mkdir(out_dir)
                        if not os.path.exists(raw_dir):
                            os.mkdir(raw_dir)
                        if not os.path.exists(att_dir):
                            os.mkdir(att_dir)
                        if not os.path.exists(satt_dir):
                            os.mkdir(satt_dir)
                        if not os.path.exists(tatt_dir):
                            os.mkdir(tatt_dir)
                        out_path = os.path.join(out_dir, 'attention/spatial', '{0:06d}.png'.format(count))
                        cv2.imwrite(out_path, jet_map)
                        out_path = os.path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
                        cv2.imwrite(out_path, v_img)
                        out_path = os.path.join(out_dir, 'attention/temporal', '{0:06d}.png'.format(count))
                        cv2.imwrite(out_path, t_map)
                        count += 1

        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # data['get_logit'] = True
            v_img = data['img_group_0'].data[0]
            v_img = v_img.view(v_img.size(1), v_img.size(2), v_img.size(3), v_img.size(4), v_img.size(5))
            result = model(return_loss=False, rescale=True, **data)
            result, attention, t_attention = result

        results.append(result)

        if rank == 0:
            batch_size = data['img_group_0'].data[0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            print('temp_dir', tmpdir)
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoinls'
                                           't file')
    parser.add_argument(
        '--gpus', default=8, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--log', help='output log file')
    parser.add_argument('--fcn_testing', action='store_true', default=False,
                        help='whether to use fcn testing')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='whether to flip videos')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--ignore_cache', action='store_true', help='whether to ignore cache')
    parser.add_argument('-s', '--save_attention_map', dest='save_attention_map', action='store_true',
                        help='save Attention map')
    args = parser.parse_args()
    print('args==>>', args)
    return args


def main():
    args = parse_args()

    assert args.out, ('Please specify the output path for results')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if cfg.model.get('necks', None) is not None:
        cfg.model.necks.att_head_config = None

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8
    if args.fcn_testing:
        cfg.model['cls_head'].update({'fcn_testing': True})
        cfg.model.update({'fcn_testing': True})
    if args.flip:
        cfg.model.update({'flip': True})

    dataset = obj_from_dict(cfg.data.val, datasets, dict(test_mode=True))

    if args.ignore_cache and args.out is not None:
        if not distributed:
            if args.gpus == 1:
                model = build_recognizer(
                    cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
                load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
                model = MMDataParallel(model, device_ids=[0])

                data_loader = build_dataloader(
                    dataset,
                    imgs_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    num_gpus=1,
                    dist=False,
                    shuffle=False)
                outputs = single_test(model, data_loader)
            else:
                model_args = cfg.model.copy()
                model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
                model_type = getattr(recognizers, model_args.pop('type'))

                outputs = parallel_test(
                    model_type,
                    model_args,
                    args.checkpoint,
                    dataset,
                    _data_func,
                    range(args.gpus),
                    workers_per_gpu=args.proc_per_gpu)
        else:
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            model = build_recognizer(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)
    else:
        try:
            if distributed:
                rank, _ = get_dist_info()
                if rank == 0:
                    outputs = mmcv.load(args.out)
            else:
                outputs = mmcv.load(args.out)
        except:
            raise FileNotFoundError

    rank, _ = get_dist_info()
    if args.out:
        if rank == 0:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            gt_labels = []
            for i in range(len(dataset)):
                ann = dataset.get_ann_info(i)
                gt_labels.append(ann['label'])

            results = [res.squeeze() for res in outputs]
            top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
            #runner.mode = 'val'
            print("Top-1 Accuracy = {}".format(top1))
            print("Top-5 Accuracy = {}".format(top5))

if __name__ == '__main__':
    main()
