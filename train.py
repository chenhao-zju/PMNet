r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DAM import DAM
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset


def train(args, epoch, model, dataloader, optimizer, training, add_loss=True, k=1., nshot=1):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        if nshot==1:
            logit_mask = model(batch['query_img'], batch['query_mask'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        else:
            logit_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        if add_loss:
            logit_mask, mid_loss, _ = logit_mask
        
        
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if add_loss:
            loss = loss + k*mid_loss
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

        if not training:
            if Visualizer.visualize:
                Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                    batch['query_img'], batch['query_mask'],
                                                    pred_mask, batch['class_id'], idx,
                                                    iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()
    print(args)

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    args.model_mask = eval(args.model_mask)
    model = DAM(args.backbone, args.feature_extractor_path, False, original=args.original, add_4dconv=args.add_4dconv, 
                skip_mode=args.skip_mode, pooling_mix=args.pooling_mix,
                mixing_mode=args.mixing_mode, mix_out=args.mix_out, combine_mode=args.combine_mode, model_mask=args.model_mask)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)

    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.nshot)
    if args.local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)

        add_loss = args.add_loss or args.skip_query_mask or args.upmix
        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, dataloader_trn, optimizer, training=True, add_loss=add_loss, k=args.weight, nshot=args.nshot)

        # evaluation
        if args.local_rank == 0:
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(args, epoch, model, dataloader_val, optimizer, training=False, add_loss=add_loss, k=args.weight, nshot=args.nshot)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)

            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
            print(f'best_val_miou: {best_val_miou}')

    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
