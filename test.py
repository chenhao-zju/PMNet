r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import torch.nn as nn
import torch

from model.DAM import DAM
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        pred_mask = logit_mask.argmax(dim=1)

        assert pred_mask.size() == batch['query_mask'].size()


        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    args.model_mask = eval(args.model_mask)

    model = DAM(args.backbone, args.feature_extractor_path, False, original=args.original, add_4dconv=args.add_4dconv, 
                skip_mode=args.skip_mode, pooling_mix=args.pooling_mix,
                mixing_mode=args.mixing_mode, mix_out=args.mix_out, combine_mode=args.combine_mode, model_mask=args.model_mask)
    model.eval()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)
    # model = model.cuda()

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    pre_weight = model.state_dict()
    pre_dict = torch.load(args.load)

    pre_weight_keys = pre_weight.keys()

    non_loaded_state = []
    for key, value in pre_dict.items():
        if key in pre_weight_keys:
            pre_weight[key] = pre_dict[key]
        else:
            non_loaded_state.append(key)
    print(f'we have not use this model state_dict: {non_loaded_state}')
    print(f'we have loaded model from: {args.load}')
    model.load_state_dict(pre_weight)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
