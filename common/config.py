r"""config"""
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation')

    # common
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='swin', choices=['resnet50', 'resnet101', 'swin'])
    parser.add_argument('--feature_extractor_path', type=str, default='')
    parser.add_argument('--logpath', type=str, default='./logs')

    # for train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    # for test
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vispath', type=str, default='./vis')
    parser.add_argument('--use_original_imgsize', action='store_true')

    # for model
    parser.add_argument('--original', type=str, default=False)
    # parser.add_argument('--add_loss', type=str, default=False)
    # parser.add_argument('--use_fpn', type=str, default=False)
    # parser.add_argument('--use_pool', type=str, default=False)
    # parser.add_argument('--new_mix_conv', type=str, default=False)
    # parser.add_argument('--cross_mix', type=str, default=False)
    # parser.add_argument('--add_gaussian', type=str, default=False)
    # parser.add_argument('--add_low', type=str, default=False)
    # parser.add_argument('--add_bottle_layer', type=str, default=False)
    # parser.add_argument('--new_skip', type=str, default=False)
    parser.add_argument('--add_4dconv', type=str, default=False)
    # parser.add_argument('--use_convnext', type=str, default=False)
    # parser.add_argument('--add_pool4d', type=str, default=False)
    # parser.add_argument('--skip_query_mask', type=str, default=False)
    # parser.add_argument('--use_aspp', type=str, default=False)
    # parser.add_argument('--upmix', type=str, default=False)
    # parser.add_argument('--multi_cross', type=str, default=False)
    # parser.add_argument('--adjcaent_cross', type=str, default=False)
    # parser.add_argument('--only_last', type=str, default=False)
    parser.add_argument('--skip_mode', type=str, default="concat")
    parser.add_argument('--pooling_mix', type=str, default="concat")
    parser.add_argument('--mixing_mode', type=str, default="concat")
    parser.add_argument('--mix_out', type=str, default="mixer3")
    parser.add_argument('--combine_mode', type=str, default="add")
    parser.add_argument('--model_mask', type=str, default="[1,2,3]")

    parser.add_argument('--weight', type=float, default=1.)

    args = parser.parse_args()
    return args