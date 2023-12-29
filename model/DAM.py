r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from builtins import int
from functools import reduce
from operator import add
from xml.etree.ElementTree import TreeBuilder
from einops import rearrange

import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from timm.models.layers import DropPath


from .base.swin_transformer import SwinTransformer
from .base.transformer import MultiHeadedAttention, PositionalEncoding
from .base.conv4d import make_building_block

# from base.swin_transformer import SwinTransformer
# from base.transformer import MultiHeadedAttention, PositionalEncoding
# from base.conv4d import make_building_block


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mid_mixer_conv(in_channel, mid_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, mid_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU(),
                        nn.Conv2d(mid_channel, out_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU())

def detect_head_conv(in_channel, mid_channel):
    return nn.Sequential(nn.Conv2d(in_channel, mid_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU(),
                        nn.Conv2d(mid_channel, 2, (3, 3), padding=(1, 1), bias=True))


def mix_conv2d(inch, outch, ksz, stride, pad, group=4):
    return nn.Sequential(nn.Conv2d(inch, outch, kernel_size=ksz, stride=stride, padding=pad, bias=True),
                        nn.GroupNorm(group, outch),
                        nn.ReLU(inplace=True))

class upchannel_block(nn.Module):
    def __init__(self, in_channel, out_channels=[16, 32, 64], group=4) -> None:
        super(upchannel_block, self).__init__()
        outch1, outch2, outch3 = out_channels
        self.conv1 = mix_conv2d(in_channel, outch1, ksz=3, stride=1, pad=1, group=group)
        self.conv23 = mix_conv2d(outch1, outch2-outch1, ksz=3, stride=1, pad=1, group=group)
        self.conv21 = mix_conv2d(outch1, outch1, ksz=1, stride=1, pad=0, group=group)
        self.conv33 = mix_conv2d(outch2, outch3-outch2, ksz=3, stride=1, pad=1, group=group)
        self.conv31 = mix_conv2d(outch2, outch2, ksz=1, stride=1, pad=0, group=group)
        self.conv4 = mix_conv2d(outch3, outch3, ksz=1, stride=1, pad=0, group=group)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.concat([self.conv23(x1), self.conv21(x1)], dim=1)
        x3 = torch.concat([self.conv33(x2), self.conv31(x2)], dim=1)
        x4 = self.conv4(x3)
        return x4

class samechannel_block(nn.Module):
    def __init__(self, in_channel, group=4) -> None:
        super(samechannel_block, self).__init__()
        outch1, outch2 = in_channel//2, in_channel//4
        self.conv13 = mix_conv2d(outch1, outch1, ksz=3, stride=1, pad=1, group=group)
        self.conv11 = mix_conv2d(outch1, outch1, ksz=1, stride=1, pad=0, group=group)
        self.conv23 = mix_conv2d(outch2, outch2, ksz=3, stride=1, pad=1, group=group)
        self.conv21 = mix_conv2d(outch2, outch2, ksz=1, stride=1, pad=0, group=group)
        self.conv33 = mix_conv2d(outch2, outch2, ksz=3, stride=1, pad=1, group=group)
        self.conv4 = mix_conv2d(in_channel, in_channel, ksz=3, stride=1, pad=1, group=group)

    def forward(self, x):
        x11, x13 = torch.chunk(x, chunks=2, dim=1)
        x13 = self.conv13(x13)
        x11 = self.conv11(x11)
        x21, x23 = torch.chunk(x13, chunks=2, dim=1)
        x23 = self.conv23(x23)
        x21 = self.conv21(x21)
        x33 = self.conv33(x23)
        x4 = torch.cat([x11, x21, x33], dim=1)
        x5 = self.conv4(x4)
        return x5


class DAM(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, original=True, 
                        add_4dconv=False, skip_mode='concat', 
                        pooling_mix='concat', mixing_mode='concat', mix_out='mixer3', combine_mode='add', model_mask=[1,2,3]):
        super(DAM, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        self.original = original

        self.add_4dconv = add_4dconv

        self.skip_mode = skip_mode
        self.pooling_mix = pooling_mix
        self.mixing_mode = mixing_mode
        self.mix_out = mix_out
        self.combine_mode = combine_mode
        self.model_mask = model_mask

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)          # self.nlayers = [a, b, c, d] --> [a, a+b, a+b+c, a+b+c+d]
        self.model = DAM_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, original=self.original, 
                            add_4dconv=self.add_4dconv, skip_mode=self.skip_mode, pooling_mix=self.pooling_mix, 
                            mixing_mode=self.mixing_mode, combine_mode=self.combine_mode, model_mask=self.model_mask)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, query_masks, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)

        logit_mask = self.model(query_feats, query_masks, support_feats, support_mask.clone() )

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        query_mask = batch['query_mask']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, query_mask, support_imgs.squeeze(1), support_masks.squeeze(1))
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_feats, query_mask, n_support_feats, support_masks.clone(), nshot)


        return logit_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class DAM_model(nn.Module):
    def __init__(self, in_channels, stack_ids, original=False, add_4dconv=False, skip_mode='mix', pooling_mix='concat', 
                        mixing_mode='concat', mix_out='mixer3', combine_mode='add', model_mask=[1,2,3]):
        super(DAM_model, self).__init__()

        self.model_mask = model_mask

        self.stack_ids = stack_ids
        self.original = original


        self.add_4dconv = add_4dconv

        self.skip_mode = skip_mode
        self.pooling_mix = pooling_mix
        self.mixing_mode = mixing_mode
        self.mix_out = mix_out

        self.combine_mode = combine_mode



        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3 = 16, 32, 64
        self.feature_dim = [96, 48, 24, 12]

        self.head = 4



        # DAM blocks
        if self.add_4dconv:
            self.linears = nn.ModuleDict()
            self.conv_4d_blocks = nn.ModuleDict()
        else:
            self.DAM_blocks = nn.ModuleDict()
        
        self.pe = nn.ModuleDict()
        for idx in self.model_mask:
            inch = in_channels[idx]
            pe_dim = inch
            if self.add_4dconv:
                print('add 4D conv')
                in_channel = stack_ids[idx]-stack_ids[idx-1]
                self.linears[str(idx)] = clones(nn.Linear(pe_dim, pe_dim), 2)

                self.conv_4d_blocks[str(idx)] = make_building_block(in_channel, [outch1,outch1], kernel_sizes=[5,3], spt_strides=[1,1], type='6dconv')
               

            else:
                print('original')
                self.DAM_blocks[str(idx)] = MultiHeadedAttention(h=8, d_model=inch, dropout=0.5, add_gaussian=self.add_gaussian, add_bottle_layer=self.add_bottle_layer)
           
            self.pe[str(idx)] = PositionalEncoding(d_model=pe_dim, dropout=0.5)



        # conv blocks
        self.conv_layer = nn.ModuleDict()

        self.key = 1
        for idx in self.model_mask:
            in_channel = self.key*(stack_ids[idx]-stack_ids[idx-1]) if idx>0 else self.key*stack_ids[idx]
            if self.add_4dconv: in_channel = outch1

            self.conv_layer[str(idx)] = upchannel_block(in_channel=in_channel, out_channels=[outch1, outch2, outch3])


        if self.combine_mode == 'add':
            print('add')
            in_channel4 = in_channel5 = in_channel6 = outch3
        elif self.combine_mode == 'concat':
            print('concat')
            in_channel4 = 2*outch3 if 1 in self.model_mask else outch3
            in_channel5 = 2*outch3 if 2 in self.model_mask else outch3
            in_channel6 = 2*outch3 if 3 in self.model_mask else outch3

        # name of layer is 4-6 from top to bottom
        if 2 in self.model_mask or 3 in self.model_mask:
            # self.conv4 = self.build_conv_block(in_channel4, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
            # self.conv4 = self.build_conv_block(in_channel4, [outch3, outch3], [3, 3], [1, 1]) # 1/32 + 1/16
            self.conv4 = samechannel_block(in_channel=in_channel4)
        if 1 in self.model_mask or 2 in self.model_mask:
            # self.conv5 = self.build_conv_block(in_channel5, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8
            # self.conv5 = self.build_conv_block(in_channel5, [outch3, outch3], [3, 3], [1, 1]) # 1/16 + 1/8
            self.conv5 = samechannel_block(in_channel=in_channel5)
        # self.conv6 = self.build_conv_block(in_channel6, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8
        # self.conv6 = samechannel_block(in_channel=in_channel6)


        

        # mixer blocks
        # in_channel = outch3+2*in_channels[1]+2*in_channels[0]
        # in_channel = in_channel if self.mixing_mode=='concat' else outch3
        # in_channel = in_channel
        in_channel = outch3+in_channels[1]+in_channels[0]

        self.mixer1 = detect_head_conv(in_channel, outch3) if self.mix_out=='mixer1' else mid_mixer_conv(in_channel, outch3, outch2)

        if self.mix_out=='mixer2' or self.mix_out=='mixer3':
            print('mixer_output2')
            self.mixer2 = detect_head_conv(outch2, outch2) if self.mix_out=='mixer2' else mid_mixer_conv(outch2, outch2, outch1)

        if self.mix_out=='mixer3':
            print('mixer_output3')
            self.mixer3 = detect_head_conv(outch1, outch1)


    def compute_parameter(self):
        para = 0
        for idx in self.model_mask:
            for m in self.conv_4d_blocks[idx]:
                para += len(m.weight.view(-1))

    def forward(self, query_feats, query_masks, support_feats, support_mask, nshot=1):
        coarse_masks = {i:[] for i in self.model_mask}
        target_masks = {}
        support_masks = {}
        coarse_similaritys = {i:[] for i in self.model_mask}
        low_query_feats = {}
        upsample_times = 2
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx<self.stack_ids[0]: 
                if 0 not in self.model_mask: continue
                else: key = '0'
            if idx<self.stack_ids[1] and idx>=self.stack_ids[0]: 
                if 1 not in self.model_mask: continue
                else: key = '1'
            if idx<self.stack_ids[2] and idx>=self.stack_ids[1]: 
                if 2 not in self.model_mask: continue
                else: key = '2'
            if idx<self.stack_ids[3] and idx>=self.stack_ids[2]: 
                if 3 not in self.model_mask: continue
                else: key = '3'

            # query_feat = self.q_bottlenecks[key](query_feat)

            bsz, ch, ha, wa = query_feat.size()
            query_mask = F.interpolate(query_masks.unsqueeze(1).float(), (ha, wa), mode='bilinear',
                                     align_corners=True)
            q_mask = query_mask.view(bsz, -1).unsqueeze(-1)

            # reshape the input feature and mask
            query_feat_reshape = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                # support_feat = self.s_bottlenecks[key](support_feat)
                s_mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True)
                # support_feat = support_feat * s_mask
                mask = s_mask.view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                # support_feat = torch.stack([self.s_bottlenecks[key](support_feats[k][idx]) for k in range(nshot)])
                # support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                if self.qs_mix:
                    mid_feature = rearrange(support_feat.max(dim=0).values, 'b c h w -> b (h w) c').contiguous()
                support_feat = rearrange(support_feat, 'n b c h w -> (n b) (h w) c').contiguous()
                s_mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask], dim=1)
                # mask = s_mask.view(bsz, -1)
                mask = rearrange(s_mask, 'n b c h w -> b (n c h w)')


            # positioned embedding
            query = self.pe[key](query_feat_reshape)
            support = self.pe[key](support_feat)


            if self.add_4dconv:
                query, support = [l(x) for l, x in zip(self.linears[key], (query, support))]
                query = rearrange(query, 'b p (head c) -> b head p c', head=self.head)
                support = rearrange(support, '(n b) p (head c) -> b head (n p) c', n=nshot, head=self.head)
                similarity = query@support.transpose(-1, -2)
                if self.add_5dconv:
                    similarity = rearrange(similarity, 'b head (h1 w1) (n h2 w2) -> (n b) head h1 w1 h2 w2', h1=ha, w1=wa, n=nshot, h2=ha, w2=wa)
                else:
                    similarity = rearrange(similarity, 'b head (h1 w1) (n h2 w2) -> (n head b) h1 w1 h2 w2', h1=ha, w1=wa, n=nshot, h2=ha, w2=wa)
                coarse_similaritys[int(key)].append(similarity)
                support_masks[int(key)] = mask
            else:
                coarse_mask = self.DAM_blocks[key](query, support, mask, nshot=nshot)
                coarse_masks[int(key)].append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, ha, wa))

            if idx+1 in self.stack_ids:
                target_masks[int(key)] = query_mask.squeeze(1)

        
        if self.add_4dconv:
            coarse_masks = {}
            for idx in self.model_mask:
                bsz, ch, ha, wa = query_feats[self.stack_ids[idx] - 1].size()
                ori_similarity = torch.stack(coarse_similaritys[idx], dim=1).contiguous()
                # print(mix_similarity.shape)
                mix_similarity = self.conv_4d_blocks[str(idx)](ori_similarity)
                if self.add_5dconv:
                    mix_similarity = rearrange(mix_similarity, '(n b) t head h1 w1 h2 w2 -> b (head t) (h1 w1) (n h2 w2)', n=nshot, head=self.head)
                else:
                    mix_similarity = rearrange(mix_similarity, '(n head b) t h1 w1 h2 w2 -> b (head t) (h1 w1) (n h2 w2)', n=nshot, head=self.head)
                    
                num_channel = mix_similarity.shape[1]
                support_mask0 = support_masks[idx]
                support_mask0 = support_mask0.repeat(num_channel, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)

                mix_similarity = F.softmax(mix_similarity, dim=-1)

                coarse_mask = torch.matmul(mix_similarity, support_mask0)
                coarse_mask = rearrange(coarse_mask, 'b (head t) (h w) n -> (b n) head t h w', h=ha, w=wa, head=self.head)
                coarse_mask = torch.mean(coarse_mask, 1)

                coarse_masks[idx] = coarse_mask
        

        coarse_masks0 = {}
        for key, mask_list in coarse_masks.items():
            if self.add_4dconv or self.add_pool4d:
                value = mask_list.clone()
            else:
                bsz, ha, wa = mask_list[0].size()
                value = torch.stack(mask_list, dim=1).contiguous()

            coarse_masks0[key] = self.conv_layer[str(key)](value)
        


        mix_masks = {}
        if 3 in self.model_mask:
            coarse_masks3 = coarse_masks0[3]
            bsz, ch, ha, wa = coarse_masks3.size()
            coarse_masks3 = F.interpolate(coarse_masks3, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[3] = coarse_masks3
            if 2 in self.model_mask:
                if self.combine_mode == 'add':
                    mix = coarse_masks3 + coarse_masks0[2]
                elif self.combine_mode == 'concat':
                    mix = torch.cat([coarse_masks3, coarse_masks0[2]], dim=1)
            else:
                mix = coarse_masks3
            mix = self.conv4(mix)

        if 2 in self.model_mask or 3 in self.model_mask:
            coarse_masks2 = mix if 3 in self.model_mask else coarse_masks0[2]
            bsz, ch, ha, wa = coarse_masks2.size()
            coarse_masks2 = F.interpolate(coarse_masks2, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[2] = coarse_masks2
            if 1 in self.model_mask:
                if self.combine_mode == 'add':
                    mix = coarse_masks2 + coarse_masks0[1]
                elif self.combine_mode == 'concat':
                    mix = torch.cat([coarse_masks2, coarse_masks0[1]], dim=1)
            else:
                mix = coarse_masks2
            mix = self.conv5(mix)
        elif 1 in self.model_mask:
            mix = coarse_masks0[1]
            mix = self.conv5(mix)
        
        if 1 in self.model_mask:
            bsz, ch, ha, wa = mix.size()
            mix0 = F.interpolate(mix, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[1] = mix0

        # skip connect 1/8 and 1/4 features (concatenation)
        query_feat = low_query_feats[1] if self.new_skip else query_feats[self.stack_ids[1] - 1]
        mix = torch.cat((mix, query_feat), 1)
        upsample_size = (mix.size(-1) * upsample_times,) * 2

        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)

        query_feat = query_feats[self.stack_ids[0] - 1]
        
        
        mix = torch.cat((mix, query_feat), 1)

       
        out = self.mixer1(mix)
        upsample_size = (out.size(-1)*2*upsample_times,)*2 if self.mix_out=='mixer1' else (out.size(-1)*upsample_times,)*2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        if self.mix_out=='mixer2' or self.mix_out=='mixer3':
            out = self.mixer2(out)
            upsample_size = (out.size(-1) * upsample_times,) * 2
            out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        if self.mix_out=='mixer3':
            out = self.mixer3(out)
        
        logit_mask = out.clone()

        return logit_mask





if __name__ == '__main__':
    query_image = torch.randn(2, 3, 384, 384).to('cuda:5')
    support_image = torch.randn(2, 3, 384, 384).to('cuda:5')
    support_mask = torch.ones(2, 384, 384).to('cuda:5')

    src = './model/dcama/resnet50_a1h-35c100f8.pth'

    model = DAM(backbone='resnet50', pretrained_path=src, use_original_imgsize=False, original=True, 
                add_4dconv=True, skip_mode='mix', pooling_mix='concat', mixing_mode='concat', 
                mix_out='mixer3', combine_mode='add', model_mask=[2,3]).to('cuda:5')

    outputs = model(query_image, support_mask, support_image, support_mask)

    # print(outputs.shape)
    print(outputs[0].shape)
    print(outputs[1])
