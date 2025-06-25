"""
SpineTR model and criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""

import torch, math
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from models.backbone import build_backbone
from models.transformer import build_transformer
from torch import nn

class SpineTR(nn.Module):
    """ This is the module that performs vertebrates detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, feature_map_dim=[12,12,12], aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: binary classification for vertebrate existance #2
            num_queries: number of vertebrates, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            feature_map_dim: feature map dimension, i.e.[12,12,12] for input patch size [192, 192, 192]
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.proj = nn.Linear(feature_map_dim[0] * feature_map_dim[1] * feature_map_dim[2], 25)  # 25

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        query_embed = self.query_embed.weight

        assert mask is not None
        src_proj = self.input_proj(src)
        hs, memory, encoder_attn, decoder_attn, decoder_self_attn = self.transformer(src_proj, mask, query_embed, pos[-1])

        bs = features[-1].tensors.shape[0]

        cnn_plus_tencoder = src_proj.view(bs, src_proj.shape[1], -1)
        cnn_outputs_coord = self.proj(cnn_plus_tencoder).permute(0,2,1)
        hs_combined_feature = hs + cnn_outputs_coord

        outputs_class = self.class_embed(hs_combined_feature).softmax(-1)

        outputs_coord = self.bbox_embed(hs_combined_feature).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, encoder_attn, decoder_attn, decoder_self_attn

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class Criterion(nn.Module):
    def __init__(self, num_classes, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses

    # 分类损失
    def loss_labels(self, outputs, targets, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # 分类预测logits，预测概率
        # 全0张量 [batch_size, num_queries]
        batch_size, num_queries = src_logits.shape[:2]
        # print(batch_size, num_queries)

        target_classes = torch.zeros((batch_size, num_queries),
                                     dtype=torch.int64,
                                     device=src_logits.device)

        for i in range(batch_size):
            # 提取当前样本的标签（非零值）
            labels = targets['labels'][i]  # [num_queries]
            valid_labels = labels[labels != 0]  # 过滤无效标签（值为0的位置）

            # 将有效标签对应的位置设为1
            # 注意：valid_labels 中的值是目标位置的索引，例如16、17等
            target_classes[i, valid_labels] = 1

        # print(target_classes)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        if log:
            # 计算分类错误率
            pred_classes = torch.argmax(src_logits, dim=-1)  # [batch_size, num_queries]
            correct = (pred_classes == target_classes).sum().float()
            total = target_classes.numel()  # 总查询数 = batch_size * num_queries
            losses['class_error'] = 1 - correct / total

        return losses

    def loss_boxes(self, outputs, targets):
        assert 'pred_boxes' in outputs
        loss_bbox = torch.tensor(0, dtype=torch.float).cuda()
        loss_giou = torch.tensor(0, dtype=torch.float).cuda()
        loss_edges_sphere = torch.tensor(0, dtype=torch.float).cuda()
        bs = 1

        # 批次大小
        valid_num = torch.sum(targets['exist'])
        # 有效椎体总数（避免除以零）
        num_boxes = 0.0001 + valid_num

        # 从 outputs 中提取预测的球心坐标 [bs, num_queries, 3]
        sphere_locs = outputs['pred_boxes'][:, :, :3] # (x,y,z,r)取前3维，即(x,y,z)
        # print(f'sphere_locs: {sphere_locs}')
        # print(sphere_locs.shape)
        # 生成边缘位移向量
        zero_tensor = torch.zeros((bs, 1, 3), device=sphere_locs.device)
        sphere_locs1 = torch.cat((sphere_locs, zero_tensor), dim=1)  # 末尾补零
        sphere_locs2 = torch.cat((zero_tensor, sphere_locs), dim=1)  # 开头补零

        etop = (sphere_locs1 - sphere_locs2)[:, :25, :]  # 前向位移
        ebot = (sphere_locs2 - sphere_locs1)[:, 1:, :]  # 后向位移

        edges_from_sphere = torch.cat((etop, ebot), dim=2)  # 合并位移 [bs, num_queries, 6]

        labels = targets['labels']
        valid_labels = labels[labels != 0]
        # print(f'valid_labels: {valid_labels}')

        # print(targets['sphere_coords'][0, 1])
        for i in valid_labels:
            # loss_reg
            # print(f'i: {i}')
            center_true = targets['sphere_coords'][0, i]
            # print(f'center_true: {center_true}')
            center_pred = outputs['pred_boxes'][0, i]
            # print(f'center_pred: {center_pred}')
            loss_bbox += torch.abs(center_pred - center_true).sum(dim=0)
            # print(loss_bbox)

        # loss_edges
        edges_mask = targets['has_edges']
        src_edges = edges_from_sphere * edges_mask
        target_edges = targets['edges'] * edges_mask
        loss_edges_sphere = F.l1_loss(src_edges, target_edges, reduction='none').sum()

        # loss_giou
        pred_spheres = outputs['pred_boxes']
        targets_spheres = targets['sphere_coords']
        # print(f'pred_spheres: {pred_spheres}')
        # print(f'targets_spheres: {targets_spheres}')

        def build_cube(spheres):
            centers = spheres[..., :3]
            radii = spheres[..., 3:]
            cubes = torch.cat([
                centers - radii, # [x - r, y - r, z - r]
                centers + radii  # [x + r, y + r, z + r]
            ], dim=-1)
            return cubes
        
        pred_cubes = build_cube(pred_spheres)
        pred_cubes_flat = pred_cubes.view(-1, 6)
        target_cubes = build_cube(targets_spheres)
        target_cubes_flat = target_cubes.view(-1, 6)
        # print(f'pred_cubes.shape: {pred_cubes.shape}')
        # print(f'target_cubes.shape: {target_cubes.shape}')
        # print(f'target_cubes: {target_cubes}')

        giou = box_ops.generalized_box_iou(pred_cubes_flat, target_cubes_flat)
        loss_giou = 1 - giou.mean()

        # 归一化损失
        losses = {
            'loss_bbox': loss_bbox / num_boxes,
            'loss_giou': loss_giou / num_boxes,
            'loss_edges_sphere': loss_edges_sphere / num_boxes
        }
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_logits, out_bbox, out_masks = outputs['pred_logits'], outputs['pred_boxes'], None

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 3

        prob = out_logits
        scores, labels = torch.max(prob,dim=-1)

        img_w, img_h, img_d = target_sizes.unbind(1)
        img_r = torch.tensor([math.sqrt(img_w**2+img_h**2+img_d**2)]).cuda()
        scale_fct = torch.stack([img_w.float(), img_h.float(), img_d.float(), img_r], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = SpineTR(
        backbone,
        transformer,
        num_classes=2,  # note: binary classification for vertebrate existance
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_edges_sphere'] = args.bbox_loss_coef # additional edge restriction on predicted sphere coordinates

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    criterion = Criterion(2, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors