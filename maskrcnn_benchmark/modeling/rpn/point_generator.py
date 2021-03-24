import torch
import torch.nn as nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

class PointGenerator(nn.Module):
    def __init__(self, fpn_stride, octave=2, scales_per_octave=3, aspect_ratios=(0.5, 1.0, 2.0)):
        super(PointGenerator, self).__init__()
        self.fpn_stride = fpn_stride
        # (8, 16, 32, 64, 128)
        self.anchor_per_position = scales_per_octave * len(aspect_ratios)

    def forward(self, image_list, feature_maps):
        assert len(feature_maps) == len(self.fpn_stride)

        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        points_over_all_feature_maps = []
        for grid_size, stride in zip(grid_sizes, self.fpn_stride):
            points_over_all_feature_maps.append(self.grid_points(grid_size, stride, self.anchor_per_position))

        points = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            points_in_image = []
            for points_per_feature_map in points_over_all_feature_maps:
                points_in_image.append(points_per_feature_map.clone())
            points.append(points_in_image)

        return points


    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, n_per_pos=9, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y, row_major=False)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        dim_col = all_points.size(1)
        all_points = all_points.repeat(1, n_per_pos).view(-1, dim_col)
        return all_points

        """
            row_major = True
            for feature map of size (3, 4), stride = 16
            tensor([[ 0.,  0., 16.],
                    [16.,  0., 16.],
                    [32.,  0., 16.],
                    [48.,  0., 16.],
                    [ 0., 16., 16.],
                    [16., 16., 16.],
                    [32., 16., 16.],
                    [48., 16., 16.],
                    [ 0., 32., 16.],
                    [16., 32., 16.],
                    [32., 32., 16.],
                    [48., 32., 16.]])
            
            (y, x) (h, w)
            row_major = False
            tensor([[ 0.,  0., 16.],
                    [ 0., 16., 16.],
                    [ 0., 32., 16.],
                    [ 0., 48., 16.],
                    [16.,  0., 16.],
                    [16., 16., 16.],
                    [16., 32., 16.],
                    [16., 48., 16.],
                    [32.,  0., 16.],
                    [32., 16., 16.],
                    [32., 32., 16.],
                    [32., 48., 16.]])

        """

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

        """
            for feature map of size (4, 5), valid_size (4, 4)
            valid = 
                tensor([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    dtype=torch.uint8)

        """

def make_point_generator_retinanet(config):
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS # (0.5, 1.0, 2.0)
    octave = config.MODEL.RETINANET.OCTAVE # 2
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE # 3

    point_generator = PointGenerator(config.MODEL.RETINANET.ANCHOR_STRIDES, 
        octave, scales_per_octave, aspect_ratios)
    return point_generator