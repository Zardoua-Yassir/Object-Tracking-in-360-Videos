import torch
from config import cfg
import numpy as np
import matplotlib.pyplot as plt


class LocLoss(torch.nn.Module):
    def __init__(self, x_grid, y_grid):
        super(LocLoss, self).__init__()
        # xy grids will be used to compute the localization targets. Computed only once
        self.x_grid, self.y_grid = x_grid, y_grid

    def forward(self, loc_prd, loc_tar, ifunc):
        self.loc_prd = loc_prd
        self.loc_tar = loc_tar
        self.ifunc = ifunc

        self.compute_iou_loss()
        self.infer_batch_loss()
        return self.loc_loss

    def compute_iou_loss(self):
        """
        Requires: self.loc_prd, self.loc_tar, self.ifunc
        Implements the IoU loss layer described in <https://dl.acm.org/doi/abs/10.1145/2964284.2967274>
        :param loc_prd: regression prediction tensor of size (batch, 4, out_height, out_width), where out_height
        and out_width correspond to the height and with of the model's output, respectively.
        :param loc_tar: regression target tensor corresponding to reg_pred.
        :param ifunc: indicator function (equation 5 in SiamCAR paper)
        Note: the second dimension in both reg_pred and loc_tar is 4, which indexes L,T,R,B coordinates.
        :return:
        """
        if isinstance(self.loc_prd, np.ndarray):
            self.loc_prd = torch.from_numpy(self.loc_prd)
        if isinstance(self.loc_tar, np.ndarray):
            self.loc_tar = torch.from_numpy(self.loc_tar)

        self.l_tar, self.t_tar, self.r_tar, self.b_tar = \
            self.loc_tar[:, 0, :, :], self.loc_tar[:, 1, :, :], self.loc_tar[:, 2, :, :], self.loc_tar[:, 3, :, :]
        self.l_prd, self.t_prd, self.r_prd, self.b_prd = \
            self.loc_prd[:, 0, :, :], self.loc_prd[:, 1, :, :], self.loc_prd[:, 2, :, :], self.loc_prd[:, 3, :, :]

        self.tar_ltrb_box_area = (self.t_tar + self.b_tar) * (self.l_tar + self.r_tar)
        self.prd_ltrb_box_area = (self.t_prd + self.b_prd) * (self.l_prd + self.r_prd)

        self.intersec_h = torch.min(self.t_prd, self.t_tar) + torch.min(self.b_prd, self.b_tar)
        self.intersec_w = torch.min(self.l_prd, self.l_tar) + torch.min(self.r_prd, self.r_tar)

        self.intersec_area = self.intersec_h * self.intersec_w
        self.union_area = self.tar_ltrb_box_area + self.prd_ltrb_box_area - self.intersec_area
        self.iou = self.intersec_area / self.union_area

        self.zero_loss = torch.zeros_like(self.ifunc)
        self.iou_loss = torch.where(self.ifunc, -torch.log(self.iou), self.zero_loss)  # the iou loss is equal to -
        # torch.log(iou) where iou_loss = torch.where(ifunc, iou, zero_loss)  # the iou loss is equal to -
        # torch.log(iou) where ifunc is True, and 0 (i.e., zero_loss) where ifunc is False
        return self.iou_loss

    def infer_batch_loss(self):
        self.ifunc_sum = self.ifunc.sum(dim=(1, 2))
        self.loc_loss = (self.iou_loss.sum(dim=(1, 2)) / self.ifunc_sum).mean()
        return self.loc_loss


class CenLossWithLogits(torch.nn.Module):
    def __init__(self):
        super(CenLossWithLogits, self).__init__()
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, cen_pred, cen_tar, ifunc):
        """

        ..Note: I validated this loss function by comparing its output to a customized implementation. I'm leveraging
        however the built-in PyTorch BCWWithLogitsLoss to take advantage of numerical stability (log max exp trick).
        :param cen_pred:
        :param cen_tar:
        :param ifunc:
        :return:
        """
        self.cen_pred = cen_pred.squeeze(1)  # current shape: (batch, 1, out_size, out_size), squeeze to remove
        # redundant dimension.
        self.cen_tar = cen_tar  # current shape: (batch, out_size, out_size)
        self.ifunc = ifunc  # current shape: (batch, out_size, out_size)

        # reshape to size [batch, out_size * out_size]
        self.cen_pred = self.cen_pred.view(cfg.TRAIN.BATCH_SIZE, -1)  # for each batch, flatten the prediction logits
        self.cen_tar = self.cen_tar.view(cfg.TRAIN.BATCH_SIZE, -1)  # for each batch, flatten the targets
        self.ifunc = self.ifunc.view(cfg.TRAIN.BATCH_SIZE, -1)  # for each batch, flatten the indicator function

        # for each batch, compute the loss for each (i, j) of the predicted centerness logits ->
        self.cen_loss = self.bce_logits_loss(self.cen_pred, self.cen_tar)  # order of arguments is important, output
        # shape is [batch, out_size * out_size].

        self.cen_loss = self.cen_loss * self.ifunc  # Set all losses not corresponding to locations (i, j) where
        # indicator function is True to Zero: PyTorch considers False as 0 when multiplying
        # compute the average losses of the batches
        self.cen_loss = (self.cen_loss.sum(1) / self.ifunc.sum(1))  # For each batch, compute the average loss by
        # dividing on the number of (i, j) points where indicator function is 1.
        self.cen_loss = self.cen_loss.mean()  # Compute the mean loss over all batches.
        return self.cen_loss


class ClsLossWithLogits(torch.nn.Module):
    def __init__(self):
        """
        self.ce_logits_loss Can accept input shape (minibatch,C,d1, d2, .., dk) with k ≥ 1 for the K-dimensional case.
        The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.
        In our case, k = 2 and input shape is (minibatch, 2, out_size, out_size), where out_size = cfg.TRAIN.OUTPUT_SIZE
        """
        super(ClsLossWithLogits, self).__init__()
        self.ce_logits_loss = torch.nn.CrossEntropyLoss()  # uses logits

    def forward(self, cls_pred, cls_tar):
        self.cls_pred = cls_pred
        self.cls_tar = cls_tar
        self.cls_loss = self.ce_logits_loss(self.cls_pred, self.cls_tar)
        return self.cls_loss


# Overall siamcarv1_loss function
class SiamCarLoss(torch.nn.Module):
    def __init__(self):
        super(SiamCarLoss, self).__init__()
        # compute xy grids

        self.x_grid, self.y_grid = self.compute_xy_grid(cfg.TRAIN.OUTPUT_SIZE,
                                                        cfg.TRAIN.SEARCH_SIZE,
                                                        cfg.TRAIN.BATCH_SIZE)
        self.cls_ones = torch.ones(cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)

        self.loc_loss_fn = LocLoss(self.x_grid, self.y_grid)
        self.cen_loss_fn = CenLossWithLogits()
        self.cls_loss_fn = ClsLossWithLogits()

    def forward(self, pred_dict, gt_bbox):
        """
        :param pred_dict: a dictionary of the model's predictions. Keys are: 'loc', 'cen', and 'cls', representing
        the localization, centerness, and classification predictions, respectively.
        :param gt_bbox: bounding box annotations corresponding to predictions, given as a mini-batch of GT bounding box
        corner coordinates within the srch_window image. Using this parameter only, targets corresponding to all three
        predictions ('loc', 'cen', and 'cls') can be computed.
        :return: the loss value
        """
        self.gt_bbox = gt_bbox
        self.compute_loc_targets()  # compute compatible labels (with pred_dict)
        self.compute_indicator_func()
        self.loc_loss = self.loc_loss_fn(pred_dict["loc"], self.loc_tar, self.ifunc)
        self.compute_cen_targets()
        self.cen_loss = self.cen_loss_fn(pred_dict["cen"], self.cen_tar, self.ifunc)

        self.cls_tar = self.compute_cls_tar(self.ifunc)
        self.cls_loss = self.cls_loss_fn(pred_dict["cls"], self.cls_tar)

        print("Loc loss = ", self.loc_loss.item())
        print("Cen loss = ", self.cen_loss.item())
        print("Cls loss = ", self.cls_loss.item())

        self.weighted_loss = (3 * self.loc_loss) + self.cen_loss + self.cls_loss

        self.loss_dic = {"loc": self.loc_loss,
                         "cen": self.cen_loss,
                         "cls": self.cls_loss,
                         "weighted": self.weighted_loss}
        return self.loss_dic

    def compute_xy_grid(self, out_size, search_size, batch_size):
        """
        Computes two grids self.x_grid, self.y_grid, of size (out_height, out_width) in numpy using meshgrid.

        :param out_size: width and height of the model's output maps. Must be speficied by a single integer or a tuple
        of integers (out_height, out_width) in case the width is not equal to the height
        :param search_size: width and height of the input srch_window region (denoted in the original paper as X). Must be
        speficied by a single integer or a tuple of integers (search_height, search_width) in case the width is not
        equal to the height
        :return:
        x_grid: A 2D numpy array containing all x coordinates of the srch_window region, starting from 0 to search_width - 1,
        that correspond to i coordinates of the output map.
        y_grid: A 2D numpy array containing all y coordinates of the srch_window region, starting from 0 to search_height - 1,
        that correspond to j coordinates of the output map.
        """
        out_height, out_width = out_size if isinstance(out_size, tuple) else (out_size, out_size)
        search_height, search_width = search_size if isinstance(search_size, tuple) else (search_size, search_size)

        x_range = np.linspace(0, search_width - 1, out_width)  # a vector of out_width elements, going from 0 to
        # search_width - 1 with uniform steps.
        y_range = np.linspace(0, search_height - 1, out_height)
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)

        # duplicate the grids on the mini-batch dimension (for computational compatibility)
        self.x_grid = np.stack([self.x_grid] * batch_size, axis=0)
        self.y_grid = np.stack([self.y_grid] * batch_size, axis=0)

        return self.x_grid, self.y_grid

    def compute_loc_targets(self, gt_bbox=None):
        """
        Computes and returns a mini-batch of the regression labels localizing the object within the srch_window region
        :param gt_bbox: a mini-batch of the labeled object's bounding box corner coordinates given within the srch_window
        image. Corners are: top-left (x0, y0) and bottom-right (x1, y1). gt_bbox.shape = (mini_batch_size, 4), where 4
        corresponds to the four scalars x0, y0, x1, y1
        :return: regression labels loc_tar as a mini-batch
        """
        # size = torch.Size([batch_size])
        if gt_bbox is not None:
            self.gt_bbox = gt_bbox
        self.x0, self.y0, self.x1, self.y1 = \
            self.gt_bbox[:, 0], self.gt_bbox[:, 1], self.gt_bbox[:, 2], self.gt_bbox[:, 3]

        self.x0 = self.x0.resize(cfg.TRAIN.BATCH_SIZE, 1, 1).numpy()  # To allow numpy broadcasting during add and
        # subtraction

        # resize, then convert to numpy
        self.y0 = self.y0.resize(cfg.TRAIN.BATCH_SIZE, 1, 1).numpy()
        self.x1 = self.x1.resize(cfg.TRAIN.BATCH_SIZE, 1, 1).numpy()
        self.y1 = self.y1.resize(cfg.TRAIN.BATCH_SIZE, 1, 1).numpy()

        self.left_target = self.x_grid - self.x0  # numpy broadcasting
        self.top_target = self.y_grid - self.y0
        self.right_target = self.x1 - self.x_grid
        self.bottom_target = self.y1 - self.y_grid

        self.loc_tar = np.stack((self.left_target, self.top_target, self.right_target, self.bottom_target), axis=1)
        # loc_tar shape = (4
        # , cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE). Stacking is done on axis=1 because axis=0 correspond to batch
        self.loc_tar = torch.from_numpy(self.loc_tar)  # required for operand type compatibility
        return self.loc_tar

    def compute_indicator_func(self, loc_tar=None):
        """
        Compute the indicator function, which is a matrix indicating the points (i, j) whose correspondent (x, y) fall
        within the target fine_bbox.
        :param loc_tar: a torch tensor of regression labels of size (batch, 4, out_height, out_width)
        :return: an indicator function as a Boolean Tensor of size (batch, 1, out_height, out_width)
        """
        if loc_tar is not None:
            self.loc_tar = loc_tar
        self.loc_tar = torch.clip(input=self.loc_tar, min=0)  # By clipping negative values to zero while
        # keeping positive values unchanged, we allow them to behave as Boolean Falses and Trues, respectively.
        self.ifunc = torch.all(self.loc_tar, dim=1, keepdim=False)  # perform the 'logical and' between the l,t,
        # r,b maps  of each mini-batch, which are located at dim=1. dim=1, which has 4 maps, will be reduced to 1 map.
        # keepdim if True keeps the dimension of the reduced dimension (dim=1 is reduced from 4 to 1) in the output
        # tensor.

        # Making sure each batch corresponds to an indicator function with at least one True point (i,j), basically
        # means there is an object in the srch_window region. Otherwise, unexpected behavior will be triggered, specifically
        # at the loss functions

        self.valid_ifunc = False
        self.valid_ifunc = torch.any(torch.any(self.ifunc, dim=(1, 2)))
        if not self.valid_ifunc:
            raise ValueError("Input error: At least one True value required for indicator functions of each batch.")
        return self.ifunc

    def compute_cen_targets(self, loc_tar=None, ifunc=None):
        if loc_tar is not None:
            self.loc_tar = loc_tar
        if ifunc is not None:
            self.ifunc = ifunc

        if isinstance(self.loc_tar, np.ndarray):
            self.loc_tar = torch.from_numpy(self.loc_tar)

        self.l_tar, self.t_tar, self.r_tar, self.b_tar =\
            self.loc_tar[:, 0, :, :], self.loc_tar[:, 1, :, :], self.loc_tar[:, 2, :, :], self.loc_tar[:, 3, :, :]

        self.cen_tar = self.ifunc * torch.sqrt((torch.min(self.l_tar, self.r_tar)/torch.max(self.l_tar, self.r_tar)) *
                                               (torch.min(self.t_tar, self.b_tar)/torch.max(self.t_tar, self.b_tar)))
        return self.cen_tar

    def display_cen_map(self, cen_tensor, batch_index=0):
        # Convert tensor to numpy array
        cen_tensor_np = cen_tensor[batch_index].detach().numpy()
        # Display the numpy array using Matplotlib
        plt.imshow(cen_tensor_np, cmap='gray')
        plt.axis('off')
        plt.show()

    def compute_cls_tar(self, ifunc=None):
        if ifunc is not None:
            self.ifunc = ifunc
        self.cls_fg_tar = self.cls_ones * self.ifunc
        self.cls_bg_tar = self.cls_ones * torch.logical_not(self.ifunc)
        self.cls_tar = torch.stack((self.cls_fg_tar, self.cls_bg_tar), dim=1)  # self.cls_tar[0] outputs self.cls_fg_tar
        # , self.cls_tar[0] outputs self.cls_bg_tar
        return self.cls_tar