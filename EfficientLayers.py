import torch
import torch.nn as nn

from torch import autograd
import torch.nn.functional as F

import time
import einops
from einops import einsum


class NormalLinear(autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        """
        Forward pass of the linear layer
        x: input tensor (b, i)
        W: weight tensor (o, i)
        b: bias tensor (o)

        Returns:
        y: output tensor (b, o)
        """
        ctx.save_for_backward(x, W, b)

        return F.linear(x, W, b)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of the linear layer
        dy: gradient of the output (b, o)

        Returns:
        dx: gradient of the input (b, i)
        dw: gradient of the weights (o, i)
        db: gradient of the bias (o)
        """
        x, W, b = ctx.saved_tensors

        dx, dw, db = None, None, None

        # Calculate the gradient of the input
        if ctx.needs_input_grad[0]:
            dx = dy@W

        # Calculate the gradient of the weights
        if ctx.needs_input_grad[1]:
            dw = dy.T@x
        
        # Calculate the gradient of the bias
        if ctx.needs_input_grad[2]:
            db = torch.sum(dy, dim=0)

        return dx, dw, db

class EfficientLinear(autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b, percentage, unified, device):
        """
        Forward pass of the linear layer
        x: input tensor (b, i)
        W: weight tensor (o, i)
        b: bias tensor (o)
        percentage: percentage of the weights to keep
        unified: whether to apply the topk operation on the entire batch or each sample
        device: device to store the tensors

        Returns:
        y: output tensor (b, o)
        """
        ctx.save_for_backward(x, W, b)

        ctx.percentage = percentage
        ctx.unified = unified
        ctx.device = device

        return F.linear(x, W, b)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of the linear layer
        dy: gradient of the output (b, o)

        Returns:
        dx: gradient of the input (b, i)
        dw: gradient of the weights (o, i)
        db: gradient of the bias (o)    
        """
        x, W, b = ctx.saved_tensors
        bt, in_c, out_c = x.size(0), x.size(1), W.size(0)

        dx, dw, db = None, None, None

        if not ctx.unified:
            _, idx = torch.topk(torch.abs(dy), int(ctx.percentage * out_c), dim=1, sorted=False)  # (b, selected_o)
            dy = torch.gather(dy, 1, idx)

            # Calculate the gradient of the input
            if ctx.needs_input_grad[0]:
                dx = einsum(dy, W[idx, :], 'b o, b o i -> b i')
            
            # Calculate the gradient of the weights
            if ctx.needs_input_grad[1]:
                dw = torch.zeros((bt, out_c), device=ctx.device).scatter_(1, idx, dy).T@x
            
            # Calculate the gradient of the bias
            if ctx.needs_input_grad[2]:
                db = torch.zeros((bt, out_c), device=ctx.device).scatter_(1, idx, dy)
                db = torch.sum(db, dim=0)
        else:
            _, idx = torch.topk(torch.sum(torch.abs(dy), axis=0), int(ctx.percentage * out_c), sorted=False)
            dy = dy[:, idx]

            # Calculate the gradient of the input
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, W[idx, :])
            
            # Calculate the gradient of the weights, dy: (b, selected_o), x: (b, i)
            if ctx.needs_input_grad[1]:
                dw = torch.mm(dy.T, x)
                dw = torch.zeros((out_c, in_c), device=ctx.device).index_copy_(0, idx, dw)
            
            # Calculate the gradient of the bias
            if ctx.needs_input_grad[2]:
                db = torch.sum(dy, dim=0)
                db = torch.zeros((out_c), device=ctx.device).index_copy_(0, idx, db)

        return dx, dw, db, None, None, None


class CustomLinear(torch.nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, mode=None, percentage=0.5, unified=True, device='cuda'):
        super(CustomLinear, self).__init__()
        assert mode is not None, 'Please provide mode'

        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.mode = mode
        self.percentage = percentage
        self.unified = unified
        self.device = device

        self.W = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        nn.init.kaiming_uniform_(self.W, a=5**0.5)  # Initialize weights

        self.b = None
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
            nn.init.zeros_(self.b)  # Initialize bias

    def forward(self, x):
        if self.mode == 'normal':
            return NormalLinear.apply(x, self.W, self.b)
        elif self.mode == 'efficient':
            return EfficientLinear.apply(x, self.W, self.b, self.percentage, self.unified, self.device)


def img2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    if pad > 0:
        img = F.pad(input_data, (pad, pad, pad, pad), mode='constant', value=0)
    else:
        img = input_data

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w), device=input_data.device)

    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]

    col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2img(col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
    N, C, H, W = input_shape
    # Reshape and permute the columns back into the image format
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = tmp1.permute(0, 3, 4, 5, 1, 2)
    
    # Prepare a tensor for the output image with appropriate padding
    img = torch.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), device=col.device)
    
    # Accumulate contributions from the columns back to the corresponding image locations
    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
    
    # Crop the padded area to get the original image dimensions back
    return img[:, :, pad:H + pad, pad:W + pad]


class NormalConv2d(autograd.Function):
    """Note: PyTorch didn't divide the gradients by the batch size in the backward pass of the convolutional layer, then we follow the same behavior."""

    @staticmethod
    def calculate_output_size_(input_h, input_w, filter_h, filter_w, padding, stride=1):
        output_h = (input_h - filter_h + 2 * padding) // stride + 1    
        output_w = (input_w - filter_w + 2 * padding) // stride + 1
        return (output_h, output_w)

    @staticmethod
    def forward(ctx, x, W, b, stride, padding, dilation, groups, use_bias, built_in):
        """
        Forward pass of the convolutional layer
        x: input tensor (b, i, h, w)
        W: weight tensor (o, i, kh, kw)
        b: bias tensor (o)
        stride: stride of the convolution
        padding: padding of the convolution
        dilation: dilation of the convolution
        groups: number of groups for grouped convolution

        Returns:
        y: output tensor (b, o, h', w')
        """

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.use_bias = use_bias
        ctx.built_in = built_in

        if ctx.built_in:
            ctx.save_for_backward(x, W, b)
            return F.conv2d(x, W, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
        else:
            bt, in_c, in_h, in_w = x.size()
            out_c, _, kh, kw = W.size()
            out_h, out_w = NormalConv2d.calculate_output_size_(in_h, in_w, kh, kw, ctx.padding, ctx.stride)

            ctx.in_c = in_c
            ctx.in_h = in_h
            ctx.in_w = in_w
            ctx.kh = kh
            ctx.kw = kw
            ctx.out_h = out_h
            ctx.out_w = out_w

            col_x = img2col(x, kh, kw, ctx.stride, ctx.padding) # (b * out_h * out_w, in_c * kh * kw)
            col_w = W.reshape(out_c, -1).T                      # (in_c * kh * kw, out_c)
            if ctx.use_bias:
                col_b = b.reshape(-1, out_c)                    # (1, out_c)

            ctx.save_for_backward(col_x, col_w)

            out1 = torch.mm(col_x, col_w) + (col_b if ctx.use_bias else 0)
            out2 = out1.reshape(bt, out_h, out_w, -1)
            z = out2.permute(0, 3, 1, 2)

            return z

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of the convolutional layer
        dy: gradient of the output (b, o, h', w')

        Returns:
        dx: gradient of the input (b, i, h, w)
        dw: gradient of the weights (o, i, kh, kw)
        db: gradient of the bias (o)
        """
        dx, dw, db = None, None, None

        if not ctx.built_in:
            col_x, col_w = ctx.saved_tensors                                # col_x: (b * out_h * out_w, in_c * kh * kw), col_w: (in_c * kh * kw, out_c)

            bt, out_c, out_h, out_w = dy.size()
            col_delta_in = dy.permute(0, 2, 3, 1).reshape(-1, out_c)        # (b * out_h * out_w, out_c)

            if ctx.use_bias and ctx.needs_input_grad[2]:
                db = torch.sum(col_delta_in, dim=0, keepdim=False)          # (out_c)

            if ctx.needs_input_grad[1]:
                col_dW = torch.mm(col_x.T, col_delta_in)                        # (in_c * kh * kw, out_c)
                dw = col_dW.T.reshape(out_c, ctx.in_c, ctx.kh, ctx.kw)

            if ctx.needs_input_grad[0]:
                col_delta_out = torch.mm(col_delta_in, col_w.T)                 # (b * out_h * out_w, in_c * kh * kw)
                dx = col2img(col_delta_out, (bt, ctx.in_c, ctx.in_h, ctx.in_w), ctx.kh, ctx.kw, ctx.stride, ctx.padding, ctx.out_h, ctx.out_w)
        else:
            x, W, b = ctx.saved_tensors
            
            if ctx.needs_input_grad[0]:
                # Calculate the gradient of the input
                # dx = F.conv_transpose2d(dy, W, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                dx = torch.nn.grad.conv2d_input(x.shape, W, dy, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
            if ctx.needs_input_grad[1]:
                # Calculate the gradient of the weights
                dw = torch.nn.grad.conv2d_weight(x, W.shape, dy, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
            if ctx.use_bias and ctx.needs_input_grad[2]:
                # Calculate the gradient of the bias
                db = torch.sum(dy, dim=(0, 2, 3))

        return dx, dw, db, None, None, None, None, None, None


class EfficientConv2d(autograd.Function):

    @staticmethod
    def calculate_output_size_(input_h, input_w, filter_h, filter_w, padding, stride=1):
        output_h = (input_h - filter_h + 2 * padding) // stride + 1    
        output_w = (input_w - filter_w + 2 * padding) // stride + 1
        return (output_h, output_w)
    
    @staticmethod
    def select_topk_idx_(dy, percentage, out_c, random):

        if not random:
            bs_ave_dy = dy.abs().mean(dim=(0,2,3))
            _, idx = torch.topk(bs_ave_dy, max(int(percentage * out_c), 1), dim=0, sorted=False)    # idx: (selected_o)
        else:
            # select random output channels without replacement
            idx = torch.randperm(out_c, device=dy.device)[:max(int(percentage * out_c), 1)]       # idx: (selected_o)

        return idx
    
    @staticmethod
    def forward(ctx, x, W, b, stride, padding, dilation, groups, use_bias, percentage, unified, sparse, built_in, random, only_W, drop_behavior, device):
        """
        Forward pass of the convolutional layer
        x: input tensor (b, i, h, w)
        W: weight tensor (o, i, kh, kw)
        b: bias tensor (o)
        stride: stride of the convolution
        padding: padding of the convolution
        dilation: dilation of the convolution
        groups: number of groups for grouped convolution
        percentage: percentage of the gradient to keep
        unified: whether to apply the topk operation on the entire batch or each sample
        sparse: whether to use sparse gradients
        built_in: whether to use built-in functions
        random: whether to randomly select the topk output channels
        only_W: whether to only apply the topk gradient on the weights
        drop_behavior: drop behavior for sensitivity 1 analysis
        device: device to store the tensors

        Returns:
        y: output tensor (b, o, h', w')
        """

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.use_bias = use_bias

        ctx.percentage = percentage
        ctx.unified = unified
        ctx.sparse = sparse
        ctx.built_in = built_in
        ctx.random = random
        ctx.only_W = only_W
        ctx.drop_behavior = drop_behavior
        ctx.device = device

        bt, in_c, in_h, in_w = x.size()
        out_c, _, kh, kw = W.size()
        ctx.in_c = in_c
        ctx.in_h = in_h
        ctx.in_w = in_w
        ctx.kh = kh
        ctx.kw = kw

        if ctx.built_in:
            ctx.save_for_backward(x, W, b)
            return F.conv2d(x, W, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
        else:
            out_h, out_w = EfficientConv2d.calculate_output_size_(in_h, in_w, kh, kw, ctx.padding, ctx.stride)

            col_x = img2col(x, kh, kw, ctx.stride, ctx.padding)
            col_w = W.reshape(out_c, -1).T
            if ctx.use_bias:
                col_b = b.reshape(-1, out_c)

            ctx.save_for_backward(col_x, col_w)

            out1 = torch.mm(col_x, col_w) + (col_b if ctx.use_bias else 0)
            out2 = out1.reshape(bt, out_h, out_w, -1)
            z = out2.permute(0, 3, 1, 2)

            return z

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of the convolutional layer
        dy: gradient of the output (b, o, h', w')

        Returns:
        dx: gradient of the input (b, i, h, w)
        dw: gradient of the weights (o, i, kh, kw)
        db: gradient of the bias (o)
        """
        bt, out_c, out_h, out_w = dy.size()
        
        dx, dw, db = None, None, None

        if ctx.built_in:
            x, W, b = ctx.saved_tensors
            if ctx.unified:
                if ctx.sparse:
                    assert ctx.only_W == False, 'Sparse gradient is not implemented for only_W'

                    idx = EfficientConv2d.select_topk_idx_(dy, ctx.percentage, out_c, ctx.random) # (selected_o,)
                    # dy[:, idx, :, :] = 0 # this is a bug, should not set the selected output channels to zero
                    mask = torch.zeros_like(dy)
                    mask[:, idx, :, :] = 1
                    dy = dy * mask

                    # # correct the expectation of the gradient if it's random mode, but seems not good
                    # if ctx.random:
                    #     dy = dy / ctx.percentage

                    if ctx.needs_input_grad[0]:
                        # Calculate the gradient of the input
                        # dx = F.conv_transpose2d(dy, W, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                        dx = torch.nn.grad.conv2d_input(x.shape, W, dy, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                    if ctx.needs_input_grad[1]:
                        # Calculate the gradient of the weights
                        dw = torch.nn.grad.conv2d_weight(x, W.shape, dy, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
                    if ctx.needs_input_grad[2]:
                        # Calculate the gradient of the bias
                        db = torch.sum(dy, dim=(0, 2, 3))
                else:
                    """Use built-in functions to calculate the gradients"""
                    idx = EfficientConv2d.select_topk_idx_(dy, ctx.percentage, out_c, ctx.random) # (selected_o,)
                    # dy = dy[:, idx, :, :] # (b, selected_o, h', w')
                    # if ctx.random:
                    #     dy = dy / ctx.percentage

                    if ctx.use_bias and ctx.needs_input_grad[2]:
                        db = torch.sum(dy[:, idx, :, :], dim=(0, 2, 3))
                        db = torch.zeros((out_c), device=ctx.device).index_copy_(0, idx, db)

                    if ctx.needs_input_grad[1]:
                        dw = torch.nn.grad.conv2d_weight(x, (idx.size(0), ctx.in_c, ctx.kh, ctx.kw), dy[:, idx, :, :], stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
                        dw = torch.zeros((out_c, ctx.in_c, ctx.kh, ctx.kw), device=ctx.device).index_copy_(0, idx, dw)

                    if ctx.needs_input_grad[0]:
                        if ctx.only_W:
                            # if ctx.random:
                            #     dy = dy * ctx.percentage # do not change the expectation of the gradient here
                            dx = torch.nn.grad.conv2d_input(x.shape, W, dy, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                        else:
                            dx = torch.nn.grad.conv2d_input(x.shape, W[idx, ...], dy[:, idx, :, :], stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
            else:
                """This is only for sensitivity analysis, not for real usage"""
                if ctx.sparse:
                    assert ctx.only_W == False, 'Not implemented for only_W'

                    assert ctx.drop_behavior in ['drop_h_w', 'drop_ch_h_w', 'drop_ch'], 'Should be either drop_h_w, drop_ch_h_w, or drop_ch.'

                    if ctx.drop_behavior == 'drop_h_w':
                        bs_ave_dy = dy.abs().mean(dim=(0,1)).reshape(out_h * out_w)
                        _, idx = torch.topk(bs_ave_dy, max(int(ctx.percentage * out_h * out_w), 1), dim=0, sorted=False)    # idx: (selected_(out_h * out_w))
                        dy = dy.reshape(bt, out_c, out_h * out_w)
                        # dy[:, :, idx] = 0 # this is a bug, should not set the selected to zero
                        mask = torch.zeros_like(dy)
                        mask[:, :, idx] = 1
                        dy = dy * mask
                        dy = dy.reshape(bt, out_c, out_h, out_w)

                    if ctx.drop_behavior == 'drop_ch_h_w':
                        bs_ave_dy = dy.abs().mean(dim=(0)).reshape(out_c * out_h * out_w)
                        _, idx = torch.topk(bs_ave_dy, max(int(ctx.percentage * out_c * out_h * out_w), 1), dim=0, sorted=False)    # idx: (selected_(out_c * out_h * out_w))
                        dy = dy.reshape(bt, out_c * out_h * out_w)
                        # dy[:, idx] = 0 # this is a bug, should not set the selected to zero
                        mask = torch.zeros_like(dy)
                        mask[:, idx] = 1
                        dy = dy * mask
                        dy = dy.reshape(bt, out_c, out_h, out_w)

                    if ctx.drop_behavior == 'drop_ch':
                        bs_ave_dy = dy.abs().mean(dim=(0,2,3))
                        _, idx = torch.topk(bs_ave_dy, max(int(ctx.percentage * out_c), 1), dim=0, sorted=False)    # idx: (selected_o)
                        # dy[:, idx, :, :] = 0 # this is a bug, should not set the selected output channels to zero
                        mask = torch.zeros_like(dy)
                        mask[:, idx, :, :] = 1
                        dy = dy * mask

                    if ctx.needs_input_grad[0]:
                        # Calculate the gradient of the input
                        # dx = F.conv_transpose2d(dy, W, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                        dx = torch.nn.grad.conv2d_input(x.shape, W, dy, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                    if ctx.needs_input_grad[1]:
                        # Calculate the gradient of the weights
                        dw = torch.nn.grad.conv2d_weight(x, W.shape, dy, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
                    if ctx.needs_input_grad[2]:
                        # Calculate the gradient of the bias
                        db = torch.sum(dy, dim=(0, 2, 3))
                else:
                    raise NotImplementedError('Not implemented yet')
        else:
            assert ctx.only_W == False, 'Non-built-in (img2col,col2img) mode only supports only_W=False'

            col_x, col_w = ctx.saved_tensors        # col_x: (b * out_h * out_w, in_c * kh * kw), col_w: (in_c * kh * kw, out_c)
            if ctx.unified:
                if ctx.sparse:
                    raise NotImplementedError('Not implemented yet')
                else:
                    """take mean on batch and select topk output channels, this is more memory efficient"""
                    idx = EfficientConv2d.select_topk_idx_(dy, ctx.percentage, out_c, ctx.random) # (selected_o,)
                    col_delta_in = dy.permute(0, 2, 3, 1).reshape(-1, out_c)[:, idx] # (b * out_h * out_w, selected_o)
                    # if ctx.random:
                    #     col_delta_in = col_delta_in / ctx.percentage

                    if ctx.use_bias and ctx.needs_input_grad[2]:
                        db = torch.sum(col_delta_in, dim=0, keepdim=False)
                        db = torch.zeros((out_c), device=ctx.device).index_copy_(0, idx, db)

                    if ctx.needs_input_grad[1]:
                        col_dW = torch.mm(col_x.T, col_delta_in)                        # (in_c * kh * kw, selected_o)
                        col_dW = torch.zeros((ctx.in_c * ctx.kh * ctx.kw, out_c), device=ctx.device).index_copy_(1, idx, col_dW)
                        dw = col_dW.T.reshape(out_c, ctx.in_c, ctx.kh, ctx.kw)

                    if ctx.needs_input_grad[0]:
                        col_delta_out = torch.mm(col_delta_in, col_w.T[idx, :])         # (b * out_h * out_w, in_c * kh * kw)
                        dx = col2img(col_delta_out, (bt, ctx.in_c, ctx.in_h, ctx.in_w), ctx.kh, ctx.kw, ctx.stride, ctx.padding, out_h, out_w)
            else:
                raise NotImplementedError('Not implemented yet')
                # # each sample and pixel select topk output channels, seems not efficient and not good as unified mode
                # col_delta_in = dy.permute(0, 2, 3, 1).reshape(-1, out_c) # (b * out_h * out_w, out_c)
                # _, idx = torch.topk(torch.sum(torch.abs(col_delta_in), dim=1).flatten(), max(int(ctx.percentage * col_delta_in.size(0)), 1), dim=0, sorted=False)
                # col_delta_in = col_delta_in[idx]                                # (selected_(b * out_h * out_w), out_c)

                # if ctx.use_bias and ctx.needs_input_grad[2]:
                #     db = torch.sum(col_delta_in, dim=0, keepdim=False)

                # if ctx.needs_input_grad[1]:
                #     col_dW = torch.mm(col_x.T[:, idx], col_delta_in)                # (in_c * kh * kw, out_c)
                #     dw = col_dW.T.reshape(out_c, ctx.in_c, ctx.kh, ctx.kw)

                # if ctx.needs_input_grad[0]:
                #     col_delta_out = torch.mm(col_delta_in, col_w.T)                 # (b * out_h * out_w, in_c * kh * kw)
                #     col_delta_out = torch.zeros((bt * out_h * out_w, ctx.in_c * ctx.kh * ctx.kw), device=ctx.device).index_copy_(0, idx, col_delta_out)
                #     dx = col2img(col_delta_out, (bt, ctx.in_c, ctx.in_h, ctx.in_w), ctx.kh, ctx.kw, ctx.stride, ctx.padding, out_h, out_w)
                    
                # """take mean on batch and select different topk for each output channel, this is extremely GPU memory consuming because of broadcasting"""
                # bs_ave_dy = dy.abs().mean(dim=0).reshape(out_c, -1) # (out_c, out_h * out_w)
                # _, idx = torch.topk(bs_ave_dy, max(int(ctx.percentage * out_h * out_w), 1), dim=1, sorted=False)    # idx: (out_c, selected_(out_h * out_w))
                # col_delta_in = dy.view(bt, out_c, out_h * out_w).gather(2, idx.unsqueeze(0).expand(bt, out_c, idx.size(1))).permute(0, 2, 1).reshape(-1, out_c) # (b * selected_(out_h * out_w), out_c)

                # if ctx.use_bias:
                #     db = torch.sum(col_delta_in, dim=0, keepdim=False)
                # else:
                #     db = None

                # col_x = col_x.view(bt, out_h * out_w, ctx.in_c * ctx.kh * ctx.kw).permute(0, 2, 1)[..., idx]    # (b, in_c * kh * kw, out_c, selected_(out_h * out_w))
                # col_x = col_x.permute(1, 0, 3, 2).reshape(ctx.in_c * ctx.kh * ctx.kw, -1, out_c)                # (in_c * kh * kw, b * selected_(out_h * out_w), out_c)
                # col_dW = einsum(col_x, col_delta_in, 'ikk bs o, bs o -> ikk o')                         # (in_c * kh * kw, out_c)
                # dw = col_dW.T.reshape(out_c, ctx.in_c, ctx.kh, ctx.kw)

                # col_delta_out = torch.zeros((bt * out_h * out_w, out_c), device=ctx.device).scatter_(0, idx.T, col_delta_in) # (b * out_h * out_w, out_c)
                # col_delta_out = torch.mm(col_delta_out, col_w.T)                 # (b * out_h * out_w, in_c * kh * kw)
                # dx = col2img(col_delta_out, (bt, ctx.in_c, ctx.in_h, ctx.in_w), ctx.kh, ctx.kw, ctx.stride, ctx.padding, ctx.out_h, ctx.out_w)


        return dx, dw, db, None, None, None, None, None, None, None, None, None, None, None, None, None


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 mode=None, percentage=0.5, unified=True, sparse=False, built_in=True, random=False, only_W=False, drop_behavior="", device='cuda'):
        '''
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the kernel
        stride: stride of the convolution
        padding: padding of the convolution
        dilation: dilation of the convolution
        groups: number of groups for grouped convolution
        bias: whether to use bias

        mode: 'normal' or 'efficient'
        percentage: percentage of the gradient to keep
        unified: whether to apply the topk operation on the kernel-level, default is True
        sparse: whether to set pruned gradients to zero, default is False
        built_in: whether to use built-in functions, default is True
        random: whether to randomly select the k largest elements, default is False
        only_W: whether to apply the topk gradient on the weights only, default is False
        drop_behavior: drop behavior, default is "", only used for sensitivity 1 analysis
        device: device to store the tensors
        '''
        super(CustomConv2d, self).__init__()
        assert mode is not None, 'Please provide mode'

        self.use_bias = bias
        self.mode = mode
        self.percentage = percentage
        self.unified = unified
        self.sparse = sparse
        self.built_in = built_in
        self.random = random
        self.only_W = only_W
        self.drop_behavior = drop_behavior
        self.device = device
        # Ensure kernel_size is a tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        # Common attributes for both modes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.weights, a=5**0.5)  # Initialize weights

        self.bias = None
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.mode == 'normal' or self.percentage == 1.0:
            return NormalConv2d.apply(x, self.weights, self.bias, self.stride, self.padding, self.dilation, self.groups, self.use_bias, self.built_in)
        elif self.mode == 'efficient':
            return EfficientConv2d.apply(x, self.weights, self.bias, self.stride, self.padding, self.dilation, self.groups, self.use_bias, self.percentage, self.unified, self.sparse, self.built_in, self.random, self.only_W, self.drop_behavior, self.device)
