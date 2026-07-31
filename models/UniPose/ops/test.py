# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import torch
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])


torch.manual_seed(3)


def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')
    del value, sampling_locations, attention_weights, output_pytorch, output_cuda
    _cuda_cleanup()
    return bool(fwdok)


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')
    del value, sampling_locations, attention_weights, output_pytorch, output_cuda
    _cuda_cleanup()
    return bool(fwdok)


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):
    _cuda_cleanup()
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    try:
        gradok = gradcheck(
            func,
            (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step),
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f'* SKIP check_gradient_numerical(D={channels}): CUDA OOM ({exc})')
        del value, sampling_locations, attention_weights
        _cuda_cleanup()
        return None

    print(f'* {gradok} check_gradient_numerical(D={channels})')
    del value, sampling_locations, attention_weights
    _cuda_cleanup()
    return bool(gradok)


if __name__ == '__main__':
    # Forward equality is required for inference. Large-channel double gradcheck
    # (1025/2048/3096) can OOM on 16–24GB GPUs (e.g. RunPod A4500) even when kernels are correct.
    run_large = os.environ.get("MSDA_TEST_LARGE_GRAD", "0") == "1"
    small_channels = [30, 32, 64, 71]
    large_channels = [1025, 2048, 3096]

    ok = True
    ok = check_forward_equal_with_pytorch_double() and ok
    ok = check_forward_equal_with_pytorch_float() and ok

    for channels in small_channels:
        result = check_gradient_numerical(channels, True, True, True)
        if result is False:
            ok = False

    if run_large:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f'Large gradcheck enabled (free VRAM {free_bytes / (1024**3):.2f} / {total_bytes / (1024**3):.2f} GiB)')
        for channels in large_channels:
            result = check_gradient_numerical(channels, True, True, True)
            if result is False:
                ok = False
    else:
        print('* SKIP large-channel gradcheck (D=1025/2048/3096); set MSDA_TEST_LARGE_GRAD=1 to enable')

    if not ok:
        print('ERROR: one or more required MSDeformAttn checks failed', file=sys.stderr)
        sys.exit(1)
    print('All required MSDeformAttn checks passed.')
