# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL block-scaled FP8 projection for single-token gfx1201 decode."""

from __future__ import annotations

import functools
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

_SHAPE_WAVES = {
    (2048, 5120): 8,
    (4352, 5120): 8,
    (5120, 2176): 4,
    (5120, 768): 4,
}


def _raw(value: Any) -> Any:
    return value.ir_value() if hasattr(value, "ir_value") else value


@functools.lru_cache
def _on_r9700(device: torch.device) -> bool:
    arch = torch.cuda.get_device_properties(device).gcnArchName
    name = torch.cuda.get_device_name(device)
    return arch.split(":", 1)[0] == "gfx1201" and "R9700" in name


def supports(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    output_dtype: torch.dtype,
    block_size: list[int],
) -> bool:
    if (
        weight.device.type != "cuda"
        or activation.device != weight.device
        or activation_scales.device != weight.device
        or weight_scales.device != weight.device
    ):
        return False
    if not _on_r9700(weight.device):
        return False
    if output_dtype != torch.bfloat16 or block_size != [128, 128]:
        return False
    if activation.dtype != torch.float8_e4m3fn:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.shape not in _SHAPE_WAVES:
        return False

    n, k = weight.shape
    return (
        activation.ndim == 2
        and activation.shape == (1, k)
        and activation.is_contiguous()
        and activation.data_ptr() % 8 == 0
        and weight.stride(1) == 1
        and weight.stride(0) >= k
        and weight.stride(0) <= 0x7FFFFFFF
        and weight.stride(0) % 8 == 0
        and weight.data_ptr() % 8 == 0
        and activation_scales.dtype == torch.float32
        and activation_scales.shape == (1, k // 128)
        and activation_scales.is_contiguous()
        and weight_scales.dtype == torch.float32
        and weight_scales.shape == (n // 128, k // 128)
        and weight_scales.is_contiguous()
    )


@functools.lru_cache
def _build_launcher(n: int, k: int) -> Any:
    waves = _SHAPE_WAVES[(n, k)]
    groups = k // 128
    block = waves * 32
    grid = n // (waves * 16)

    @flyc.kernel(
        name=f"rdna4_fp8_m1_n{n}_k{k}_w{waves}_p2",
        known_block_size=[block, 1, 1],
    )
    def projection_kernel(
        activation: fx.Tensor,
        weight: fx.Tensor,
        output: fx.Tensor,
        activation_scales: fx.Tensor,
        weight_scales: fx.Tensor,
        weight_stride_n: fx.Int32,
    ):
        ptr1 = ir.Type.parse("!llvm.ptr<1>")
        v2i32 = ir.Type.parse("vector<2xi32>")
        tid = gpu.thread_id("x")
        bid = gpu.block_id("x")
        wave = tid // 32
        lane = tid % 32
        lane16 = lane % 16
        k_half = lane // 16
        column = bid * (waves * 16) + wave * 16 + lane16

        activation_base = fx.Int64(fx.ptrtoint(fx.get_iter(activation)))
        weight_base = fx.Int64(fx.ptrtoint(fx.get_iter(weight)))
        output_base = fx.Int64(fx.ptrtoint(fx.get_iter(output)))
        activation_scale_base = fx.Int64(fx.ptrtoint(fx.get_iter(activation_scales)))
        weight_scale_base = fx.Int64(fx.ptrtoint(fx.get_iter(weight_scales)))

        def ptr(base: Any, byte_offset: Any) -> Any:
            return llvm.inttoptr(ptr1, _raw(base + fx.Int64(byte_offset)))

        def load_activation(group: Any, step: int) -> Any:
            k_byte = fx.Int64(group * 128 + step * 16) + fx.Int64(k_half * 8)
            return llvm.LoadOp(v2i32, ptr(activation_base, k_byte), alignment=8).result

        def load_weight(group: Any, step: int) -> Any:
            k_byte = fx.Int64(group * 128 + step * 16) + fx.Int64(k_half * 8)
            byte_offset = fx.Int64(column) * fx.Int64(weight_stride_n) + k_byte
            return llvm.LoadOp(v2i32, ptr(weight_base, byte_offset), alignment=8).result

        total = fx.Float32(0.0)
        for group, state in range(0, groups, 1, init=[total]):
            loop_total = fx.Float32(state[0])
            group_acc = fx.full(8, 0.0, fx.Float32)
            activation_fragment = load_activation(group, 0)
            weight_fragment = load_weight(group, 0)

            for step in range_constexpr(7):
                next_activation_fragment = load_activation(group, step + 1)
                next_weight_fragment = load_weight(group, step + 1)
                group_acc = rocdl.wmma_f32_16x16x16_fp8_fp8(
                    group_acc.type,
                    activation_fragment,
                    weight_fragment,
                    group_acc,
                ).result
                activation_fragment = next_activation_fragment
                weight_fragment = next_weight_fragment

            group_acc = rocdl.wmma_f32_16x16x16_fp8_fp8(
                group_acc.type,
                activation_fragment,
                weight_fragment,
                group_acc,
            ).result
            activation_scale = llvm.LoadOp(
                T.f32, ptr(activation_scale_base, group * 4), alignment=4
            ).result
            scale_index = (column // 128) * groups + group
            weight_scale = llvm.LoadOp(
                T.f32,
                ptr(weight_scale_base, fx.Int64(scale_index) * fx.Int64(4)),
                alignment=4,
            ).result
            group_value = fx.Vector(group_acc)[0]
            scaled_group = fx.Float32(group_value) * fx.Float32(activation_scale)
            next_total = fx.Float32(
                fmath.fma(scaled_group, fx.Float32(weight_scale), loop_total)
            )
            results = yield [next_total]

        total = fx.Float32(results)
        if lane < 16:
            output_ptr = ptr(output_base, fx.Int64(column) * fx.Int64(2))
            llvm.StoreOp(_raw(total.to(fx.BFloat16)), output_ptr, alignment=2)

    @flyc.jit
    def launch(
        activation: fx.Tensor,
        weight: fx.Tensor,
        output: fx.Tensor,
        activation_scales: fx.Tensor,
        weight_scales: fx.Tensor,
        weight_stride_n: fx.Int32,
        stream: fx.Stream,
    ):
        projection_kernel(
            activation,
            weight,
            output,
            activation_scales,
            weight_scales,
            weight_stride_n,
        ).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def run(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
) -> None:
    n, k = weight.shape
    converted = [
        flyc.from_dlpack(tensor)
        for tensor in (
            activation,
            weight,
            output,
            activation_scales,
            weight_scales,
        )
    ]
    stream = fx.Stream(torch.cuda.current_stream(weight.device).cuda_stream)
    _build_launcher(n, k)(*converted, weight.stride(0), stream=stream)
