# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import regex as re

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.platforms import current_platform


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, quantization tests only run GPUs
    if current_platform.is_cpu():
        return False
    try:
        current_platform.verify_quantization(quant_method)
    except ValueError:
        return False
    if current_platform.is_xpu():
        return True
    capability = current_platform.get_device_capability()
    assert capability is not None

    min_capability = get_quantization_config(quant_method).get_min_capability()

    return capability.to_int() >= min_capability


def _test_online_quant_peak_mem_impl(
    quantization_arg_value,
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch,
    *,
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    runner_kwargs: dict | None = None,
    expected_model_memory_gib: float = 6.7,
    max_peak_over_model_gib: float | None = None,
    expected_log_substring: str | None = None,
) -> None:
    # Note: `allenai/OLMoE-1B-7B-0125-Instruct` was selected because:
    # 1. it covers both Linear and MoE paths
    # 2. it is already used by other tests in CI, so adding it here
    #    does not increase disk space for CI runners
    # I really wanted to use `ibm-granite/granite-3.0-1b-a400m-base`
    # which I think is the smallest MoE model in vLLM (2.5 GiB bf16,
    # 1.3 GiB fp8), but could not as adding one more model makes CI
    # run out of disk space.
    # Force spawn to ensure caplog_mp_spawn works consistently
    # (it relies on VLLM_LOGGING_CONFIG_PATH which spawn reads but fork ignores)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    effective_runner_kwargs = dict(runner_kwargs or {})
    if quantization_arg_value is not None:
        effective_runner_kwargs["quantization"] = quantization_arg_value

    with (
        caplog_mp_spawn(logging.DEBUG) as log_holder,
        vllm_runner(
            model_name,
            enforce_eager=True,
            **effective_runner_kwargs,
        ) as llm,
    ):
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        assert outputs and outputs[0][0], "Model returned no generated token IDs"
        print(outputs[0][1])

    log_text = log_holder.text
    if expected_log_substring is not None:
        assert expected_log_substring in log_text, (
            f"Could not find expected quantization path: {expected_log_substring}"
        )

    # Parse memory usage from captured logs
    model_memory_gib = None
    peak_memory_gib = None
    for line in log_text.splitlines():
        if model_memory_gib is None:
            match = re.search(r"Model loading took ([\d.]+) GiB memory", line)
            if match:
                model_memory_gib = float(match.group(1))
        if peak_memory_gib is None:
            match = re.search(
                r"Peak GPU memory after loading weights: ([\d.]+) GiB", line
            )
            if match:
                peak_memory_gib = float(match.group(1))

    assert model_memory_gib is not None, "Could not find model loading memory log"
    assert peak_memory_gib is not None, "Could not find peak memory log"
    print(f"GPU memory used after loading weights: {model_memory_gib} GiB")
    print(f"Peak GPU memory usage while loading weights: {peak_memory_gib} GiB")

    # for allenai/OLMoE-1B-7B-0125-Instruct the number we see today is 9.06
    # GiB on CUDA, which is 1.36x above model_memory_gib. A slightly higher
    # number is expected as when we load and quantize weights in a streaming
    # fashion we need to have individual weights in bf16 + fp8 alive at the
    # same time.
    assert model_memory_gib < expected_model_memory_gib, (
        f"{model_memory_gib=} higher than {expected_model_memory_gib}"
    )
    if max_peak_over_model_gib is None:
        expected_peak_memory_gib = expected_model_memory_gib * 1.4
        assert peak_memory_gib < expected_peak_memory_gib, (
            f"{peak_memory_gib=} higher than {expected_peak_memory_gib}"
        )
    else:
        peak_over_model_gib = peak_memory_gib - model_memory_gib
        assert peak_over_model_gib < max_peak_over_model_gib, (
            f"{peak_over_model_gib=} higher than {max_peak_over_model_gib}"
        )
