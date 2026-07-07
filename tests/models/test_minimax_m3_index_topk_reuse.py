# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.minimax_m3 import (
    MiniMaxM3Config,
    MiniMaxM3TextConfig,
    minimax_m3_should_skip_index_topk,
    minimax_m3_sparse_attention_layer_ids,
    minimax_m3_uses_index_topk_reuse,
)


def _mini_config(**kwargs):
    sparse_attention_freq = [0, 1, 0, 1, 1, 1, 1, 1]
    return MiniMaxM3TextConfig(
        num_hidden_layers=len(sparse_attention_freq),
        moe_layer_freq=[1] * len(sparse_attention_freq),
        sparse_attention_config={
            "sparse_attention_freq": sparse_attention_freq,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
        },
        **kwargs,
    )


def test_minimax_m3_sparse_layers_use_sparse_ordinals():
    config = _mini_config()

    assert minimax_m3_sparse_attention_layer_ids(config) == {1, 3, 4, 5, 6, 7}


def test_minimax_m3_index_topk_reuse_requires_use_index_cache():
    config = _mini_config(index_topk_freq=4)

    assert not minimax_m3_uses_index_topk_reuse(config)
    for layer_id in (1, 3, 4, 5, 6, 7):
        assert minimax_m3_should_skip_index_topk(config, layer_id)[0] is False


def test_minimax_m3_index_topk_freq_uses_sparse_layer_ordinal():
    config = _mini_config(use_index_cache=True, index_topk_freq=4)

    skips = {
        layer_id: minimax_m3_should_skip_index_topk(config, layer_id)
        for layer_id in (1, 3, 4, 5, 6, 7)
    }

    assert skips == {
        1: (False, 0),
        3: (True, 1),
        4: (True, 2),
        5: (True, 3),
        6: (False, 4),
        7: (True, 5),
    }


def test_minimax_m3_index_topk_pattern_uses_sparse_layer_ordinal():
    config = _mini_config(
        use_index_cache=True,
        index_topk_pattern=["F", "S", "F", "S", "F", "S"],
    )

    skips = {
        layer_id: minimax_m3_should_skip_index_topk(config, layer_id)
        for layer_id in (1, 3, 4, 5, 6, 7)
    }

    assert skips == {
        1: (False, 0),
        3: (True, 1),
        4: (False, 2),
        5: (True, 3),
        6: (False, 4),
        7: (True, 5),
    }


def test_minimax_m3_index_topk_offset_uses_sparse_layer_ordinal():
    config = _mini_config(
        use_index_cache=True,
        index_topk_freq=4,
        index_skip_topk_offset=1,
    )

    skips = {
        layer_id: minimax_m3_should_skip_index_topk(config, layer_id)
        for layer_id in (1, 3, 4, 5, 6, 7)
    }

    assert skips == {
        1: (False, 0),
        3: (False, 1),
        4: (True, 2),
        5: (True, 3),
        6: (True, 4),
        7: (False, 5),
    }


def test_minimax_m3_index_topk_freq_must_be_positive():
    config = _mini_config(use_index_cache=True, index_topk_freq=0)

    with pytest.raises(ValueError, match="index_topk_freq"):
        minimax_m3_uses_index_topk_reuse(config)


def test_minimax_m3_flat_index_topk_overrides_promote_to_text_config():
    config = MiniMaxM3Config()
    config.update(
        {
            "use_index_cache": True,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 1,
        }
    )

    text_config = config.get_text_config()

    assert text_config.use_index_cache is True
    assert text_config.index_topk_freq == 4
    assert text_config.index_skip_topk_offset == 1
