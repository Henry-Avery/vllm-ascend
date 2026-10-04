# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the actual MLA forward routing with CPU tensors and external op spies."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from . import test_flashmla_contract as contract
from .test_flashmla_dspark_lifecycle import MLA, load_class

api = contract.api


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_forward_calls_loaded_external_operator_and_logs_only_after_return(api, draft, capturing, fail, monkeypatch):
    tokens, heads = 3, 8
    attention = Mock(side_effect=RuntimeError("kernel launch failed")) if fail else Mock()
    # Each call owns its output, just like the external operator.
    if not fail:
        attention.side_effect = lambda *args, **kwargs: (torch.ones(heads, tokens, 512), None)
    ops = SimpleNamespace(
        __file__="/test-wheel/cann_ops_transformer/ops/__init__.py",
        flash_mla_with_kvcache=attention,
        flash_mla_with_kvcache_metadata=Mock(),
    )
    with patch.dict(api.FlashMLAAdapter.load.__func__.__globals__, import_module=Mock(return_value=ops)):
        adapter = api.FlashMLAAdapter.load(api.FlashMLAConfig(heads, 0.125, mask_mode=0 if draft else 3))
    flash = SimpleNamespace(
        adapter=adapter,
        query=torch.empty(tokens, heads, 576, dtype=torch.bfloat16),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        cache_lens=torch.tensor([3], dtype=torch.int32),
        cu=torch.tensor([0, tokens], dtype=torch.int32),
        used_q=torch.tensor([2], dtype=torch.int32),
        schedule=torch.zeros(4096, dtype=torch.int32),
        attn_mask=None if draft else torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1),
        token_live=torch.tensor([True, True, False]),
    )
    metadata = SimpleNamespace(num_decodes=1, num_prefills=0, num_decode_tokens=tokens, external_flashmla=flash)
    extra = SimpleNamespace(num_tokens=tokens, is_draft_model=draft, capturing=capturing)
    impl_cls = load_class(
        MLA,
        "AscendMLAImpl",
        ["forward", "_forward_external_flashmla"],
        dict(
            torch=torch,
            _EXTRA_CTX=extra,
            logger=api.logger,
            record_attention_compute_start=Mock(),
            maybe_save_kv_layer_to_connector=Mock(),
        ),
    )
    impl = impl_cls()
    impl.num_heads, impl.v_head_dim, impl.kv_lora_rank = heads, 4, 512
    impl.use_mla_rope = True
    impl.use_output_gate = impl.fa_quant_layer = impl.enable_mlapo = False
    impl._flashmla_logged_modes = set()
    impl.get_num_actual_tokens = lambda _: tokens
    impl._mla_preprocess = Mock(
        return_value=(
            SimpleNamespace(ql_nope=torch.ones(tokens, heads, 512), q_pe=torch.ones(tokens, heads, 64)),
            None,
        )
    )
    impl._forward_decode = Mock(side_effect=AssertionError("FIA decode must not be called"))
    impl._v_up_proj = lambda latent: latent[..., :4].transpose(0, 1).reshape(tokens, heads * 4)
    impl.o_proj = lambda value, **kwargs: (value,)
    cache = torch.zeros(2, 128, 1, 576, dtype=torch.bfloat16)
    hidden = torch.zeros(tokens, 4, dtype=torch.bfloat16)
    output = torch.empty(tokens, heads * 4, dtype=torch.bfloat16)
    api.logger.reset_mock()

    def forbid_host_read(*args, **kwargs):
        pytest.fail("Dispatch diagnostics must not read device tensor contents")

    with monkeypatch.context() as guarded:
        for method in ("item", "cpu", "numpy", "tolist"):
            guarded.setattr(torch.Tensor, method, forbid_host_read)
        if fail:
            with pytest.raises(RuntimeError, match="kernel launch failed"):
                impl.forward("layer", hidden, cache, metadata, output)
            api.logger.info_once.assert_not_called()
            assert not impl._flashmla_logged_modes
        else:
            for _ in range(2):
                assert impl.forward("layer", hidden, cache, metadata, output) is output

    assert attention.call_count == (1 if fail else 2)
    assert attention.call_args.args[1] is cache
    assert attention.call_args.kwargs["metadata"] is flash.schedule
    assert attention.call_args.kwargs["mask_mode"] == (0 if draft else 3)
    impl._forward_decode.assert_not_called()
    if not fail:
        api.logger.info_once.assert_called_once()
        args = api.logger.info_once.call_args.args
        assert args[1:3] == ("draft" if draft else "target", "capture" if capturing else "eager/warmup")
        torch.testing.assert_close(output[:2], torch.ones_like(output[:2]))
        assert torch.count_nonzero(output[2]) == 0


def test_profile_forward_cannot_report_an_external_dispatch():
    logger = Mock()
    cls = load_class(MLA, "AscendMLAImpl", ["forward"], dict(logger=logger))
    impl = cls()
    impl._forward_external_flashmla = Mock()
    output = torch.ones(1, 8)
    assert impl.forward("layer", None, None, None, output) is output
    assert torch.count_nonzero(output) == 0
    impl._forward_external_flashmla.assert_not_called()
    logger.info_once.assert_not_called()
