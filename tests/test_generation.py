"""CPU regression checks for conditional generation objectives and sequence masks."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WIDE = load_module("generation_wide", ROOT / "model_gen_comparison/model_gen/model_gen.py")
DEEP = load_module("generation_deep", ROOT / "model_gen_comparison/model_gen_deep/model_gen_deep.py")


class Vocabulary:
    s2i = {"<sos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3, "C": 4, "O": 5}

    def __len__(self):
        return len(self.s2i)


def batch():
    torch.manual_seed(19)
    return SimpleNamespace(
        b_cls=torch.randn(2, 4), e_cls=torch.randn(2, 3), morgan=torch.randn(2, 6),
        b_seq=torch.randn(2, 3, 4), b_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        e_seq=torch.randn(2, 3, 5), e_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]),
        target_seq=torch.tensor([[0, 4, 5, 1], [0, 4, 1, 2]]),
        y=torch.tensor([1.0, 0.0]), affinity=torch.tensor([8.0, 4.0]),
    )


def extra_padding(data):
    copy = SimpleNamespace(**vars(data))
    for sequence, mask in (("b_seq", "b_mask"), ("e_seq", "e_mask")):
        x = getattr(data, sequence)
        valid = getattr(data, mask)
        x = x.masked_fill(~valid.bool().unsqueeze(-1), float("nan"))
        setattr(copy, sequence, torch.cat([x, torch.full((2, 2, x.size(-1)), float("nan"))], dim=1))
        setattr(copy, mask, torch.cat([valid, torch.zeros(2, 2, dtype=valid.dtype)], dim=1))
    return copy


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.models = {
            "wide": WIDE.FusionDeepDTAGen(Vocabulary(), bermol_dim=4, esm_seq_dim=5, esm_cls_dim=3, morgan_dim=6).eval(),
            "deep": DEEP.FusionDeepDTAGen(Vocabulary(), bermol_dim=4, esm_seq_dim=5).eval(),
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def forward_with_scores(self, model, data):
        captured = []
        handle = model.word_pred.register_forward_hook(lambda module, args, result: captured.append(result))
        try:
            with patch("torch.randn_like", side_effect=torch.zeros_like):
                output = model(data)
        finally:
            handle.remove()
        return output, captured[-1]

    def test_padding_does_not_change_logits_or_losses(self):
        data = batch()
        padded = extra_padding(data)
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                base, scores = self.forward_with_scores(model, data)
                added, added_scores = self.forward_with_scores(model, padded)
                for index in (0, 2, 3, 4):
                    self.assertTrue(torch.isfinite(added[index]).all())
                    torch.testing.assert_close(base[index], added[index], atol=3e-5, rtol=1e-4)
                torch.testing.assert_close(scores, added_scores, atol=3e-5, rtol=1e-4)

    def test_masked_features_have_zero_finite_gradients(self):
        for name, model in self.models.items():
            with self.subTest(model=name):
                model.zero_grad(set_to_none=True)
                data = extra_padding(batch())
                data.b_seq.requires_grad_()
                data.e_seq.requires_grad_()
                output, _ = self.forward_with_scores(model, data)
                (output[2] + output[3] + 0.001 * output[4]).backward()
                for sequence, mask in ((data.b_seq, data.b_mask), (data.e_seq, data.e_mask)):
                    self.assertTrue(torch.isfinite(sequence.grad).all())
                    self.assertEqual(sequence.grad[~mask.bool()].abs().sum().item(), 0)
                    self.assertGreater(sequence.grad[mask.bool()].abs().sum().item(), 0)

    def test_language_loss_predicts_next_token(self):
        data = batch()
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                output, scores = self.forward_with_scores(model, data)
                expected = F.cross_entropy(scores[:, :-1].reshape(-1, 6), data.target_seq[:, 1:].reshape(-1), ignore_index=2)
                same_position = F.cross_entropy(scores.reshape(-1, 6), data.target_seq.reshape(-1), ignore_index=2)
                torch.testing.assert_close(output[3], expected)
                self.assertGreater((output[3] - same_position).abs().item(), 1e-5)

    def test_decoder_cannot_read_future_targets(self):
        data = batch()
        changed = SimpleNamespace(**vars(data))
        changed.target_seq = data.target_seq.clone()
        changed.target_seq[:, 2] = 5
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                _, before = self.forward_with_scores(model, data)
                _, after = self.forward_with_scores(model, changed)
                torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-5)

    def test_decoder_receives_target_padding_mask(self):
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                observed = []
                handle = model.decoder[0].layer.register_forward_pre_hook(
                    lambda layer, args, kwargs: observed.append(kwargs["tgt_key_padding_mask"]), with_kwargs=True)
                try:
                    self.forward_with_scores(model, batch())
                finally:
                    handle.remove()
                torch.testing.assert_close(observed[0], batch().target_seq.eq(2))

    def test_kl_uses_the_sampling_distribution(self):
        data = batch()
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                captured = {}
                handles = [
                    model.z_mean.register_forward_hook(lambda module, args, value: captured.update(mu=value)),
                    model.z_logvar.register_forward_hook(lambda module, args, value: captured.update(raw=value)),
                ]
                try:
                    output, _ = self.forward_with_scores(model, data)
                finally:
                    for handle in handles:
                        handle.remove()
                effective = -captured["raw"].float().abs() if name == "wide" else 6 * captured["raw"].float().tanh()
                terms = -0.5 * (1 + effective - captured["mu"].float().square() - effective.exp())
                if name == "wide":
                    expected = terms.masked_fill(~data.b_mask.bool().T.unsqueeze(-1), 0).sum(dim=(0, 2)).mean()
                else:
                    expected = terms.sum(dim=-1).mean()
                torch.testing.assert_close(output[4], expected)
                with patch("torch.randn_like", side_effect=torch.ones_like):
                    sample = model.reparameterize(captured["mu"], captured["raw"])
                torch.testing.assert_close(sample, captured["mu"] + (0.5 * effective).exp())

    def test_empty_masks_and_targets_fail_explicitly(self):
        for name, model in self.models.items():
            with self.subTest(model=name):
                data = batch()
                data.b_mask[0] = 0
                with self.assertRaises(ValueError):
                    model(data)
                data = batch()
                data.target_seq = data.target_seq[:, :1]
                with self.assertRaises(ValueError):
                    model(data)

    def test_zero_affinity_is_preserved_and_bad_temperature_rejected(self):
        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                values = []
                handle = model.affinity_proj.register_forward_pre_hook(lambda module, args: values.append(args[0]))
                try:
                    with patch.object(model, "_generate", return_value=None):
                        model.generate(batch(), affinity_override=0.0)
                finally:
                    handle.remove()
                torch.testing.assert_close(values[0], torch.zeros(2, 1))
                for temperature in (0, -1, float("nan")):
                    with self.assertRaises(ValueError):
                        model._generate(torch.zeros(1, 2, 384), torch.zeros(2, 1, dtype=torch.bool), temperature=temperature)

    def test_finished_sequences_remain_padded(self):
        class Decoder(nn.Module):
            def forward(self, x, memory, **kwargs):
                return x

        class ScheduledTokens(nn.Module):
            def __init__(self):
                super().__init__()
                self.step = 0

            def forward(self, x):
                scores = x.new_full((2, 6), -10)
                tokens = (1, 4) if self.step == 0 else (5, 1)
                self.step += 1
                scores[0, tokens[0]] = 10
                scores[1, tokens[1]] = 10
                return scores

        for name, model in self.models.items():
            with self.subTest(model=name), torch.no_grad():
                with patch.object(model, "decoder", nn.ModuleList([Decoder()])), \
                     patch.object(model, "word_pred", ScheduledTokens()), patch.object(model, "max_len", 4):
                    tokens = model._generate(torch.zeros(1, 2, 384), torch.zeros(2, 1, dtype=torch.bool))
                torch.testing.assert_close(tokens[0], torch.tensor([1, 2, 2]))
                torch.testing.assert_close(tokens[1], torch.tensor([4, 1, 2]))

    def test_deep_entry_points_share_training_and_accept_legacy_arguments(self):
        folder = ROOT / "model_gen_comparison/model_gen_deep"
        sys.path.insert(0, str(folder))
        names = ("train_gen_deep_nowide", "model_gen_deep", "utils")
        saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
        try:
            direct = load_module("deep_training_entry", folder / "train_gen_deep_nowide.py")
            legacy = load_module("legacy_deep_training_entry", folder / "train_gen_deep.py")
            args = ["--data_path", "data.csv", "--bermol_path", "bermol", "--esm_seq_path", "esm", "--num_workers", "0"]
            parsed = direct.build_parser().parse_args(args)
            compatible = legacy.build_parser().parse_args(args + ["--esm_cls_path", "unused", "--morgan_dim", "1024"])
            self.assertEqual(parsed.esm_seq_path, compatible.esm_seq_path)
            self.assertEqual(compatible.num_workers, 0)
            with patch.object(legacy, "train") as training:
                legacy.main(args)
                self.assertEqual(training.call_count, 1)
                self.assertEqual(training.call_args.args[0].esm_seq_path, "esm")
        finally:
            sys.path.remove(str(folder))
            for name in names:
                sys.modules.pop(name, None)
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
