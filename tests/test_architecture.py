"""Offline architecture checks with synthetic pretrained-backbone outputs."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from ginep import GINEPConfig, HEAD_TYPES, build_model
from heads import extract_3class_probs
from ligand_graph import mol_to_atom_graph


class DummyBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(32, dim)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def synthetic_batch():
    graphs = [mol_to_atom_graph("CCO"), mol_to_atom_graph("c1ccccc1")]
    lengths = [g["atom_features"].size(0) for g in graphs]
    return {
        "full_ids": torch.ones(2, 8, dtype=torch.long),
        "full_mask": torch.ones(2, 8, dtype=torch.long),
        "pocket_ids": torch.ones(2, 6, dtype=torch.long),
        "pocket_mask": torch.ones(2, 6, dtype=torch.long),
        "tm_ids": torch.ones(2, 7, 4, dtype=torch.long),
        "tm_masks": torch.ones(2, 7, 4, dtype=torch.long),
        "motif_feat": torch.zeros(2, 12),
        "lig_ids": torch.ones(2, 5, dtype=torch.long),
        "lig_mask": torch.ones(2, 5, dtype=torch.long),
        "pharm_feat": torch.zeros(2, 16),
        "atom_features": torch.cat([g["atom_features"] for g in graphs]),
        "atom_edge_index": torch.cat([graphs[0]["atom_edge_index"],
                                       graphs[1]["atom_edge_index"] + lengths[0]], dim=1),
        "atom_edge_attr": torch.cat([g["atom_edge_attr"] for g in graphs]),
        "atom_rw_pe": torch.cat([g["atom_rw_pe"] for g in graphs]),
        "batch_atom": torch.cat([torch.full((n,), i, dtype=torch.long)
                                  for i, n in enumerate(lengths)]),
    }


class ArchitectureTest(unittest.TestCase):
    def test_full_forward_and_backward(self):
        cfg = GINEPConfig()
        torch.manual_seed(42)
        with patch("model.load_esmc_model", return_value=DummyBackbone(cfg.protein_hidden)), \
             patch("transformers.AutoModel.from_pretrained", return_value=DummyBackbone(cfg.ligand_hidden)):
            model = build_model(cfg)
        model.eval()
        output = model(synthetic_batch())
        self.assertEqual(tuple(output["logits"].shape), (2, 3))
        self.assertEqual(tuple(output["binary_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["repr"].shape), (2, 256))
        self.assertEqual(tuple(output["tm_attn"].shape), (2, 4, 6, 28))
        for value in output.values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.isfinite(value).all())
        probs = extract_3class_probs(output)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
        (output["logits"].square().mean() + output["binary_logits"].square().mean()).backward()
        for layer in (model.graph_deep_enc.token_proj, model.prot_enc.full_proj,
                      model.lig_enc.smiles_proj, model.cross_attn.W_q, model.head.net[0]):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_all_classification_heads(self):
        for head in HEAD_TYPES:
            with self.subTest(head=head):
                cfg = GINEPConfig()
                cfg.head_type = head
                cfg.device = "cpu"
                with patch("model.load_esmc_model", return_value=DummyBackbone(cfg.protein_hidden)), \
                     patch("transformers.AutoModel.from_pretrained", return_value=DummyBackbone(cfg.ligand_hidden)):
                    model = build_model(cfg).eval()
                output = model(synthetic_batch())
                self.assertEqual(output["head_type"], head)
                self.assertEqual(tuple(output["logits"].shape), (2, 3))
                self.assertEqual("binary_logits" in output, head == "mlp")
                if head == "cond_softmax":
                    self.assertEqual(tuple(output["active_logit"].shape), (2,))
                    self.assertEqual(tuple(output["cond_logits"].shape), (2, 2))
                probs = extract_3class_probs(output)
                self.assertTrue(torch.isfinite(probs).all())
                self.assertTrue((probs >= 0).all())
                torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
                loss = -torch.log(probs[:, 1].clamp_min(1e-7)).mean()
                loss.backward()
                grads = [p.grad for p in model.head.parameters() if p.requires_grad]
                self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))
                self.assertTrue(torch.isfinite(model.graph_deep_enc.token_proj.weight.grad).all())

    def test_unknown_head_rejected(self):
        cfg = GINEPConfig()
        cfg.head_type = "unsupported"
        with self.assertRaises(ValueError):
            build_model(cfg)

    def test_single_atom_and_invalid_smiles_placeholder(self):
        for smiles in ("[Na+]", "", "invalid"):
            graph = mol_to_atom_graph(smiles)
            self.assertEqual(graph["atom_features"].shape[1], 34)
            self.assertEqual(graph["atom_edge_attr"].shape[1], 12)
            self.assertEqual(graph["atom_rw_pe"].shape[1], 20)
            self.assertTrue(torch.isfinite(graph["atom_rw_pe"]).all())


if __name__ == "__main__":
    unittest.main()
