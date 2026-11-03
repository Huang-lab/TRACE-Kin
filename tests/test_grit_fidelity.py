"""Fidelity checks for models/trace_kin/grit.py against LiamMa/GRIT.

Needs torch and torch_geometric; skipped when they are missing. Run with
    python -m unittest tests.test_grit_fidelity -v
(or `python tests/test_grit_fidelity.py`). The reference is transcribed from
grit/transform/rrwp.py (add_full_rrwp) and grit/layer/grit_layer.py, with the
reference's own tensor layouts (fused E, Aw of shape (dh, H, 1), deg_coef of
shape (1, D, 2)), so the weight mapping below also shows how reference weights
would be loaded into this port.
"""
import importlib.util
import pathlib
import unittest

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import softmax as pyg_softmax
    HAVE_TORCH = True
except Exception:                                   # pragma: no cover
    HAVE_TORCH = False

GRIT_PATH = pathlib.Path(__file__).resolve().parents[1] / "models" / "trace_kin" / "grit.py"


def _load_grit():
    spec = importlib.util.spec_from_file_location("grit_under_test", GRIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _undirected(edges, offset=0):
    e = torch.tensor(edges).t() + offset
    return torch.cat([e, e.flip(0)], dim=1)


def _path(n):
    return [(i, i + 1) for i in range(n - 1)]


def _ref_full_rrwp(edge_index, n, steps):
    """add_full_rrwp for ONE graph; returns dict {(src, dst): vec} and node_pe."""
    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(1)
    inv = torch.where(deg > 0, 1.0 / deg.clamp(min=1), torch.zeros_like(deg))
    adj = adj * inv.view(-1, 1)
    pe = [torch.eye(n), adj]
    out = adj
    for _ in range(steps - 2):
        out = out @ adj
        pe.append(out)
    pe = torch.stack(pe, dim=-1)                                  # n x n x steps
    node_pe = pe.diagonal().transpose(0, 1)[:, 2:]                # drop I and A diagonals
    # rrwp.py:83 -- rel_pe_idx = stack([col, row]): edge (src=col, dst=row) holds pe[row, col]
    pairs = {(j, i): pe[i, j] for i in range(n) for j in range(n)}
    return pairs, node_pe


@unittest.skipUnless(HAVE_TORCH, "torch / torch_geometric not installed")
class TestRRWP(unittest.TestCase):
    steps = 21

    def setUp(self):
        self.grit = _load_grit()

    def _as_dict(self, attr, idx):
        return {(int(s), int(d)): attr[k] for k, (s, d) in enumerate(idx.t().tolist())}

    def test_matches_reference_on_branched_molecule(self):
        ei = _undirected(_path(6) + [(2, 6), (6, 7), (3, 8)])
        node_pe, attr, idx = self.grit.atom_rrwp(ei, torch.zeros(9, dtype=torch.long), self.steps)
        want, want_node = _ref_full_rrwp(ei, 9, self.steps)
        got = self._as_dict(attr, idx)
        self.assertEqual(set(got), set(want))
        for k in want:
            torch.testing.assert_close(got[k], want[k], atol=1e-6, rtol=0)
        torch.testing.assert_close(node_pe, want_node, atol=1e-6, rtol=0)

    def test_orientation_is_query_to_key(self):
        # star: centre 0 (degree 3), leaves 1..3. P[0,1] = 1/3 but P[1,0] = 1.
        ei = _undirected([(0, 1), (0, 2), (0, 3)])
        _, attr, idx = self.grit.atom_rrwp(ei, torch.zeros(4, dtype=torch.long), 4)
        d = self._as_dict(attr, idx)
        # edge (src=1, dst=0): query atom 0, key atom 1 -> walk probability 0 -> 1 = 1/3 at k=1
        self.assertAlmostEqual(float(d[(1, 0)][1]), 1 / 3, places=6)
        # edge (src=0, dst=1): query atom 1, key atom 0 -> walk probability 1 -> 0 = 1
        self.assertAlmostEqual(float(d[(0, 1)][1]), 1.0, places=6)

    def test_batch_equals_per_molecule_and_has_no_cross_pairs(self):
        mols = [(_undirected(_path(7)), 7), (_undirected(_path(5) + [(4, 0)]), 5), (_undirected(_path(3)), 3)]
        off, eis, bs = 0, [], []
        for k, (m, n) in enumerate(mols):
            eis.append(m + off); bs.append(torch.full((n,), k)); off += n
        node_pe, attr, idx = self.grit.atom_rrwp(torch.cat(eis, 1), torch.cat(bs), self.steps)
        batch = torch.cat(bs)
        self.assertTrue(bool((batch[idx[0]] == batch[idx[1]]).all()))
        self.assertEqual(idx.size(1), sum(n * n for _, n in mols))
        got, off = self._as_dict(attr, idx), 0
        for m, n in mols:
            want, want_node = _ref_full_rrwp(m, n, self.steps)
            for (s, d), v in want.items():
                torch.testing.assert_close(got[(s + off, d + off)], v, atol=1e-6, rtol=0)
            torch.testing.assert_close(node_pe[off:off + n], want_node, atol=1e-6, rtol=0)
            off += n

    def test_far_and_cross_fragment_pairs_are_kept_as_zeros(self):
        ei = torch.cat([_undirected(_path(30)), _undirected(_path(3), 30)], dim=1)   # chain + fragment
        _, attr, idx = self.grit.atom_rrwp(ei, torch.zeros(33, dtype=torch.long), self.steps)
        self.assertEqual(idx.size(1), 33 * 33)
        d = self._as_dict(attr, idx)
        for (s, t), v in d.items():
            if abs(s - t) > 20 or (s < 30) != (t < 30):
                self.assertEqual(float(v.abs().sum()), 0.0)


def _reference_layer_forward(P, x, e, src, dst, log_deg, H):
    """grit_layer.py forward (GritTransformerLayer + MultiHeadAttentionLayerGritSparse),
    dropout 0, batch norm in training mode, with the reference's own parameter layout."""
    N, D = x.shape
    dh = D // H
    Qh = F.linear(x, P["Q.w"], P["Q.b"]).view(N, H, dh)
    Kh = F.linear(x, P["K.w"]).view(N, H, dh)
    Vh = F.linear(x, P["V.w"]).view(N, H, dh)
    E = F.linear(e, P["E.w"], P["E.b"]).view(-1, H, 2 * dh)
    Ew, Eb = E[:, :, :dh], E[:, :, dh:]
    score = (Kh[src] + Qh[dst]) * Ew
    score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
    score = torch.relu(score + Eb)
    e_t = score
    wE = score.flatten(1)
    att = torch.einsum("ehd,dhc->ehc", score, P["Aw"]).clamp(-5.0, 5.0)
    att = pyg_softmax(att, dst, num_nodes=N)
    wV = torch.zeros_like(Vh).index_add_(0, dst, Vh[src] * att)
    rowV = torch.zeros_like(Vh).index_add_(0, dst, e_t * att)
    wV = wV + torch.einsum("nhd,dhc->nhc", rowV, P["VeRow"])
    h = wV.reshape(N, -1)
    h = (torch.stack([h, h * log_deg], dim=-1) * P["deg_coef"]).sum(-1)
    h = F.linear(h, P["Oh.w"], P["Oh.b"])
    e_new = F.linear(wE, P["Oe.w"], P["Oe.b"])
    h = F.batch_norm(x + h, None, None, training=True)
    e_new = F.batch_norm(e + e_new, None, None, training=True)
    f = F.linear(F.relu(F.linear(h, P["F1.w"], P["F1.b"])), P["F2.w"], P["F2.b"])
    h = F.batch_norm(h + f, None, None, training=True)
    return h, e_new


@unittest.skipUnless(HAVE_TORCH, "torch / torch_geometric not installed")
class TestLayerAgainstReference(unittest.TestCase):
    def setUp(self):
        self.grit = _load_grit()
        torch.manual_seed(0)

    def test_layer_computes_the_reference_function(self):
        D, H, dh = 32, 4, 8
        g = lambda *s: torch.randn(*s) * 0.3
        P = {"Q.w": g(D, D), "Q.b": g(D), "K.w": g(D, D), "V.w": g(D, D),
             "E.w": g(2 * D, D), "E.b": g(2 * D), "Aw": g(dh, H, 1), "VeRow": g(dh, H, dh),
             "deg_coef": g(1, D, 2), "Oh.w": g(D, D), "Oh.b": g(D), "Oe.w": g(D, D), "Oe.b": g(D),
             "F1.w": g(2 * D, D), "F1.b": g(2 * D), "F2.w": g(D, 2 * D), "F2.b": g(D)}

        layer = self.grit.GRITLayer(D, D, heads=H, dropout=0.0, attn_dropout=0.0)
        with torch.no_grad():
            layer.WQ.weight.copy_(P["Q.w"]); layer.WQ.bias.copy_(P["Q.b"])
            layer.WK.weight.copy_(P["K.w"]); layer.WV.weight.copy_(P["V.w"])
            # fused E (2D rows, laid out per head as [E_w | E_b]) -> separate E_w, E_b
            Ew = P["E.w"].view(H, 2 * dh, D); Eb = P["E.b"].view(H, 2 * dh)
            layer.WEw.weight.copy_(Ew[:, :dh].reshape(D, D)); layer.WEw.bias.copy_(Eb[:, :dh].reshape(D))
            layer.WEb.weight.copy_(Ew[:, dh:].reshape(D, D)); layer.WEb.bias.copy_(Eb[:, dh:].reshape(D))
            layer.wA.copy_(P["Aw"][:, :, 0].t())
            layer.VeRow.copy_(P["VeRow"])
            layer.theta1.copy_(P["deg_coef"][0, :, 0]); layer.theta2.copy_(P["deg_coef"][0, :, 1])
            layer.out_x.weight.copy_(P["Oh.w"]); layer.out_x.bias.copy_(P["Oh.b"])
            layer.out_e.weight.copy_(P["Oe.w"]); layer.out_e.bias.copy_(P["Oe.b"])
            layer.ffn[0].weight.copy_(P["F1.w"]); layer.ffn[0].bias.copy_(P["F1.b"])
            layer.ffn[3].weight.copy_(P["F2.w"]); layer.ffn[3].bias.copy_(P["F2.b"])
        layer.train()

        ei = _undirected(_path(7) + [(2, 7), (7, 8)])
        n = 9
        _, _, pair_index = self.grit.atom_rrwp(ei, torch.zeros(n, dtype=torch.long), 6)
        x, e = torch.randn(n, D), torch.randn(pair_index.size(1), D)
        log_deg = torch.rand(n, 1) * 2
        got_x, got_e = layer(x, e, pair_index, log_deg)
        want_x, want_e = _reference_layer_forward(P, x, e, pair_index[0], pair_index[1], log_deg, H)
        torch.testing.assert_close(got_x, want_x, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(got_e, want_e, atol=1e-4, rtol=1e-4)


@unittest.skipUnless(HAVE_TORCH, "torch / torch_geometric not installed")
class TestInitAndEncoder(unittest.TestCase):
    def setUp(self):
        self.grit = _load_grit()
        torch.manual_seed(0)

    def test_reference_init_matches_xavier_on_the_reference_shapes(self):
        D, H = 512, 8
        dh = D // H
        layer = self.grit.GRITLayer(D, D, heads=H, ref_init=True)
        xav = lambda *shape: torch.nn.init.xavier_normal_(torch.empty(*shape)).std().item()
        checks = {"WQ": (layer.WQ.weight.std().item(), xav(D, D)),
                  "E_w": (layer.WEw.weight.std().item(), xav(2 * D, D)),     # fused Linear(D, 2D)
                  "E_b": (layer.WEb.weight.std().item(), xav(2 * D, D)),
                  "wA": (layer.wA.std().item(), xav(dh, H, 1)),
                  "deg_coef": (layer.theta1.std().item(), xav(1, D, 2))}
        for name, (got, want) in checks.items():
            self.assertAlmostEqual(got / want, 1.0, delta=0.08, msg=name)
        # default init is untouched: scaler starts as the identity
        plain = self.grit.GRITLayer(D, D, heads=H)
        self.assertTrue(bool((plain.theta1 == 1).all() and (plain.theta2 == 0).all()))

    def _batch(self):
        edges = [_undirected(_path(6)), _undirected(_path(4) + [(3, 0)]), _undirected(_path(2))]
        sizes, off, eis, bs = [6, 4, 2], 0, [], []
        for k, (m, n) in enumerate(zip(edges, sizes)):
            eis.append(m + off); bs.append(torch.full((n,), k)); off += n
        ei, batch = torch.cat(eis, 1), torch.cat(bs)
        return (torch.randint(0, 20, (off, 1)), torch.randn(off, 43),
                torch.randint(1, 5, (ei.size(1), 1)), ei, batch)

    def test_no_bias_where_the_reference_has_none(self):
        enc = self.grit.GRITLigandEncoder(43, 32, n_layers=1, grit_dim=16, pe_steps=8)
        self.assertIsNone(enc.pair_pe_in.bias)
        self.assertIsNone(enc.layers[0].WK.bias)
        self.assertIsNone(enc.layers[0].WV.bias)
        self.assertIsNotNone(enc.layers[0].WQ.bias)

    def test_forward_backward_is_finite_for_both_inits(self):
        for ref_init in (False, True):
            enc = self.grit.GRITLigandEncoder(43, 32, n_layers=2, grit_dim=16, pe_steps=8,
                                              dropout=0.1, ref_init=ref_init)
            enc.train()
            atom_h, mol_pool, p = enc(*self._batch())
            self.assertEqual(tuple(atom_h.shape), (12, 32))
            self.assertEqual(tuple(mol_pool.shape), (3, 32))
            self.assertIsNone(p)
            (atom_h.square().mean() + mol_pool.square().mean()).backward()
            for name, prm in enc.named_parameters():
                self.assertTrue(prm.grad is None or bool(torch.isfinite(prm.grad).all()), name)


if __name__ == "__main__":
    unittest.main()
