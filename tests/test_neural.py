import unittest
import numpy as np
import torch
from bigbrane.neural import build, count_parameters, log_predict, shifted, seed_word
from bigbrane.protocol import FitSpec


torch.set_num_threads(1)


class NeuralTests(unittest.TestCase):
    def test_shift_counts_and_seed_pairing(self):
        x = np.arange(64)[None, :] % 8
        y = shifted(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y[0, 0], 8)
        np.testing.assert_array_equal(y[:, 1:], x[:, :-1])
        for family in ('rnn', 'attention'):
            a = FitSpec(20260931, 0, 0, 0, 0, family, 0, 0)
            b = FitSpec(20260931, 0, 0, 0, 0, family, 0, 1)
            self.assertEqual(a.seed(0), b.seed(0))
            self.assertEqual(a.seed(1, 5), b.seed(1, 5))
            self.assertEqual(seed_word(a.seed(0)), int(np.random.SeedSequence(a.seed(0)).generate_state(1, dtype=np.uint64)[0]))

    def test_parameters_initialization_and_repeat(self):
        for family, count in (('rnn', 280), ('attention', 712)):
            a = build(family, 17); b = build(family, 17)
            self.assertEqual(count_parameters(a), count)
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)
                if name.endswith('bias') or name in ('bh', 'by'):
                    self.assertEqual(tensor.abs().sum().item(), 0)
            if family == 'rnn':
                torch.testing.assert_close(a.Wh @ a.Wh.T, torch.eye(8), atol=1e-6, rtol=0)
            else:
                torch.testing.assert_close(a.n1.weight, torch.ones(8))
                torch.testing.assert_close(a.n2.weight, torch.ones(8))

    def test_causality_reset_and_long_positions(self):
        x = np.array([[0, 1, 2, 3, 4], [7, 6, 5, 4, 3]])
        for family in ('rnn', 'attention'):
            model = build(family, 19)
            before = log_predict(model, x)
            changed = x.copy(); changed[:, 2:] = 7-changed[:, 2:]
            after = log_predict(model, changed)
            np.testing.assert_allclose(before[:, :3], after[:, :3], atol=1e-6, rtol=0)
            np.testing.assert_allclose(before[0], log_predict(model, x[:1])[0], atol=1e-6, rtol=0)
            np.testing.assert_allclose(before[:, 0], np.repeat(before[:1, 0], 2, axis=0), atol=1e-6, rtol=0)
            current = x.copy(); current[:, 0] = 7-current[:, 0]
            self.assertGreater(np.max(np.abs(before[:, 1]-log_predict(model, current)[:, 1])), 1e-6)
            long = log_predict(model, np.zeros((1, 256), dtype=np.int64))
            self.assertTrue(np.isfinite(long).all())
            np.testing.assert_allclose(np.exp(long).sum(-1), 1, atol=1e-12)

    def test_equations_and_gradients(self):
        inputs = torch.tensor([[8, 1, 2], [8, 3, 4]])
        for family in ('rnn', 'attention'):
            model = build(family, 23).double()
            if family == 'rnn':
                h = torch.zeros(2, 8, dtype=torch.float64); outputs = []
                for t in range(3):
                    h = torch.tanh(model.E(inputs[:, t]) @ model.We.T+h @ model.Wh.T+model.bh)
                    outputs.append(h @ model.Wy.T+model.by)
                expected = torch.stack(outputs, 1)
            else:
                pos = torch.arange(3, dtype=torch.float64)[:, None]
                scale = 10000.**(torch.arange(0, 8, 2, dtype=torch.float64)/8)
                pe = torch.empty(3, 8, dtype=torch.float64)
                pe[:, ::2] = torch.sin(pos/scale); pe[:, 1::2] = torch.cos(pos/scale)
                v = model.E(inputs)+pe
                def norm(x, layer):
                    return (x-x.mean(-1, keepdim=True))/torch.sqrt(x.var(-1, unbiased=False, keepdim=True)+1e-5)*layer.weight+layer.bias
                a = norm(v, model.n1)
                scores = (a @ model.q.weight.T) @ (a @ model.k.weight.T).transpose(1, 2)/np.sqrt(8)
                scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), 1), -torch.inf)
                r = v+torch.softmax(scores, -1) @ (a @ model.v.weight.T) @ model.o.weight.T
                h = r+torch.relu(norm(r, model.n2) @ model.w1.weight.T+model.w1.bias) @ model.w2.weight.T+model.w2.bias
                expected = h @ model.wy.weight.T+model.wy.bias
            torch.testing.assert_close(model(inputs), expected, atol=1e-12, rtol=0)
            model.zero_grad(); loss = model(inputs).square().mean(); loss.backward()
            parameter = model.E.weight
            analytic = parameter.grad[8, 0].item(); original = parameter[8, 0].item(); delta = 1e-6
            with torch.no_grad():
                parameter[8, 0] = original+delta; high = model(inputs).square().mean().item()
                parameter[8, 0] = original-delta; low = model(inputs).square().mean().item()
                parameter[8, 0] = original
            self.assertAlmostEqual(analytic, (high-low)/(2*delta), places=6)


if __name__ == '__main__':
    unittest.main()
