import unittest


class TestProposerOptsBuild(unittest.TestCase):
    def test_line_mode_when_single_step(self):
        from latent_opt import build_proposer_opts

        opts = build_proposer_opts(
            iter_steps=1, iter_eta=None, trust_r=None, gamma_orth=0.2
        )
        self.assertEqual(opts.mode, "line")
        self.assertIsNone(opts.eta)

    def test_iter_mode_when_eta_positive(self):
        from latent_opt import build_proposer_opts

        opts = build_proposer_opts(
            iter_steps=1, iter_eta=0.1, trust_r=2.0, gamma_orth=0.0
        )
        self.assertEqual(opts.mode, "iter")
        self.assertAlmostEqual(opts.eta, 0.1)


if __name__ == "__main__":
    unittest.main()
