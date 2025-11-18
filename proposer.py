from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from latent_logic import (
    propose_pair_prompt_anchor_iterative,
    propose_pair_prompt_anchor_linesearch,
)
from latent_state import LatentState


@dataclass
class ProposerOpts:
    mode: str = "line"          # 'line' or 'iter'
    trust_r: Optional[float] = None
    gamma: float = 0.0
    steps: int = 3
    eta: Optional[float] = None


def propose_next_pair(
    state: LatentState,
    prompt: str,
    *,
    mode: str = "line",
    trust_r: Optional[float] = None,
    gamma: float = 0.0,
    steps: int = 3,
    eta: Optional[float] = None,
    opts: Optional[ProposerOpts] = None,
):
    """Unified proposer API that wraps iterative and line-search variants.

    - mode='line' → line-search along w (default).
    - mode='iter' → small projected steps along w; honors `steps` and `eta`.
    You can pass either explicit kwargs or a ProposerOpts instance via `opts`.
    """
    if opts is not None:
        mode = opts.mode
        trust_r = opts.trust_r
        gamma = opts.gamma
        steps = opts.steps
        eta = opts.eta
    if str(mode).lower() == "iter":
        return propose_pair_prompt_anchor_iterative(
            state, prompt, steps=int(max(1, steps)), eta=eta, trust_r=trust_r, gamma=gamma
        )
    return propose_pair_prompt_anchor_linesearch(state, prompt, trust_r=trust_r, gamma=gamma)


def build_proposer_opts(
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
    gamma_orth: float,
) -> ProposerOpts:
    """Small helper to build ProposerOpts from UI values.

    Keeps the logic for choosing 'iter' vs 'line' in one place.
    """
    mode = "iter" if (int(iter_steps) > 1 or (iter_eta is not None and float(iter_eta) > 0.0)) else "line"
    eta = float(iter_eta) if (iter_eta is not None and float(iter_eta) > 0.0) else None
    return ProposerOpts(mode=mode, trust_r=trust_r, gamma=float(gamma_orth), steps=int(iter_steps), eta=eta)
