from latent_state import (
    LatentState,
    init_latent_state,
    save_state,
    load_state,
    dumps_state,
    loads_state,
    state_summary,
)
from latent_logic import (
    propose_latent_pair_ridge,
    propose_pair_prompt_anchor,
    propose_pair_prompt_anchor_iterative,
    propose_pair_prompt_anchor_linesearch,
    z_to_latents,
    z_from_prompt,
    update_latent_ridge,
)
from dataclasses import dataclass
from typing import Optional
from latent_logic import (
    propose_pair_prompt_anchor_iterative as _propose_iter,
    propose_pair_prompt_anchor_linesearch as _propose_line,
)


@dataclass
class ProposerOpts:
    mode: str = "line"  # 'line' or 'iter'
    trust_r: Optional[float] = None
    gamma: float = 0.0
    steps: int = 3
    eta: Optional[float] = None


def build_proposer_opts(
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
    gamma_orth: float,
) -> ProposerOpts:
    mode = (
        "iter"
        if (int(iter_steps) > 1 or (iter_eta is not None and float(iter_eta) > 0.0))
        else "line"
    )
    eta = float(iter_eta) if (iter_eta is not None and float(iter_eta) > 0.0) else None
    return ProposerOpts(
        mode=mode,
        trust_r=trust_r,
        gamma=float(gamma_orth),
        steps=int(iter_steps),
        eta=eta,
    )


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
    if opts is not None:
        mode = opts.mode
        trust_r = opts.trust_r
        gamma = opts.gamma
        steps = opts.steps
        eta = opts.eta
    if str(mode).lower() == "iter":
        return _propose_iter(
            state,
            prompt,
            steps=int(max(1, steps)),
            eta=eta,
            trust_r=trust_r,
            gamma=gamma,
        )
    return _propose_line(state, prompt, trust_r=trust_r, gamma=gamma)

__all__ = [
    "LatentState",
    "init_latent_state",
    "save_state",
    "load_state",
    "dumps_state",
    "loads_state",
    "state_summary",
    "propose_latent_pair_ridge",
    "propose_pair_prompt_anchor",
    "propose_pair_prompt_anchor_iterative",
    "propose_pair_prompt_anchor_linesearch",
    "propose_next_pair",
    "build_proposer_opts",
    "ProposerOpts",
    "z_to_latents",
    "z_from_prompt",
    "update_latent_ridge",
]
