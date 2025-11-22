import re


def test_sidebar_shows_step_scores():
    # Stub streamlit with sidebar write capture
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic(pre_images=False)
    # Build a minimal latent state with non-zero w so scores are non-trivial
    from latent_state import init_latent_state

    ls = init_latent_state()  # defaults to 448x448 → reasonable d
    ls.w[0] = 1.0
    # Render step scores for 3 steps using Ridge scorer
    from ipo.ui import ui

    ui.render_iter_step_scores(
        st, ls, prompt="p", vm_choice="Ridge", iter_steps=3, iter_eta=None, trust_r=None
    )
    writes = getattr(st, "sidebar_writes", [])
    # Expect a consolidated line with three numeric step scores
    matched = False
    for w in writes:
        if w.startswith("Step scores:"):
            # e.g., "Step scores: 0.123, 0.246, 0.369"
            nums = re.findall(r"-?\d+\.\d+", w)
            matched = len(nums) >= 3
            break
    assert matched, f"no per-step scores rendered; writes={writes}"
