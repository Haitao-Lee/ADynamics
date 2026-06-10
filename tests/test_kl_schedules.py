"""
Smoke tests for the KL schedule dispatcher in utils/kl_schedules.py.

Verifies:
  A) linear_kl_weight ramps 0 -> target over warmup, then constant
  B) cyclical_kl_weight oscillates between low_frac*target and target
  C) get_kl_weight dispatches based on config["kl_strategy"]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kl_schedules import (
    linear_kl_weight,
    cyclical_kl_weight,
    get_kl_weight,
)


def test_A_linear_kl() -> None:
    """A) Linear schedule: 0 -> target, then constant."""
    print("\n[Test A] linear_kl_weight (legacy)")
    target = 0.3
    warmup = 10
    # Before warmup: ramps linearly
    assert abs(linear_kl_weight(0, target, warmup) - target / warmup) < 1e-6
    assert abs(linear_kl_weight(4, target, warmup) - target * 5 / warmup) < 1e-6
    # At end of warmup: target
    assert abs(linear_kl_weight(warmup - 1, target, warmup) - target) < 1e-6
    # After warmup: constant target
    assert linear_kl_weight(100, target, warmup) == target
    print(f"  [OK] ramps 0->{target} over {warmup} epochs, then constant")


def test_B_cyclical_kl() -> None:
    """B) Cyclical schedule oscillates between low_frac*target and target."""
    print("\n[Test B] cyclical_kl_weight")
    target = 0.3
    cycle_len = 10
    low_frac = 0.1
    half = cycle_len / 2.0  # 5.0
    # First half of cycle: should climb
    v0 = cyclical_kl_weight(0, target, cycle_len, low_frac)
    v_mid_climb = cyclical_kl_weight(int(half) - 1, target, cycle_len, low_frac)
    assert abs(v0 - target * low_frac) < 1e-6, f"epoch 0 should be at low_frac, got {v0}"
    assert v_mid_climb > v0, f"late in climb should be higher than start, got {v_mid_climb} vs {v0}"
    # Peak of climb: pos = half - 1 -> frac = (half-1)/half = 0.8 with cycle_len=10
    # weight = target * (low_frac + (1-low_frac)*0.8) = 0.3 * 0.82 = 0.246
    expected_peak = target * (low_frac + (1.0 - low_frac) * (half - 1) / half)
    assert abs(v_mid_climb - expected_peak) < 1e-6, (
        f"peak should be {expected_peak}, got {v_mid_climb}"
    )
    # Second half: should drop. The lowest point in the cycle is at the
    # boundary (pos_in_cycle wraps to 0), not at the last epoch — the climb
    # back up starts as soon as pos crosses half.
    v_drop = cyclical_kl_weight(cycle_len - 1, target, cycle_len, low_frac)
    assert v_drop < v_mid_climb, f"end of cycle should be lower than peak, got {v_drop} vs {v_mid_climb}"
    # v_drop is just before the boundary: pos = cycle_len-1, half=5.0
    # frac = 1.0 - (cycle_len-1 - half)/half = 1.0 - 4/5 = 0.2
    # weight = 0.3 * (0.1 + 0.9*0.2) = 0.3 * 0.28 = 0.084
    expected_drop = target * (low_frac + (1.0 - low_frac) * 0.2)
    assert abs(v_drop - expected_drop) < 1e-6, (
        f"last epoch of cycle should be at {expected_drop}, got {v_drop}"
    )
    # Next cycle starts again from low
    v_next = cyclical_kl_weight(cycle_len, target, cycle_len, low_frac)
    assert abs(v_next - target * low_frac) < 1e-6, f"start of next cycle should be at low, got {v_next}"
    # Bounds: always in [low_frac*target, target]
    for ep in range(3 * cycle_len):
        v = cyclical_kl_weight(ep, target, cycle_len, low_frac)
        assert target * low_frac - 1e-6 <= v <= target + 1e-6, (
            f"ep {ep} gave {v}, out of [{target*low_frac}, {target}]"
        )
    print(f"  [OK] cycle_len={cycle_len}, low_frac={low_frac}, target={target}")
    print(f"       low={target*low_frac:.4f}, peak~{expected_peak:.4f}, period={cycle_len}ep")


def test_C_dispatch() -> None:
    """C) get_kl_weight respects config['kl_strategy']."""
    print("\n[Test C] get_kl_weight dispatch")

    # Strategy: linear
    cfg_linear = {
        "kl_weight": 0.3,
        "kl_strategy": "linear",
        "kl_warmup_epochs": 10,
    }
    w_lin, strat = get_kl_weight(20, cfg_linear)
    assert strat == "linear", f"expected 'linear', got {strat}"
    assert abs(w_lin - 0.3) < 1e-6, f"epoch 20 (post-warmup) should be 0.3, got {w_lin}"

    # Strategy: cyclical
    cfg_cyc = {
        "kl_weight": 0.3,
        "kl_strategy": "cyclical",
        "kl_cycle_len": 10,
        "kl_cycle_low_frac": 0.1,
    }
    w_cyc, strat = get_kl_weight(0, cfg_cyc)
    assert strat == "cyclical", f"expected 'cyclical', got {strat}"
    assert abs(w_cyc - 0.03) < 1e-6, f"cycle start should be low, got {w_cyc}"

    # Default: linear, with default kl_warmup_epochs=30 -> w(0) = 0.3 * 1/30
    cfg_def = {"kl_weight": 0.3}
    w_def, strat = get_kl_weight(0, cfg_def)
    assert strat == "linear", f"default strategy should be 'linear', got {strat}"
    assert abs(w_def - 0.3 / 30) < 1e-6, f"epoch 0 default linear should be 0.3/30, got {w_def}"
    print(f"  [OK] linear: w(20)={w_lin} (target), w(0)={w_def} (warmup ramp, default warmup=30)")
    print(f"       cyclical: w(0)={w_cyc} (low), peaks at 0.3 every {cfg_cyc['kl_cycle_len']}ep")


def main() -> None:
    print("=" * 60)
    print("KL Schedule Tests (cyclical vs linear)")
    print("=" * 60)
    tests = [
        ("A: linear_kl_weight", test_A_linear_kl),
        ("B: cyclical_kl_weight", test_B_cyclical_kl),
        ("C: get_kl_weight dispatch", test_C_dispatch),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            raise
    print("\n" + "=" * 60)
    print(f"All {len(tests)} tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
