"""
Lattice Deduction Transformer from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Lattice Deduction Transformer from Scratch (Davis, Haller, Alfarano, Santolucito, 2026), on 4 by 4 Sudoku.

Story: enumerate the 288 solutions and generate unique puzzles; check that the
classical singles operator solves them all; train a recurrent lattice transformer
on-policy inside the solve loop with the asymmetric loss and the alpha-operator
targets; then measure the paper's three effects at toy scale: more training means
less search, parallel symmetric chains lift accuracy, and soundness is what a
rule verifier restores when the conflict head is not yet reliable.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    solutions = all_solutions(4, 2, 2)
    train_set = make_dataset(300, (5, 8), solutions, seed=1)
    test_set = make_dataset(100, (5, 8), solutions, seed=2)
    print(f"{len(solutions)} complete 4x4 grids; {len(train_set)} training puzzles and {len(test_set)} test puzzles with 5 to 8 givens")

    # paper values except lam_ce (0.2 -> 2.0) and lr (3e-3 -> 5e-3), which shorten the initial plateau at toy scale
    cfg = dict(w_pos=4.0, w_neg=0.5, lam_cls=0.1, lam_ce=2.0, theta_elim=0.1, theta_cls=0.6, tau=1.5,
               batch_size=64, pool_size=64, max_age=100, warmup=50, lr=5e-3)
    lines = ldt_experiment(train_set, test_set, cfg, steps=800, seed=0, eval_at=(300, 500), checkpoint_puzzles=30)

    print("\n1. The symbolic reference")
    print("   " + lines[0])
    print("\n2. On-policy training with the asymmetric loss")
    print("   " + lines[1])
    print("   Eliminations rise as the model learns; unsound ones are the soundness gauge.")
    print("\n3. Train/test compute trade-off")
    print("   " + lines[2])
    print("   Better deduction means fewer forward passes per puzzle: search shrinks as training grows.")
    print("\n4. Deduction, search, and soundness")
    for line in lines[3:]:
        print("   " + line)
    print("   Parallel chains under fresh symmetries lift accuracy; wrong answers are undetected conflicts,")
    print("   which the paper removes with more training and this toy removes with a rule verifier.")


if __name__ == "__main__":
    main()

