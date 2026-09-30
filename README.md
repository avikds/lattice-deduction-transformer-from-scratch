# Lattice Deduction Transformer from Scratch

The 2026 paper 'Lattice Deduction Transformers', rebuilt in pure PyTorch on 4 by 4 Sudoku. Represent partial knowledge as the grid powerset lattice, implement the abstraction operator that turns known solutions into per-step targets, and write the symbolic solver the model will be measured against. Then build the recurrent transformer that reads a lattice state, unrolls a shared stack with the input re-injected, and emits candidate and conflict logits at every iteration; the asymmetric loss that makes elimination conservative; and the step operator that thresholds, detects conflicts and branches at random. Train it on-policy from a pool of its own partially deduced states, wrapped in Sudoku's symmetries, and run the parallel solve with restarts. The report reproduces the paper's shape of result at toy scale: deduction replaces search as training proceeds, parallel chains lift accuracy, and soundness is what separates a correct answer from a confident wrong one.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** peer_mask
- [x] **2.** encode_puzzle
- [x] **3.** alpha_target
- [x] **4.** propagate_singles
- [x] **5.** all_solutions
- [x] **6.** make_dataset
- [x] **7.** apply_symmetry
- [x] **8.** augmented_queue
- [x] **9.** LatticeEmbedding
- [x] **10.** TransformerLayer
- [x] **11.** LatticeDeductionTransformer
- [x] **12.** asymmetric_bce
- [x] **13.** ldt_loss
- [x] **14.** branch_pin
- [x] **15.** solve_step
- [x] **16.** SolvePool
- [x] **17.** train_step
- [x] **18.** train_ldt
- [x] **19.** solve_puzzle
- [x] **20.** evaluate
- [x] **21.** ldt_experiment

## Results

```
288 complete 4x4 grids; 300 training puzzles and 100 test puzzles with 5 to 8 givens

1. The symbolic reference
   symbolic ref: 100/100 test puzzles

2. On-policy training with the asymmetric loss
   training: 800 steps, time=24.9898, elims=96723, false_elim=18
   Eliminations rise as the model learns; unsound ones are the soundness gauge.

3. Train/test compute trade-off
   train/test trend: 300 steps: accuracy=0.0000, soundness=0.5667, p50=10.0000; 500 steps: accuracy=0.6000, soundness=0.6667, p50=7.0000; 800 steps: accuracy=0.9667, soundness=0.9667, p50=2.0000
   Better deduction means fewer forward passes per puzzle: search shrinks as training grows.

4. Deduction, search, and soundness
   deduction only: accuracy=0.9200, soundness=0.9200, wrong=8, abstain=0, p50=2.0000, p90=3.0000
   parallel search: accuracy=1.0000, soundness=1.0000, wrong=0, abstain=0, p50=1.0000, p90=2.0000
   parallel search: accuracy=1.0000, soundness=1.0000, wrong=0, abstain=0, p50=1.0000, p90=2.0000
   Parallel chains under fresh symmetries lift accuracy; wrong answers are undetected conflicts,
   which the paper removes with more training and this toy removes with a rule verifier.
```
