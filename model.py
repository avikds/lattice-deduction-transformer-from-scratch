"""
Lattice Deduction Transformer from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - peer_mask
import torch

def sudoku_units(n, box_rows, box_cols):
    # The Sudoku must be partitionable into rectangular boxes.
    assert n % box_rows == 0, "n must be divisible by box_rows"
    assert n % box_cols == 0, "n must be divisible by box_cols"
    assert box_rows * box_cols <= n, "Each box cannot contain more than n cells"

    units = []

    # Rows: n units, each containing n cells.
    for r in range(n):
        units.append([(r, c) for c in range(n)])

    # Columns: n units, each containing n cells.
    for c in range(n):
        units.append([(r, c) for r in range(n)])

    # Boxes in row-major order.
    boxes_per_row = n // box_cols
    boxes_per_col = n // box_rows

    for br in range(boxes_per_col):
        for bc in range(boxes_per_row):
            box = []
            for r in range(br * box_rows, (br + 1) * box_rows):
                for c in range(bc * box_cols, (bc + 1) * box_cols):
                    box.append((r, c))
            units.append(box)

    return units


def peer_mask(n, box_rows, box_cols):
    # Create the full list of Sudoku units.
    units = sudoku_units(n, box_rows, box_cols)

    # peers[r, c, r2, c2] indicates whether (r2, c2)
    # is a different cell sharing a row, column, or box with (r, c).
    peers = torch.zeros((n, n, n, n), dtype=torch.bool)

    for unit in units:
        for r, c in unit:
            for r2, c2 in unit:
                # A cell is not its own peer.
                if (r, c) != (r2, c2):
                    peers[r, c, r2, c2] = True

    return peers

# Step 2 - encode_puzzle
import torch.nn.functional as F

def encode_puzzle(grid):
    # grid has shape (..., n, n), with 0 representing an empty cell.
    # The output has shape (..., n, n, n), where the last axis
    # represents candidate digits 1..n.
    n = grid.shape[-1]

    # Candidate digit indices: 0..n-1, corresponding to digits 1..n.
    digits = torch.arange(n, device=grid.device)

    # For a given cell:
    #   - grid == 0  -> every digit remains a candidate
    #   - grid != 0  -> only the given digit remains a candidate
    candidates = (grid.unsqueeze(-1) == 0) | (
        grid.unsqueeze(-1) - 1 == digits
    )

    return candidates


def onehot_grid(grid):
    # grid is assumed to be fully filled, with digits 1..n.
    # Convert each digit into a one-hot vector along a new final axis.
    n = grid.shape[-1]

    return F.one_hot(grid.long() - 1, num_classes=n).bool()


def alive_counts(x):
    # Count the number of surviving candidates for each cell.
    # (..., n, n, n) -> (..., n, n)
    return x.sum(dim=-1)


def is_solved(x):
    # A lattice state is solved exactly when every cell
    # has one and only one remaining candidate.
    counts = alive_counts(x)
    return (counts == 1).all(dim=(-2, -1))


def is_bottom(x):
    # The state is at the bottom of the lattice when at least
    # one cell has no remaining candidates.
    counts = alive_counts(x)
    return (counts == 0).any(dim=(-2, -1))


def decode(x):
    # Return the decided digit for each cell.
    # Cells with exactly one candidate are decoded to 1..n;
    # undecided cells are returned as 0.
    counts = alive_counts(x)
    decided = counts == 1

    # argmax gives the candidate index 0..n-1.
    digits = x.long().argmax(dim=-1) + 1

    # Keep only decided cells; encode all other cells as 0.
    return torch.where(decided, digits, torch.zeros_like(digits))

# Step 3 - alpha_target
def consistent(y, x):
    # y and x have broadcastable shapes ending in (n, n, n).
    # A one-hot solution y is consistent with x when every candidate
    # bit that is alive in y is also alive in x.
    #
    # (~y) | x is True everywhere except where y is True and x is False.
    # Reduce over the three lattice axes to obtain one boolean per
    # leading/batch position.
    return ((~y) | x).all(dim=(-3, -2, -1))


def alpha_target(x, Y, prev=None):
    # x:    (B, n, n, n)          current lattice states
    # Y:    (B, K, n, n, n)       one-hot candidate states for solutions
    # prev: (B, n, n, n), optional previous non-empty target
    
    # Check which solutions are consistent with each state.
    # x is expanded along the solution axis so it broadcasts against Y.
    cons = consistent(Y, x.unsqueeze(1))  # (B, K)

    # A row is dead when none of its K solutions is consistent.
    dead = ~cons.any(dim=1)  # (B,)

    # Keep only the solutions that are consistent with each state.
    # Then OR across the consistent solutions.
    consistent_solutions = Y & cons.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    solution_union = consistent_solutions.any(dim=1)  # (B, n, n, n)

    # The target is restricted to candidates that are both alive in x
    # and alive in at least one consistent solution.
    target = x & solution_union

    # When no solution is consistent with a row, optionally fall back
    # to the previous non-empty target of that chain.
    if prev is not None:
        fallback = x & prev
        target = torch.where(
            dead.view(-1, 1, 1, 1),
            fallback,
            target
        )

    return target, dead

# Step 4 - propagate_singles
def naked_singles(x, peers):
    # Work on a copy so the input state is not modified in-place.
    out = x.clone()

    # A cell is a naked single when exactly one candidate remains.
    counts = x.sum(dim=-1)
    single_cells = counts == 1

    # Get the candidate digit index for each cell. For cells that are not
    # singles, the value is irrelevant because we mask them below.
    single_digit = x.long().argmax(dim=-1)

    # For every decided cell, remove its digit from all of its peers.
    n = x.shape[0]
    for r in range(n):
        for c in range(n):
            if single_cells[r, c]:
                d = int(single_digit[r, c])
                out[:, :, d] = out[:, :, d] & ~peers[r, c]

    return out


def hidden_singles(x, units):
    # Work on a copy so the input state is not modified in-place.
    out = x.clone()

    n = x.shape[0]

    # For each unit and each digit, count how many cells can still
    # contain that digit. If exactly one cell can contain it, that
    # cell must be pinned to that digit.
    for unit in units:
        for d in range(n):
            possible_cells = [
                (r, c) for r, c in unit if x[r, c, d]
            ]

            if len(possible_cells) == 1:
                r, c = possible_cells[0]

                # Pin the cell to this unique digit.
                out[r, c, :] = False
                out[r, c, d] = True

    return out


def propagate_singles(x, units, peers):
    # Alternate naked and hidden singles until reaching a fixed point
    # or until a contradiction creates an empty cell.
    out = x.clone()

    while True:
        # Stop immediately if the state has already reached the bottom.
        if (out.sum(dim=-1) == 0).any():
            return out

        previous = out.clone()

        # First propagate all currently decided digits to their peers.
        out = naked_singles(out, peers)

        # Then apply hidden-single deductions.
        out = hidden_singles(out, units)

        # Stop if the new state contains an empty cell.
        if (out.sum(dim=-1) == 0).any():
            return out

        # If neither rule changed anything, the fixed point has been reached.
        if torch.equal(out, previous):
            return out

# Step 5 - all_solutions
import numpy as np

def all_solutions(n, box_rows, box_cols):
    # Generate every valid completed Sudoku grid using backtracking.
    # Cells are visited in row-major order and digits are tried in
    # increasing order, so the resulting solutions are in search order.
    if n % box_rows != 0 or n % box_cols != 0:
        raise ValueError("n must be divisible by both box_rows and box_cols")
    if box_rows * box_cols != n:
        raise ValueError("box_rows * box_cols must equal n")

    grid = np.zeros((n, n), dtype=np.int64)

    # Track digits already used in each row, column, and box.
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]

    boxes_per_row = n // box_cols
    boxes_per_col = n // box_rows
    box_used = [set() for _ in range(boxes_per_row * boxes_per_col)]

    solutions = []

    def box_index(r, c):
        return (r // box_rows) * boxes_per_row + (c // box_cols)

    def backtrack(pos):
        # All cells have been filled.
        if pos == n * n:
            solutions.append(grid.copy())
            return

        # Row-major traversal.
        r, c = divmod(pos, n)
        b = box_index(r, c)

        # Digits are tried in increasing order: 1..n.
        for d in range(1, n + 1):
            if d in row_used[r] or d in col_used[c] or d in box_used[b]:
                continue

            # Place the digit.
            grid[r, c] = d
            row_used[r].add(d)
            col_used[c].add(d)
            box_used[b].add(d)

            # Continue with the next cell.
            backtrack(pos + 1)

            # Undo the placement.
            grid[r, c] = 0
            row_used[r].remove(d)
            col_used[c].remove(d)
            box_used[b].remove(d)

    backtrack(0)

    # Convert the list of grids into the required NumPy array.
    return np.stack(solutions, axis=0).astype(np.int64)


def count_matching(puzzle, solutions):
    # puzzle has shape (n, n), with 0 denoting an empty cell.
    # solutions has shape (S, n, n).
    #
    # A solution matches the puzzle when every non-zero given agrees
    # with the corresponding cell in that solution.

    puzzle = np.asarray(puzzle)
    solutions = np.asarray(solutions)

    if puzzle.ndim != 2:
        raise ValueError("puzzle must have shape (n, n)")
    if solutions.ndim != 3:
        raise ValueError("solutions must have shape (S, n, n)")
    if solutions.shape[1:] != puzzle.shape:
        raise ValueError("puzzle and solutions have incompatible shapes")

    # Only compare positions containing givens.
    givens = puzzle != 0

    if not np.any(givens):
        return int(solutions.shape[0])

    matches = np.all(
        solutions[:, givens] == puzzle[givens],
        axis=1
    )

    return int(matches.sum())

# Step 6 - make_dataset
import random

def make_puzzle(solution, n_givens, solutions, rng):
    # Start from a copy so the original solution is never modified.
    puzzle = np.array(solution, copy=True)

    n = puzzle.shape[0]

    if puzzle.shape != (n, n):
        raise ValueError("solution must have shape (n, n)")

    if not (0 <= n_givens <= n * n):
        raise ValueError("n_givens must be between 0 and n*n")

    # Visit cells in an order determined by rng.shuffle.
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)

    # Try blanking cells one by one. A cell is blanked only when the
    # resulting puzzle still has exactly one solution.
    for r, c in cells:
        # We already have the requested number of givens.
        if np.count_nonzero(puzzle) <= n_givens:
            break

        # Save the current value so it can be restored if necessary.
        value = puzzle[r, c]
        puzzle[r, c] = 0

        # Keep the blank only if exactly one full solution matches.
        if count_matching(puzzle, solutions) != 1:
            puzzle[r, c] = value

    return puzzle


def make_dataset(num, n_givens_range, solutions, seed):
    # Use one shared Random instance so that solution selection, the
    # requested number of givens, and cell shuffling all follow the
    # deterministic random sequence implied by seed.
    rng = random.Random(seed)

    if len(solutions) == 0:
        raise ValueError("solutions must contain at least one solution")

    data = []

    for _ in range(num):
        # Choose the source solution and number of givens using rng.
        solution_index = rng.randrange(len(solutions))
        n_givens = rng.randint(*n_givens_range)

        solution = solutions[solution_index]

        # Construct the puzzle while preserving a unique solution.
        puzzle = make_puzzle(
            solution,
            n_givens,
            solutions,
            rng
        )

        # Convert the NumPy arrays to PyTorch tensors before encoding.
        puzzle_tensor = torch.as_tensor(puzzle, dtype=torch.long)
        solution_tensor = torch.as_tensor(solution, dtype=torch.long)

        # x0 is the lattice state of the puzzle, while sol is the
        # one-hot lattice state of its complete solution.
        x0 = encode_puzzle(puzzle_tensor).bool()
        sol = onehot_grid(solution_tensor).bool()

        data.append((x0, sol))

    return data

# Step 7 - apply_symmetry
def apply_dihedral(x, k):
    # There are 8 elements in the square's dihedral group:
    # 4 rotations and 4 reflected rotations.
    #
    # For k >= 4, first reflect across the vertical axis by flipping
    # the column dimension (-2), then rotate the two grid dimensions
    # (-3, -2) by k mod 4 quarter-turns.
    k = int(k) % 8

    if k >= 4:
        x = x.flip(-2)

    return torch.rot90(x, k=k % 4, dims=(-3, -2))


def inverse_dihedral(k):
    # Find the inverse by applying every candidate transformation to
    # a probe whose grid entries are all distinct.
    k = int(k) % 8

    # A 3 x 3 grid with a singleton candidate axis lets us distinguish
    # every spatial transformation unambiguously.
    probe = torch.arange(9).reshape(3, 3, 1)

    transformed = apply_dihedral(probe, k)

    for candidate in range(8):
        if torch.equal(apply_dihedral(transformed, candidate), probe):
            return candidate

    # This should never be reached because the 8 dihedral elements
    # form a finite group and every element has an inverse.
    raise RuntimeError("No inverse dihedral element found")


def permute_digits(x, perm):
    # perm[d] gives the destination index for source digit index d.
    # Example: out[..., perm[d]] = x[..., d].
    out = torch.empty_like(x)

    for d in range(x.shape[-1]):
        out[..., perm[d]] = x[..., d]

    return out


def random_symmetry(n, generator):
    # Randomly choose one of the 8 dihedral transformations and one
    # permutation of the n digit channels.
    k = int(torch.randint(8, (1,), generator=generator))
    perm = torch.randperm(n, generator=generator)

    return k, perm


def apply_symmetry(x, sym):
    # Apply the digit permutation first, followed by the spatial
    # dihedral transformation.
    k, perm = sym

    x = permute_digits(x, perm)
    x = apply_dihedral(x, k)

    return x


def invert_symmetry(x, sym):
    # Undo the spatial transformation first, then undo the digit
    # permutation.
    k, perm = sym

    x = apply_dihedral(x, inverse_dihedral(k))

    # Construct the inverse permutation. If perm[d] = j, then
    # inverse_perm[j] = d.
    inverse_perm = torch.argsort(perm)

    x = permute_digits(x, inverse_perm)

    return x

# Step 8 - augmented_queue
def augmented_queue(dataset, copies, n, generator):
    # For each puzzle in the original dataset, generate `copies`
    # symmetry-augmented versions. The same symmetry is applied to
    # both the puzzle state and its solution.
    states = []
    solutions = []

    for x0, sol in dataset:
        for _ in range(copies):
            sym = random_symmetry(n, generator)

            states.append(apply_symmetry(x0, sym))
            solutions.append(apply_symmetry(sol, sym))

    # Stack all augmented states into:
    #   qx -> (N, n, n, n)
    #   qy -> (N, n, n, n)
    qx = torch.stack(states, dim=0)
    qy = torch.stack(solutions, dim=0)

    N = qx.shape[0]

    # Add the solution-set axis of size one:
    #   (N, n, n, n) -> (N, 1, n, n, n)
    qy = qy.unsqueeze(1)

    # Shuffle both tensors using exactly the same permutation so that
    # every puzzle remains paired with its corresponding solution set.
    order = torch.randperm(N, generator=generator)

    qx = qx[order]
    qy = qy[order]

    return qx, qy

# Step 9 - LatticeEmbedding
import torch.nn as nn

class LatticeEmbedding(nn.Module):
    def __init__(self, n, V, d):
        super().__init__()

        self.n = n

        # Project the V-dimensional lattice state of each cell
        # into the model dimension d.
        self.inp = nn.Linear(V, d)

        # Learned row and column positional embeddings.
        self.row_emb = nn.Parameter(torch.randn(n, d) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(n, d) * 0.02)

        # Learned CLS token, initialized to zeros.
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        # x: (B, n, n, V)
        # Convert boolean lattice states to floating point before
        # passing them through the linear projection.
        x = x.float()

        # Project each cell independently:
        # (B, n, n, V) -> (B, n, n, d)
        h = self.inp(x)

        # Add learned row and column positional information.
        # row_emb[:, None, :] -> (n, 1, d)
        # col_emb[None, :, :] -> (1, n, d)
        h = h + self.row_emb[:, None, :] + self.col_emb[None, :, :]

        # Flatten the n x n cells in row-major order.
        # (B, n, n, d) -> (B, n*n, d)
        h = h.reshape(x.shape[0], self.n * self.n, -1)

        # Prepend the CLS token to the sequence.
        # Expand only across the batch dimension.
        cls = self.cls.expand(x.shape[0], -1, -1)

        return torch.cat([cls, h], dim=1)

# Step 10 - TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()

        # Pre-normalization before the self-attention sublayer.
        self.ln1 = nn.LayerNorm(d)

        # Multi-head self-attention with batch-first input:
        # (B, sequence_length, d).
        self.attn = nn.MultiheadAttention(
            d,
            n_heads,
            batch_first=True
        )

        # Pre-normalization before the feed-forward sublayer.
        self.ln2 = nn.LayerNorm(d)

        # Position-wise feed-forward network with a 4*d hidden dimension.
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, h):
        # Pre-norm self-attention followed by a residual connection.
        attn_input = self.ln1(h)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input
        )
        h = h + attn_output

        # Pre-norm feed-forward network followed by a residual connection.
        h = h + self.ff(self.ln2(h))

        return h

