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

# Step 11 - LatticeDeductionTransformer
class LatticeDeductionTransformer(nn.Module):
    def __init__(self, n, V, d, n_heads, n_layers, n_iters, seed=0):
        super().__init__()

        # Make parameter initialization deterministic for the requested seed.
        torch.manual_seed(seed)

        self.n = n
        self.V = V
        self.n_iters = n_iters

        # Lattice input embedding.
        self.embed = LatticeEmbedding(n, V, d)

        # Shared Transformer stack used at every recurrent iteration.
        self.layers = nn.ModuleList(
            [TransformerLayer(d, n_heads) for _ in range(n_layers)]
        )

        # Final normalization and output heads.
        self.norm = nn.LayerNorm(d)
        self.cand_head = nn.Linear(d, V)
        self.cls_head = nn.Linear(d, 1)

    def forward(self, x):
        # Embed the lattice state once; this embedding is re-injected
        # into the recurrent hidden state at every iteration.
        e = self.embed(x)

        # Start the recurrent state at zero with the same shape and dtype
        # as the embedded input.
        h = torch.zeros_like(e)

        b_iters = []
        c_iters = []

        for _ in range(self.n_iters):
            # Re-inject the original input representation.
            h = h + e

            # Pass through the shared Transformer stack.
            for layer in self.layers:
                h = layer(h)

            # Normalize the current hidden state for reading the outputs.
            h_norm = self.norm(h)

            # Exclude the CLS token and reshape the remaining n*n cell
            # tokens back to the Sudoku grid.
            cell_tokens = h_norm[:, 1:, :]
            b = self.cand_head(cell_tokens).reshape(
                x.shape[0], self.n, self.n, self.V
            )

            # Read the conflict logit from the CLS token.
            c = self.cls_head(h_norm[:, 0, :]).squeeze(-1)

            b_iters.append(b)
            c_iters.append(c)

        # Stack iterations along a new leading dimension.
        b_iters = torch.stack(b_iters, dim=0)
        c_iters = torch.stack(c_iters, dim=0)

        return b_iters, c_iters

# Step 12 - asymmetric_bce
def asymmetric_bce(logits, target, w_pos, w_neg):
    # Convert the boolean target to floating point for the loss formula.
    y = target.float()

    # Stable computation of:
    # log(sigmoid(logits)) and log(sigmoid(-logits)).
    loss = -(
        w_pos * y * F.logsigmoid(logits)
        + w_neg * (1.0 - y) * F.logsigmoid(-logits)
    )

    # Mean over all candidate bits.
    return loss.mean()


def conflict_bce(cls_logits, dead):
    # Binary cross-entropy with logits for the conflict classifier.
    return F.binary_cross_entropy_with_logits(
        cls_logits,
        dead.float()
    )


def singleton_ce(logits, target):
    # Select cells whose target contains exactly one alive candidate.
    singleton = target.sum(dim=-1) == 1

    # No singleton cells: return a differentiable zero.
    if not singleton.any():
        return logits.sum() * 0.0

    # Select logits and corresponding one-hot targets for singleton cells.
    selected_logits = logits[singleton]
    selected_target = target[singleton]

    # Convert the boolean target to an integer tensor before argmax,
    # because argmax is not supported directly on bool tensors.
    selected_target = selected_target.long().argmax(dim=-1)

    # Cross-entropy over the digit axis.
    return F.cross_entropy(
        selected_logits,
        selected_target
    )

# Step 13 - ldt_loss
def ldt_loss(b_iters, c_iters, target, dead, cfg):
    # Compute the loss independently at every recurrent iteration,
    # then average the resulting losses over the iteration dimension.
    losses = []

    for b, c in zip(b_iters, c_iters):
        # Candidate prediction loss.
        loss = asymmetric_bce(
            b,
            target,
            cfg["w_pos"],
            cfg["w_neg"]
        )

        # Conflict classification loss.
        loss = loss + cfg["lam_cls"] * conflict_bce(c, dead)

        # Singleton classification loss.
        loss = loss + cfg["lam_ce"] * singleton_ce(b, target)

        losses.append(loss)

    # Average the total loss across all recurrent iterations.
    return torch.stack(losses).mean()

# Step 14 - branch_pin
def threshold_eliminate(x, probs, theta):
    # A candidate survives only when it is already alive in x and
    # its predicted probability reaches the elimination threshold.
    return x & (probs >= theta)


def branch_pin(x, logits, tau, generator):
    # x and logits have shape (B, n, n, V).
    B, n, _, V = x.shape

    # Flatten the grid cells so that each row contains n*n cells.
    x_flat = x.reshape(B, n * n, V)
    logits_flat = logits.reshape(B, n * n, V)

    # Open cells are cells with at least two surviving candidates.
    alive_counts = x_flat.sum(dim=-1)
    open_cells = alive_counts >= 2
    has_open = open_cells.any(dim=1)

    # Each open cell gets weight 1.0, while non-open cells get 0.0.
    # For rows with no open cell, place a dummy weight at index 0 so
    # torch.multinomial always has a valid distribution.
    cell_weights = open_cells.float()
    cell_weights = cell_weights.clone()
    cell_weights[~has_open, 0] = 1.0

    # First random draw: choose a cell uniformly among open cells.
    chosen_cell = torch.multinomial(
        cell_weights,
        num_samples=1,
        generator=generator
    ).squeeze(-1)

    batch_idx = torch.arange(B, device=x.device)

    # Gather the candidate mask and logits of the selected cell.
    selected_alive = x_flat[batch_idx, chosen_cell]
    selected_logits = logits_flat[batch_idx, chosen_cell]

    # Dead candidates receive -inf so they have zero probability.
    masked_logits = selected_logits.masked_fill(
        ~selected_alive,
        float("-inf")
    )

    # Rows without an open cell use all-zero logits. These rows will
    # be returned unchanged, but this keeps the second multinomial
    # draw valid as specified.
    masked_logits = torch.where(
        has_open.unsqueeze(-1),
        masked_logits,
        torch.zeros_like(masked_logits)
    )

    # Second random draw: sample a digit from the selected cell's
    # surviving candidates according to softmax(logits / tau).
    probs = torch.softmax(masked_logits / tau, dim=-1)

    chosen_digit = torch.multinomial(
        probs,
        num_samples=1,
        generator=generator
    ).squeeze(-1)

    # Pin the chosen cell for rows that actually had an open cell.
    out_flat = x_flat.clone()

    active_rows = batch_idx[has_open]
    active_cells = chosen_cell[has_open]
    active_digits = chosen_digit[has_open]

    out_flat[active_rows, active_cells, :] = False
    out_flat[active_rows, active_cells, active_digits] = True

    return out_flat.reshape_as(x)

# Step 15 - solve_step
def solve_step(model, x, cfg, generator, Y=None, prev=None):
    # Run the recurrent transformer.
    b_iters, c_iters = model(x)

    # Use the final iteration for the discrete solver step.
    # Detach because thresholding, conflict detection, and branching
    # are not differentiable.
    b = b_iters[-1].detach()
    c = c_iters[-1].detach()

    # Eliminate candidates whose predicted probability is below threshold.
    probs = torch.sigmoid(b)
    x2 = threshold_eliminate(
        x,
        probs,
        cfg["theta_elim"]
    )

    result = {}

    if Y is not None:
        # Training mode: compute the sound target from the supplied
        # solution set.
        target, dead = alpha_target(
            x,
            Y,
            prev
        )

        # Keep the original model outputs for gradient computation.
        loss = ldt_loss(
            b_iters,
            c_iters,
            target,
            dead,
            cfg
        )

        # A row conflicts when no supplied solution remains consistent
        # with the post-elimination state.
        cons_x2 = consistent(
            Y,
            x2.unsqueeze(1)
        )
        conflict = ~cons_x2.any(dim=1)

        # A state is solved when every cell has exactly one candidate
        # and at least one supplied solution is still consistent.
        solved = is_solved(x2) & ~conflict

        # Number of target bits that were incorrectly eliminated.
        false_elim = int((target & ~x2).sum().item())

        # Number of candidate bits eliminated in total.
        elims = int((x & ~x2).sum().item())

        result.update({
            "loss": loss,
            "target": target,
            "false_elim": false_elim,
            "elims": elims,
        })

    else:
        # In inference mode, conflict is determined by either reaching
        # the lattice bottom or exceeding the CLS conflict threshold.
        conflict = (
            is_bottom(x2)
            | (torch.sigmoid(c) > cfg["theta_cls"])
        )

        # Solved states must not be marked as conflicted.
        solved = is_solved(x2) & ~conflict

    # Branch only on rows that are neither conflicted nor solved.
    active = ~(conflict | solved)

    # Sample a branch for active rows.
    x3 = branch_pin(
        x2,
        b,
        cfg["tau"],
        generator
    )

    # Inactive rows retain x2; active rows receive the branched state.
    x_new = torch.where(
        active.view(-1, 1, 1, 1),
        x3,
        x2
    )

    result.update({
        "x_new": x_new,
        "conflict": conflict,
        "solved": solved,
    })

    return result

# Step 16 - SolvePool
class SolvePool:
    def __init__(self, dataset, pool_size, generator, n, copies=8):
        # Build the symmetry-augmented queue.
        self.qx, self.qy = augmented_queue(
            dataset,
            copies,
            n,
            generator
        )

        self.cursor = 0
        self.g = generator

        # Fill the initial pool from the beginning of the queue.
        init_idx = self._take(pool_size)

        self.x = self.qx[init_idx].clone()
        self.Y = self.qy[init_idx].clone()

        # The initial previous target is the part of x that agrees with
        # the unique solution in Y.
        self.prev = self.x & self.Y[:, 0]

        # Every freshly inserted state starts with age zero.
        self.age = torch.zeros(
            pool_size,
            dtype=torch.long,
            device=self.x.device
        )

    def _take(self, k):
        # Return k consecutive queue positions with wrap-around.
        N = self.qx.shape[0]

        idx = (
            torch.arange(k, device=self.qx.device) + self.cursor
        ) % N

        # Advance the cursor by k positions, retaining wrap-around.
        self.cursor = (self.cursor + k) % N

        return idx

    def sample(self, B):
        # Draw B distinct pool slots. The first B entries of a random
        # permutation give the requested sample without replacement.
        P = self.x.shape[0]

        idx = torch.randperm(
            P,
            generator=self.g,
            device=self.x.device
        )[:B]

        return (
            idx,
            self.x[idx].clone(),
            self.Y[idx].clone(),
            self.prev[idx].clone(),
        )

    def update(self, idx, x_new, target, terminal, max_age):
        # Determine which sampled slots are allowed to continue.
        # A slot continues only when it is non-terminal and has not
        # reached max_age.
        current_age = self.age[idx]

        keep = (~terminal) & (current_age < max_age)
        refill = ~keep

        # Continue eligible slots.
        if keep.any():
            keep_idx = idx[keep]

            self.x[keep_idx] = x_new[keep]

            # Age increases by one for continued chains.
            self.age[keep_idx] = current_age[keep] + 1

            # Replace prev with target only when target is non-empty.
            target_nonempty = target[keep].flatten(1).any(dim=1)

            if target_nonempty.any():
                target_idx = keep_idx[target_nonempty]
                self.prev[target_idx] = target[keep][target_nonempty]

        # Refill every sampled slot that is terminal or too old.
        if refill.any():
            refill_idx = idx[refill]
            num_refill = int(refill.sum().item())

            queue_idx = self._take(num_refill)

            new_x = self.qx[queue_idx]
            new_Y = self.qy[queue_idx]

            self.x[refill_idx] = new_x
            self.Y[refill_idx] = new_Y
            self.prev[refill_idx] = new_x & new_Y[:, 0]
            self.age[refill_idx] = 0

# Step 17 - train_step
import math
import torch.nn as nn

def lr_at(step, steps, peak, warmup):
    # Linear warmup for steps 0 .. warmup-1:
    # step 0     -> peak / warmup
    # step warmup-1 -> peak
    if step < warmup:
        return peak * (step + 1) / warmup

    # Cosine decay starts at step == warmup, where p == 0
    # and therefore the learning rate is exactly peak.
    p = (step - warmup) / max(1, steps - warmup)

    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def train_step(model, opt, pool, cfg, generator):
    # Sample a distinct batch of pool slots.
    idx, x, Y, prev = pool.sample(cfg["batch_size"])

    # Run one solver step in training mode.
    out = solve_step(
        model,
        x,
        cfg,
        generator,
        Y=Y,
        prev=prev
    )

    loss = out["loss"]

    # Backpropagation and one optimizer step.
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # Clip the total gradient norm to 1.0.
    nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    opt.step()

    # A chain terminates when it either conflicts or is solved.
    terminal = out["conflict"] | out["solved"]

    # Update the corresponding pool slots.
    pool.update(
        idx,
        out["x_new"],
        out["target"],
        terminal,
        cfg["max_age"]
    )

    return {
        "loss": float(loss.detach().item()),
        "false_elim": int(out["false_elim"]),
        "elims": int(out["elims"]),
        "solved": int(out["solved"].sum().item()),
        "conflict": int(out["conflict"].sum().item()),
    }

# Step 18 - train_ldt
def train_ldt(model, dataset, steps, cfg, seed, eval_fn=None, eval_at=()):
    # Single generator shared by the solve pool and all stochastic
    # operations during training.
    g = torch.Generator().manual_seed(seed)

    # Create the on-policy solve pool.
    pool = SolvePool(
        dataset,
        cfg["pool_size"],
        g,
        model.n
    )

    # AdamW optimizer with the specified hyperparameters.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    history = []
    evals = {}

    for step in range(steps):
        # Update every parameter group's learning rate according to
        # the warmup + cosine schedule.
        lr = lr_at(
            step,
            steps,
            cfg["lr"],
            cfg["warmup"]
        )

        for group in opt.param_groups:
            group["lr"] = lr

        # Perform one on-policy training step.
        metrics = train_step(
            model,
            opt,
            pool,
            cfg,
            g
        )

        history.append(metrics)

        # `step` is zero-based, while checkpoint evaluation steps are
        # numbered starting from 1.
        s = step + 1

        if eval_fn is not None and s in eval_at:
            evals[s] = eval_fn(model)

    return history, evals

# Step 19 - solve_puzzle
@torch.no_grad()
def solve_puzzle(model, x0, cfg, generator, chains, max_rounds, augment=True):
    # Start with `chains` identical copies of the initial lattice state.
    states = x0.unsqueeze(0).expand(chains, -1, -1, -1).clone()

    forwards = 0

    for round_idx in range(1, max_rounds + 1):
        # Draw and apply one independent symmetry to every chain.
        if augment:
            syms = [
                random_symmetry(model.n, generator)
                for _ in range(chains)
            ]

            augmented_states = [
                apply_symmetry(states[i], syms[i])
                for i in range(chains)
            ]

            batch = torch.stack(augmented_states, dim=0)
        else:
            syms = None
            batch = states

        # Run one model/solver call for all chains in parallel.
        out = solve_step(
            model,
            batch,
            cfg,
            generator
        )

        forwards += 1

        # The returned states are still in the augmented coordinate
        # systems. Undo each chain's symmetry before inspecting/keeping
        # the states.
        new_states = out["x_new"]

        if augment:
            new_states = torch.stack([
                invert_symmetry(new_states[i], syms[i])
                for i in range(chains)
            ], dim=0)

        # If any chain has solved the puzzle, return the first solved
        # chain in the original coordinate system.
        solved = out["solved"]

        if solved.any():
            first_solved = int(torch.nonzero(solved, as_tuple=False)[0, 0])

            return (
                decode(new_states[first_solved]),
                forwards,
                round_idx
            )

        # Restart conflicted chains from a fresh copy of the original
        # puzzle. Non-conflicted chains continue from their new states.
        states = new_states.clone()

        conflict = out["conflict"]

        if conflict.any():
            states[conflict] = x0

    # No chain solved within the allotted rounds.
    return None, forwards, max_rounds

# Step 20 - evaluate
def valid_grid(grid, units):
    # A valid completed Sudoku grid must have shape (n, n), and every
    # unit must contain each digit 1..n exactly once.
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        return False

    n = grid.shape[0]
    required = set(range(1, n + 1))

    for unit in units:
        values = [int(grid[r, c]) for r, c in unit]

        if set(values) != required or len(values) != n:
            return False

    return True


def evaluate(
    model,
    dataset,
    cfg,
    seed,
    units,
    chains=8,
    max_rounds=30,
    augment=True,
    verify=False
):
    # One generator is shared across all solves and all retry attempts.
    g = torch.Generator().manual_seed(seed)

    # Save the original mode so it can be restored afterwards.
    was_training = model.training
    model.eval()

    outcomes = []
    costs = []

    try:
        for x0, sol in dataset:
            total_forwards = 0
            outcome = "abstain"

            # Without verification, perform exactly one solve attempt.
            # With verification, retry invalid returned grids up to
            # 10 total solves for this puzzle.
            max_attempts = 10 if verify else 1

            for _ in range(max_attempts):
                grid, forwards, _ = solve_puzzle(
                    model,
                    x0,
                    cfg,
                    g,
                    chains=chains,
                    max_rounds=max_rounds,
                    augment=augment
                )

                total_forwards += forwards

                # No grid means the solver abstained. With verification
                # enabled, another attempt is allowed because the total
                # attempt budget has not yet been exhausted.
                if grid is None:
                    if not verify:
                        break
                    continue

                # In verification mode, reject any returned grid that is
                # not a valid Sudoku solution and retry.
                if verify and not valid_grid(grid, units):
                    continue

                # A returned, accepted grid is either correct or wrong.
                expected = decode(sol)

                if torch.equal(
                    torch.as_tensor(grid, device=expected.device),
                    expected
                ):
                    outcome = "correct"
                else:
                    outcome = "wrong"

                # A valid returned grid terminates the retry loop.
                break

            outcomes.append(outcome)
            costs.append(total_forwards)

    finally:
        # Restore the model to its original mode.
        if was_training:
            model.train()
        else:
            model.eval()

    # Aggregate evaluation metrics.
    total = len(outcomes)

    correct = outcomes.count("correct")
    wrong = outcomes.count("wrong")
    abstain = outcomes.count("abstain")

    accuracy = correct / total if total else 0.0
    soundness = (correct + abstain) / total if total else 0.0

    if costs:
        p50 = float(np.percentile(costs, 50))
        p90 = float(np.percentile(costs, 90))
    else:
        p50 = 0.0
        p90 = 0.0

    return {
        "accuracy": accuracy,
        "soundness": soundness,
        "p50": p50,
        "p90": p90,
        "wrong": wrong,
        "abstain": abstain,
        "outcomes": outcomes,
        "costs": costs,
    }

# Step 21 - ldt_experiment
import time

def ldt_experiment(
    train_set,
    test_set,
    cfg,
    steps,
    seed,
    eval_at=(300, 450),
    checkpoint_puzzles=30,
    n=4,
    box=(2, 2)
):
    units = sudoku_units(n, box[0], box[1])
    peers = peer_mask(n, box[0], box[1])

    # ---------------------------------------------------------------
    # Line 1: symbolic reference
    # ---------------------------------------------------------------
    symbolic_solved = 0

    for x0, _ in test_set:
        state = propagate_singles(
            x0.clone(),
            units,
            peers
        )

        if bool(is_solved(state)):
            symbolic_solved += 1

    lines = [
        f"symbolic ref: {symbolic_solved}/{len(test_set)} test puzzles"
    ]

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = LatticeDeductionTransformer(
        n,
        n,
        48,
        4,
        2,
        4,
        seed=seed
    )

    checkpoint_set = test_set[:checkpoint_puzzles]

    # Checkpoint evaluation is deduction-only.
    def eval_fn(m):
        return evaluate(
            m,
            checkpoint_set,
            cfg,
            seed=seed,
            units=units,
            chains=1,
            max_rounds=10,
            augment=False,
            verify=False
        )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    start = time.perf_counter()

    history, evals = train_ldt(
        model,
        train_set,
        steps,
        cfg,
        seed,
        eval_fn=eval_fn,
        eval_at=eval_at
    )

    elapsed = time.perf_counter() - start

    recent = history[-100:]
    total_elims = sum(item["elims"] for item in recent)
    total_false_elims = sum(item["false_elim"] for item in recent)

    lines.append(
        f"training: {steps} steps, time={elapsed:.4f}, "
        f"elims={total_elims}, false_elim={total_false_elims}"
    )

    # ---------------------------------------------------------------
    # Line 3: train/test trend at checkpoints and final model
    # ---------------------------------------------------------------
    trend_parts = []

    for s in sorted(evals):
        r = evals[s]
        trend_parts.append(
            f"{s} steps: accuracy={r['accuracy']:.4f}, "
            f"soundness={r['soundness']:.4f}, "
            f"p50={r['p50']:.4f}"
        )

    final_checkpoint = eval_fn(model)

    trend_parts.append(
        f"{steps} steps: accuracy={final_checkpoint['accuracy']:.4f}, "
        f"soundness={final_checkpoint['soundness']:.4f}, "
        f"p50={final_checkpoint['p50']:.4f}"
    )

    lines.append(
        "train/test trend: " + "; ".join(trend_parts)
    )

    # ---------------------------------------------------------------
    # Line 4: deduction-only final evaluation
    # ---------------------------------------------------------------
    deduction = evaluate(
        model,
        test_set,
        cfg,
        seed=seed,
        units=units,
        chains=1,
        max_rounds=10,
        augment=False,
        verify=False
    )

    lines.append(
        f"deduction only: accuracy={deduction['accuracy']:.4f}, "
        f"soundness={deduction['soundness']:.4f}, "
        f"wrong={deduction['wrong']}, "
        f"abstain={deduction['abstain']}, "
        f"p50={deduction['p50']:.4f}, "
        f"p90={deduction['p90']:.4f}"
    )

    # ---------------------------------------------------------------
    # Line 5: parallel search
    # ---------------------------------------------------------------
    parallel = evaluate(
        model,
        test_set,
        cfg,
        seed=seed,
        units=units,
        chains=8,
        max_rounds=30,
        augment=True,
        verify=False
    )

    lines.append(
        f"parallel search: accuracy={parallel['accuracy']:.4f}, "
        f"soundness={parallel['soundness']:.4f}, "
        f"wrong={parallel['wrong']}, "
        f"abstain={parallel['abstain']}, "
        f"p50={parallel['p50']:.4f}, "
        f"p90={parallel['p90']:.4f}"
    )

    # ---------------------------------------------------------------
    # Line 6: parallel search with verification
    # ---------------------------------------------------------------
    verified = evaluate(
        model,
        test_set,
        cfg,
        seed=seed,
        units=units,
        chains=8,
        max_rounds=30,
        augment=True,
        verify=True
    )

    lines.append(
        f"parallel search: accuracy={verified['accuracy']:.4f}, "
        f"soundness={verified['soundness']:.4f}, "
        f"wrong={verified['wrong']}, "
        f"abstain={verified['abstain']}, "
        f"p50={verified['p50']:.4f}, "
        f"p90={verified['p90']:.4f}"
    )

    return lines

