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

