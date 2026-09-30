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

