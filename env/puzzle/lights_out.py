"""GF(2) machinery for the OGBench Puzzle ("Lights Out") board."""

import itertools

import numpy as np

NEIGHBOR_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def press_indices(k, num_rows, num_cols):
    """Flat indices toggled by pressing button `k` (itself + orthogonal neighbours)."""
    i, j = k // num_cols, k % num_cols
    out = []
    for di, dj in NEIGHBOR_OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < num_rows and 0 <= nj < num_cols:
            out.append(ni * num_cols + nj)
    return out


def apply_press(state, k, num_rows, num_cols):
    """Return a copy of `state` with button `k` pressed."""
    out = np.asarray(state).copy()
    for idx in press_indices(k, num_rows, num_cols):
        out[idx] ^= 1
    return out


def apply_presses(state, ks, num_rows, num_cols):
    """Return a copy of `state` with every button in `ks` pressed (order is irrelevant)."""
    out = np.asarray(state).copy()
    for k in ks:
        for idx in press_indices(k, num_rows, num_cols):
            out[idx] ^= 1
    return out


def toggle_matrix(num_rows, num_cols):
    """(n, n) uint8 matrix over GF(2); column k = the effect of pressing button k."""
    n = num_rows * num_cols
    m = np.zeros((n, n), dtype=np.uint8)
    for k in range(n):
        for idx in press_indices(k, num_rows, num_cols):
            m[idx, k] ^= 1
    return m


def _rref(mat):
    """Row-reduce over GF(2). Returns (reduced, pivot_columns)."""
    mat = mat.copy()
    rows, cols = mat.shape
    r, pivots = 0, []
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if mat[i, c]), None)
        if pivot is None:
            continue
        mat[[r, pivot]] = mat[[pivot, r]]
        for i in range(rows):
            if i != r and mat[i, c]:
                mat[i] ^= mat[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return mat, pivots


def nullspace(num_rows, num_cols):
    """Basis of the toggle matrix's null space over GF(2) (empty unless the board is 4x4)."""
    m = toggle_matrix(num_rows, num_cols)
    n = m.shape[1]
    reduced, pivots = _rref(m)
    basis = []
    for free in (c for c in range(n) if c not in pivots):
        v = np.zeros(n, dtype=np.uint8)
        v[free] = 1
        for r, c in enumerate(pivots):
            v[c] = reduced[r, free]
        basis.append(v)
    return basis


def solve(init, goal, num_rows, num_cols):
    """One press set taking `init` to `goal`, as a 0/1 vector, or None if unreachable."""
    m = toggle_matrix(num_rows, num_cols)
    n = m.shape[1]
    delta = (np.asarray(init).astype(np.uint8) ^ np.asarray(goal).astype(np.uint8)).reshape(-1, 1)
    reduced, pivots = _rref(np.concatenate([m, delta], axis=1))
    if any(p == n for p in pivots):   # pivot in the augmented column -> inconsistent
        return None
    x = np.zeros(n, dtype=np.uint8)
    for r, c in enumerate(pivots):
        x[c] = reduced[r, n]
    return x


def min_presses(init, goal, num_rows, num_cols):
    """Minimum number of presses from `init` to `goal`, or None if unreachable.

    except 4x4, where it is 16.  Cheap enough to call per evaluation episode.
    """
    x = solve(init, goal, num_rows, num_cols)
    if x is None:
        return None
    basis = nullspace(num_rows, num_cols)
    best = int(x.sum())
    for bits in itertools.product((0, 1), repeat=len(basis)):
        y = x.copy()
        for b, v in zip(bits, basis):
            if b:
                y = y ^ v
        best = min(best, int(y.sum()))
    return best


def is_solvable(init, goal, num_rows, num_cols):
    return solve(init, goal, num_rows, num_cols) is not None
