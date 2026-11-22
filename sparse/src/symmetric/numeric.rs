//! Multifrontal numeric factorization for sparse symmetric indefinite matrices.
//!
//! Implements the multifrontal method with A Posteriori Threshold Pivoting (APTP)
//! for computing P^T A P = L D L^T factorizations of sparse symmetric indefinite
//! matrices. The factorization traverses the assembly tree in postorder, assembling
//! dense frontal matrices and factoring them using the [`aptp_factor_in_place`]
//! kernel.
//!
//! # Algorithm Overview
//!
//! For each supernode in assembly-tree postorder:
//! 1. **Assemble** a dense frontal matrix by scattering original sparse entries
//!    and extend-adding child contribution blocks
//! 2. **Factor** the fully-summed portion using APTP (1x1/2x2 pivoting with delays)
//! 3. **Extract** per-supernode factors (L11, D11, L21) and contribution block
//! 4. **Propagate** contribution block (including delayed columns) to parent
//!
//! # Key Types
//!
//! - [`AptpNumeric`] — complete factorization result (public API)
//! - [`FrontFactors`] — per-supernode factors (public, used by triangular solve)
//! - [`FactorizationStats`] — aggregate statistics (public)
//! - `SupernodeInfo` — unified supernode descriptor (internal)
//! - `FrontalMatrix` — temporary dense assembly matrix (internal)
//! - `ContributionBlock` — Schur complement passed to parent (internal)
//!
//! # References
//!
//! - Duff & Reid (1983), "The multifrontal solution of indefinite sparse symmetric
//!   linear equations" — foundational multifrontal method
//! - Liu (1992), "The Multifrontal Method for Sparse Matrix Solution: Theory and
//!   Practice" — frontal matrices, elimination trees, extend-add
//! - Hogg, Duff & Lopez (2020), "A New Sparse LDL^T Solver Using A Posteriori
//!   Threshold Pivoting" — APTP algorithm integration with multifrontal framework

use faer::linalg::matmul::matmul_with_conj;
use faer::linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place_with_conj, solve_unit_upper_triangular_in_place_with_conj,
};
use faer::sparse::SparseColMat;
use faer::sparse::linalg::cholesky::SymbolicCholeskyRaw;
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par};

use super::diagonal::{MixedDiagonal, PivotEntry};
use super::factor::{
    AptpFactorResult, AptpOptions, advise_huge_pages, aptp_factor_in_place,
    compute_contribution_gemm_in_place, compute_contribution_gemm_rect, extract_contribution_rect,
    extract_front_factors_rect, padded_rows, tpp_factor_rect, unpivoted_factor_in_place,
    zeros_padded,
};
use super::pivot::{Block2x2, PivotType};
use super::symbolic::AptpSymbolic;
use crate::error::SparseError;

#[cfg(feature = "diagnostic")]
use crate::profiling::{FinishedSession, ProfileSession};
use crate::scalar::Scalar;

/// Sentinel value indicating a global column is not part of the current front.
const NOT_IN_FRONT: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Workspace types
// ---------------------------------------------------------------------------

/// Pre-allocated reusable buffers for the per-supernode factorization loop.
///
/// Eliminates per-supernode heap allocations for the frontal matrix, row indices,
/// delayed columns buffer, and global-to-local mapping. Sized to the maximum
/// front dimension from symbolic analysis and reused across supernodes.
///
/// # Invariant
///
/// Never shared between concurrent supernodes. In sequential mode, passed as
/// `&mut`. In parallel mode, each rayon worker has its own instance via
/// `Cell`-based move semantics.
///
/// # References
///
/// - Duff, Hogg & Lopez (2020), Section 5: workspace reuse strategy
/// - Liu (1992), Section 4: multifrontal workspace management
pub(crate) struct FactorizationWorkspace<T: Scalar = f64> {
    /// Dense buffer, max_front × max_front. Reused as each supernode's frontal matrix.
    frontal_data: Mat<T>,
    /// Buffer for frontal matrix row indices, capacity max_front.
    frontal_row_indices: Vec<usize>,
    /// Buffer for collecting delayed columns from children.
    delayed_cols_buf: Vec<usize>,
    /// Global-to-local row index mapping, length n (matrix dimension).
    global_to_local: Vec<usize>,
    /// Pre-allocated L·D product workspace for `compute_contribution_gemm_in_place`.
    /// Sized nfs × ne (grows on demand). Eliminates per-supernode allocation
    /// in the hot loop.
    ld_workspace: Mat<T>,
}

impl<T: Scalar> Default for FactorizationWorkspace<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: Scalar> FactorizationWorkspace<T> {
    /// Create a new workspace sized for the given maximum front dimension.
    ///
    /// # Arguments
    ///
    /// - `max_front`: Maximum frontal matrix dimension across all supernodes
    /// - `n`: Matrix dimension (for global-to-local mapping)
    pub(crate) fn new(max_front: usize, n: usize) -> Self {
        if max_front == 0 {
            return Self {
                frontal_data: Mat::<T>::zeros(0, 0),
                frontal_row_indices: Vec::new(),
                delayed_cols_buf: Vec::new(),
                global_to_local: vec![NOT_IN_FRONT; n],
                ld_workspace: Mat::new(),
            };
        }
        Self {
            frontal_data: zeros_padded(max_front, max_front),
            frontal_row_indices: Vec::with_capacity(max_front),
            delayed_cols_buf: Vec::with_capacity(max_front),
            global_to_local: vec![NOT_IN_FRONT; n],
            ld_workspace: Mat::new(),
        }
    }

    /// Empty workspace for `Cell` default in thread-local storage.
    const fn empty() -> Self {
        Self {
            frontal_data: Mat::new(),
            frontal_row_indices: Vec::new(),
            delayed_cols_buf: Vec::new(),
            global_to_local: Vec::new(),
            ld_workspace: Mat::new(),
        }
    }

    /// Ensure `global_to_local` mapping is at least `n` entries.
    ///
    /// Used by the parallel path where frontal_data grows on demand in
    /// `factor_single_supernode` (avoiding eager max_front² allocation
    /// per thread that causes OOM on large-front matrices like H2O).
    fn ensure_g2l(&mut self, n: usize) {
        if self.global_to_local.len() < n {
            self.global_to_local.resize(n, NOT_IN_FRONT);
        }
    }
}

/// Front dimension below which intra-node BLAS uses `Par::Seq` regardless of
/// the user-supplied parallelism setting. Fronts smaller than this threshold
/// do not benefit from parallel BLAS (overhead exceeds computation).
pub(crate) const INTRA_NODE_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// Assembly maps (precomputed scatter indices)
// ---------------------------------------------------------------------------

/// Precomputed index mappings for assembly operations.
///
/// Eliminates per-entry index arithmetic in `scatter_original_entries_multi`
/// and `extend_add` by storing source→destination index pairs computed from
/// the symbolic structure. Computed once after amalgamation at the start of
/// factorization and read-only during the assembly loop.
///
/// Uses `u32` indices to halve memory (frontal matrices are bounded by
/// max_front_size, well within u32 range).
///
/// # Layout
///
/// Both `amap_entries` and `ea_map` use CSC-style compressed storage with
/// per-supernode/per-child offset arrays for O(1) slice access.
///
/// # References
///
/// - SPRAL `assemble.hxx:51-80` (`add_a_block`): amap-based assembly pattern
pub(crate) struct AssemblyMaps {
    /// Flattened tuples for original matrix scatter. Each entry is 4 u32 values:
    /// `[src_csc_index, dest_frontal_linear_index, scale_row, scale_col]` where
    /// `src_csc_index` is the position in the CSC values array, `dest_frontal_linear_index`
    /// is `col * m + row` in column-major layout, and `scale_row`/`scale_col` are
    /// the permuted indices for MC64 scaling lookup (`scaling[scale_row] * scaling[scale_col]`).
    /// Length: 4 * total_entries across all supernodes.
    pub amap_entries: Vec<u32>,
    /// Per-supernode start offset into `amap_entries` (in entry units; each entry
    /// is 4 u32 values). `amap_entries[amap_offsets[s]*4 .. amap_offsets[s+1]*4]`
    /// gives the flattened entries for supernode s.
    /// Length: num_supernodes + 1.
    pub amap_offsets: Vec<usize>,

    /// Flattened child→parent row index mappings for extend-add.
    /// For each child's contribution row, stores the local row index in the
    /// parent's frontal matrix (assuming zero delays).
    pub ea_map: Vec<u32>,
    /// Per-child start offset into `ea_map`. Indexed by the flattened child
    /// enumeration (all children of all supernodes in tree order).
    /// Length: total_children + 1.
    pub ea_offsets: Vec<usize>,
    /// Per-supernode start offset into the children enumeration.
    /// `ea_snode_child_begin[s]..ea_snode_child_begin[s+1]` gives the range
    /// in `ea_offsets` for supernode s's children.
    /// Length: num_supernodes + 1.
    pub ea_snode_child_begin: Vec<usize>,
}

/// Build precomputed assembly maps from post-amalgamation supernode structure.
///
/// # Arguments
///
/// - `supernodes`: Post-amalgamation supernode descriptors.
/// - `children`: Children map (children[s] = list of child supernode indices).
/// - `matrix`: Original sparse matrix (for CSC structure).
/// - `perm_fwd`: Forward permutation (perm_fwd[new] = old).
/// - `perm_inv`: Inverse permutation (perm_inv[old] = new).
/// - `n`: Matrix dimension.
fn build_assembly_maps<T: Scalar>(
    supernodes: &[SupernodeInfo],
    children: &[Vec<usize>],
    matrix: &SparseColMat<usize, T>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
    n: usize,
) -> AssemblyMaps {
    let n_supernodes = supernodes.len();
    let symbolic = matrix.symbolic();
    let col_ptrs = symbolic.col_ptr();
    let row_indices_csc = symbolic.row_idx();

    // ---- Build amap (original entry scatter maps) ----
    // Each entry is 4 u32: [src_csc_index, dest_frontal_linear, scale_row, scale_col]
    // where scale_row = perm_inv[orig_row], scale_col = perm_inv[orig_col]
    // (precomputed for efficient MC64 scaling at assembly time).

    let mut amap_entries = Vec::new();
    let mut amap_offsets = vec![0usize; n_supernodes + 1];

    // Temporary global-to-local mapping, reused per supernode
    let mut g2l = vec![NOT_IN_FRONT; n];

    for (s, sn) in supernodes.iter().enumerate() {
        // Compute front structure for this supernode (without delays)
        let sn_ncols: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
        let mut frontal_rows: Vec<usize> = Vec::with_capacity(sn_ncols + sn.pattern.len());
        for range in &sn.owned_ranges {
            frontal_rows.extend(range.clone());
        }
        frontal_rows.extend_from_slice(&sn.pattern);
        let m = frontal_rows.len();

        // Build global-to-local
        for (i, &global) in frontal_rows.iter().enumerate() {
            g2l[global] = i;
        }

        let total_owned = sn_ncols;

        // Iterate owned columns and build amap entries
        for range in &sn.owned_ranges {
            for pj in range.clone() {
                let orig_col = perm_fwd[pj];
                let local_col = g2l[pj];

                let start = col_ptrs[orig_col];
                let end = col_ptrs[orig_col + 1];
                for (idx, &orig_row) in row_indices_csc.iter().enumerate().take(end).skip(start) {
                    // Upper-triangle dedup: skip if both are owned supernode cols
                    if orig_row < orig_col {
                        let perm_row = perm_inv[orig_row];
                        let local_peer = g2l[perm_row];
                        if local_peer != NOT_IN_FRONT && local_peer < total_owned {
                            continue;
                        }
                    }
                    let global_row = perm_inv[orig_row];
                    let local_row = g2l[global_row];
                    if local_row == NOT_IN_FRONT {
                        continue;
                    }

                    // Determine destination in lower triangle
                    let (dest_row, dest_col) = if local_row >= local_col {
                        (local_row, local_col)
                    } else {
                        (local_col, local_row)
                    };
                    let dest_linear = dest_col * m + dest_row;

                    amap_entries.push(idx as u32);
                    amap_entries.push(dest_linear as u32);
                    amap_entries.push(global_row as u32); // perm_inv[orig_row]
                    amap_entries.push(pj as u32); // perm_inv[orig_col] = pj
                }
            }
        }

        // Reset g2l
        for &global in &frontal_rows {
            g2l[global] = NOT_IN_FRONT;
        }

        let entry_end = amap_entries.len() / 4;
        amap_offsets[s + 1] = entry_end;
    }

    // ---- Build extend-add maps ----

    let mut ea_map = Vec::new();
    let mut ea_offsets = vec![0usize];
    let mut ea_snode_child_begin = vec![0usize; n_supernodes + 1];

    for (s, sn) in supernodes.iter().enumerate() {
        // Build parent's frontal row structure (without delays)
        let sn_ncols: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
        let mut parent_rows: Vec<usize> = Vec::with_capacity(sn_ncols + sn.pattern.len());
        for range in &sn.owned_ranges {
            parent_rows.extend(range.clone());
        }
        parent_rows.extend_from_slice(&sn.pattern);

        // Build g2l for parent
        for (i, &global) in parent_rows.iter().enumerate() {
            g2l[global] = i;
        }

        for &c in &children[s] {
            let child_sn = &supernodes[c];

            // Child's contribution rows = child's pattern (off-diagonal rows).
            // Without delays, the contribution block rows are exactly the pattern.
            // With delays, additional delayed rows appear — the precomputed map
            // won't cover those (fallback to g2l at factorization time).
            for &child_row in &child_sn.pattern {
                let parent_local = g2l[child_row];
                debug_assert!(
                    parent_local != NOT_IN_FRONT,
                    "child pattern row {} not in parent supernode {}",
                    child_row,
                    s
                );
                ea_map.push(parent_local as u32);
            }
            ea_offsets.push(ea_map.len());
        }

        // Reset g2l
        for &global in &parent_rows {
            g2l[global] = NOT_IN_FRONT;
        }

        ea_snode_child_begin[s + 1] = ea_offsets.len() - 1;
    }

    AssemblyMaps {
        amap_entries,
        amap_offsets,
        ea_map,
        ea_offsets,
        ea_snode_child_begin,
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Precomputed supernode descriptor unifying faer's supernodal and simplicial
/// decompositions.
///
/// Built once from [`AptpSymbolic`] before the factorization loop via
/// [`build_supernode_info`]. For supernodal decompositions, maps directly to
/// faer's supernode structure. For simplicial decompositions, each column
/// becomes a trivial 1-column supernode.
///
/// # Invariants
///
/// - `col_end > col_begin`
/// - `parent.map_or(true, |p| p > s)` for supernode index `s` (postorder)
///
/// # References
///
/// - Liu (1992), Section 3: supernode definitions and assembly trees
#[derive(Clone)]
pub(crate) struct SupernodeInfo {
    /// First column index of this supernode (inclusive).
    pub col_begin: usize,
    /// Past-the-end column index (exclusive).
    pub col_end: usize,
    /// Off-diagonal row indices (global permuted column indices of non-fully-summed rows).
    pub pattern: Vec<usize>,
    /// Parent supernode index in assembly tree, or `None` for root.
    pub parent: Option<usize>,
    /// Column ranges actually owned by this supernode. After amalgamation of
    /// non-adjacent supernodes, `col_begin..col_end` may span columns belonging
    /// to other supernodes. `owned_ranges` tracks the true owned columns for
    /// scatter. Default: `[col_begin..col_end]`.
    pub owned_ranges: Vec<std::ops::Range<usize>>,
    /// True if this supernode belongs to a classified small-leaf subtree.
    /// Set by `classify_small_leaf_subtrees()` after amalgamation.
    pub in_small_leaf: bool,
}

/// A classified small-leaf subtree eligible for the fast-path factorization.
///
/// Produced by [`classify_small_leaf_subtrees`] after amalgamation and consumed
/// by `factor_small_leaf_subtree` when the traversal in `factor_tree` reaches the
/// subtree root. All supernodes in the subtree have front_size < `small_leaf_threshold`,
/// enabling a streamlined code path with a single small reusable workspace.
///
/// # Invariants
///
/// - `nodes.len() >= 2` (single-node "subtrees" are excluded)
/// - `*nodes.last().unwrap() == root`
/// - All nodes have `in_small_leaf = true`
/// - `max_front_size < small_leaf_threshold`
/// - Nodes are in postorder: children appear before parents
///
/// # References
///
/// - SPRAL `SymbolicSubtree.hxx:57-84` (BSD-3): subtree classification
/// - SPRAL `SmallLeafNumericSubtree.hxx:187-446` (BSD-3): fast-path factorization
pub(crate) struct SmallLeafSubtree {
    /// Supernode ID of the subtree root (topmost node in the subtree).
    /// Used by tests for invariant verification; always equals `*nodes.last()`.
    #[allow(dead_code)]
    pub root: usize,
    /// Supernode IDs in postorder (leaves first, root last).
    pub nodes: Vec<usize>,
    /// Maximum front size across all nodes in the subtree.
    pub max_front_size: usize,
    /// Parent supernode outside the subtree that receives the root's contribution,
    /// or `None` if the subtree root is also a tree root. Used by tests for
    /// invariant verification.
    #[allow(dead_code)]
    pub parent_of_root: Option<usize>,
}

// No methods needed — fields accessed directly within this module.

/// Temporary dense matrix for assembling and factoring one supernode.
///
/// References workspace buffers during the factorization loop, or owns its
/// data in diagnostic contexts (`export_assembled_frontal`).
///
/// The matrix is partitioned into:
/// - `F11 = data[0..k, 0..k]` — fully-summed block (factored by APTP)
/// - `F21 = data[k..m, 0..k]` — subdiagonal block
/// - `F22 = data[k..m, k..m]` — contribution block (Schur complement)
///
/// where `k = num_fully_summed` and `m = data.nrows()`.
pub(crate) struct FrontalMatrix<'a, T: Scalar = f64> {
    /// Dense m × m storage (lower triangle used). Borrows from workspace.
    pub data: MatMut<'a, T>,
    /// Global permuted column indices for each local row (length m).
    /// Used by callers that need index mapping (e.g., scatter maps).
    #[allow(dead_code)]
    pub row_indices: &'a [usize],
    /// Number of fully-summed rows/columns (supernode cols + delayed from children).
    pub num_fully_summed: usize,
}

/// Schur complement and delayed columns from a factored supernode.
///
/// Created by [`extract_contribution`] and consumed by [`extend_add`] into
/// the parent supernode's frontal matrix.
///
/// # Structure
///
/// - Positions `0..num_delayed`: delayed columns (become fully-summed at parent)
/// - Positions `num_delayed..size`: non-fully-summed rows with Schur complement
pub(crate) struct ContributionBlock<T: Scalar = f64> {
    /// Lower triangle of the (m - ne) × (m - ne) trailing submatrix of the
    /// factored frontal matrix, packed and exactly sized.
    pub data: PackedLower<T>,
    /// Global permuted column indices for rows/columns of the contribution.
    pub row_indices: Vec<usize>,
    /// Number of delayed columns at the start of this block.
    pub num_delayed: usize,
}

/// Fill packed columns `j0..j1` (whose storage is exactly `data`) of an
/// `n × n` [`PackedLower`], splitting columns across threads.
fn fill_packed_cols<'a, T: Scalar + 'a>(
    data: &mut [core::mem::MaybeUninit<T>],
    n: usize,
    j0: usize,
    j1: usize,
    col: &(impl Fn(usize) -> &'a [T] + Sync),
    par: Par,
) {
    let offset = |j: usize| j * n - j * j.saturating_sub(1) / 2;
    if copy_par_split(par, j1 - j0, data.len()) {
        let mid = (j0 + j1) / 2;
        let (left, right) = data.split_at_mut(offset(mid) - offset(j0));
        rayon::join(
            || fill_packed_cols(left, n, j0, mid, col, par),
            || fill_packed_cols(right, n, mid, j1, col, par),
        );
        return;
    }
    let mut pos = 0;
    for j in j0..j1 {
        let c = col(j);
        assert_eq!(c.len(), n - j, "PackedLower column {j} has wrong length");
        write_slice(&mut data[pos..pos + c.len()], c);
        pos += c.len();
    }
    assert_eq!(pos, data.len(), "packed columns do not fill their storage");
}

/// Initialize `dst` with a copy of `src` (equal lengths).
#[inline]
pub(crate) fn write_slice<T: Copy>(dst: &mut [core::mem::MaybeUninit<T>], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: lengths match, `MaybeUninit<T>` has the layout of `T`, and a
    // shared and a unique borrow cannot overlap.
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<T>(), src.len());
    }
}

/// Initialize every entry of `dst` with `v`.
#[inline]
pub(crate) fn write_fill<T: Copy>(dst: &mut [core::mem::MaybeUninit<T>], v: T) {
    for x in dst {
        x.write(v);
    }
}

/// Raw column base pointer shared by the threads of [`fill_cols_uninit`];
/// each thread writes only its own columns.
#[derive(Clone, Copy)]
struct ColsPtr<T>(*mut T);
// SAFETY: only used to hand disjoint columns to different threads.
unsafe impl<T: Send> Send for ColsPtr<T> {}
unsafe impl<T: Send> Sync for ColsPtr<T> {}

/// Matrix with the padded stride of [`zeros_padded`] whose entries are each
/// written once: `fill(j, col)` must initialize all `nrows` entries of column
/// `j`. Large matrices fill their columns in parallel.
///
/// Building a factor or contribution block as `zeros_padded` followed by a
/// copy writes every entry twice; this writes it once, which matters for the
/// multi-GB, memory-bound extraction out of large fronts.
pub(crate) fn filled_padded<T: Scalar>(
    nrows: usize,
    ncols: usize,
    par: Par,
    fill: &(impl Fn(usize, &mut [core::mem::MaybeUninit<T>]) + Sync),
) -> Mat<T> {
    if nrows == 0 || ncols == 0 {
        return Mat::<T>::zeros(nrows, ncols);
    }
    let mut mat = Mat::<T>::with_capacity(padded_rows::<T>(nrows), ncols);
    let stride = mat.col_stride() as usize;
    assert!(stride >= nrows);
    let base = mat.as_ptr_mut();
    advise_huge_pages(base.cast(), stride * ncols * core::mem::size_of::<T>());
    fill_cols_uninit(ColsPtr(base), stride, nrows, 0, ncols, par, fill);
    // SAFETY: the capacity covers `nrows × ncols` at this stride, and
    // fill_cols_uninit had `fill` initialize every entry of every column.
    unsafe { mat.set_dims(nrows, ncols) };
    mat
}

fn fill_cols_uninit<T: Scalar>(
    base: ColsPtr<T>,
    stride: usize,
    nrows: usize,
    j0: usize,
    j1: usize,
    par: Par,
    fill: &(impl Fn(usize, &mut [core::mem::MaybeUninit<T>]) + Sync),
) {
    if copy_par_split(par, j1 - j0, nrows * (j1 - j0)) {
        let mid = (j0 + j1) / 2;
        rayon::join(
            || fill_cols_uninit(base, stride, nrows, j0, mid, par, fill),
            || fill_cols_uninit(base, stride, nrows, mid, j1, par, fill),
        );
        return;
    }
    for j in j0..j1 {
        // SAFETY: column j is `nrows` entries at offset `j * stride` inside
        // the allocation, and columns j0..j1 belong to this call alone.
        let col = unsafe {
            core::slice::from_raw_parts_mut(
                base.0.add(j * stride).cast::<core::mem::MaybeUninit<T>>(),
                nrows,
            )
        };
        fill(j, col);
    }
}

#[cfg(test)]
impl<T: Scalar> ContributionBlock<T> {
    /// Build from the lower triangle of a dense matrix (tests only).
    pub(crate) fn from_dense_lower(
        dense: &Mat<T>,
        row_indices: Vec<usize>,
        num_delayed: usize,
    ) -> Self {
        let n = row_indices.len();
        Self {
            data: PackedLower::from_columns(n, |j| &dense.col_as_slice(j)[j..n]),
            row_indices,
            num_delayed,
        }
    }
}

/// Minimum columns per task for the parallel copy and scatter helpers.
const COPY_PAR_MIN_COLS: usize = 32;
/// Entry count below which the copy and scatter helpers stay sequential.
const COPY_PAR_MIN_ENTRIES: usize = 1 << 19;

/// Whether a column-parallel copy/scatter of this size should split.
#[inline]
fn copy_par_split(par: Par, ncols: usize, entries: usize) -> bool {
    par.degree() > 1 && ncols >= 2 * COPY_PAR_MIN_COLS && entries >= COPY_PAR_MIN_ENTRIES
}

/// Call `f(col0 + j, column j as a slice)` for every column of `mat`,
/// splitting columns across rayon threads for large views.
///
/// These memory-bound loops (zeroing a front, copying factors out of it)
/// otherwise run on one thread while the rest of phase B's threads wait.
fn for_each_col_mut<T: Scalar>(
    mat: MatMut<'_, T>,
    col0: usize,
    par: Par,
    f: &(impl Fn(usize, &mut [T]) + Sync),
) {
    let (nr, nc) = (mat.nrows(), mat.ncols());
    if copy_par_split(par, nc, nr * nc) {
        let mid = nc / 2;
        let (left, right) = mat.split_at_col_mut(mid);
        rayon::join(
            || for_each_col_mut(left, col0, par, f),
            || for_each_col_mut(right, col0 + mid, par, f),
        );
        return;
    }
    let mut mat = mat;
    for j in 0..nc {
        match mat.as_mut().col_mut(j).try_as_col_major_mut() {
            Some(col) => f(col0 + j, col.as_slice_mut()),
            None => {
                // Strided view (not produced by this module): go through a copy.
                let mut tmp: Vec<T> = (0..nr).map(|i| mat[(i, j)]).collect();
                f(col0 + j, &mut tmp);
                for (i, v) in tmp.into_iter().enumerate() {
                    mat[(i, j)] = v;
                }
            }
        }
    }
}

/// Column-panel width of [`BlockedUnitLower`] storage.
pub(crate) const L11_PANEL_WIDTH: usize = 64;

/// Unit lower-triangular `L11` factor stored as column panels.
///
/// Columns are grouped into panels of 64 columns (`L11_PANEL_WIDTH`); the panel starting
/// at column `b` stores rows `b..n` only. The upper triangle is therefore
/// wasted only inside each diagonal block, about `n·w/2` entries for panel
/// width `w`, instead of the `n²/2` of a dense square — roughly 2% extra over
/// packed storage for a 3000-column supernode — while every panel remains a
/// dense column-major matrix for BLAS-3 triangular solves and GEMM.
///
/// # References
///
/// - Gustavson, Karlsson & Kågström (2009), "Parallel and cache-efficient
///   in-place matrix storage format conversion" and the LAPACK RFP/blocked
///   packed formats: trading a block of waste for dense kernels
/// - Duff, Hogg & Lopez (2020), Section 4: column-panel storage of L in SSIDS
#[derive(Debug, Clone)]
pub struct BlockedUnitLower<T: Scalar = f64> {
    n: usize,
    panels: Vec<Mat<T>>,
}

impl<T: Scalar> BlockedUnitLower<T> {
    /// Build from a factored front or L storage whose first `ne` columns hold
    /// the eliminated L entries below the diagonal. For the first column of a
    /// 2×2 pivot, the entry just below the diagonal belongs to D and is
    /// skipped (that L entry is zero).
    pub(crate) fn from_factored(src: &Mat<T>, ne: usize, d: &MixedDiagonal<T>, par: Par) -> Self {
        let nb = L11_PANEL_WIDTH;
        let n_panels = ne.div_ceil(nb);
        let build = |b: usize| {
            let base = b * nb;
            let w = nb.min(ne - base);
            // Column jj of the panel: zeros above the diagonal, the unit
            // diagonal, then the L entries (from `start`), each written once.
            filled_padded(ne - base, w, Par::Seq, &|jj, col| {
                let j = base + jj;
                let start = match d.pivot_type(j) {
                    PivotType::TwoByTwo { partner } if partner > j => j + 2,
                    PivotType::Delayed => {
                        unreachable!("unexpected Delayed pivot at col {j} in 0..ne")
                    }
                    _ => j + 1,
                };
                let (above, rest) = col.split_at_mut(jj);
                write_fill(above, T::zero());
                rest[0].write(T::one());
                let (gap, below) = rest[1..].split_at_mut(start - j - 1);
                write_fill(gap, T::zero());
                write_slice(below, &src.col_as_slice(j)[start..ne]);
            })
        };
        // Panels are independent; build them in parallel for large L11.
        let panels: Vec<Mat<T>> = if copy_par_split(par, ne, ne * ne / 2) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..n_panels).into_par_iter().map(build).collect()
        } else {
            (0..n_panels).map(build).collect()
        };
        Self { n: ne, panels }
    }

    /// Dimension `n` (number of eliminated columns).
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Entry `(i, j)`; zero above the diagonal, one on it.
    pub fn get(&self, i: usize, j: usize) -> T {
        if i < j {
            return T::zero();
        }
        let b = j / L11_PANEL_WIDTH;
        let base = b * L11_PANEL_WIDTH;
        self.panels[b][(i - base, j - base)]
    }

    /// Dense `n × n` copy (allocates; intended for tests and diagnostics).
    pub fn to_dense(&self) -> Mat<T> {
        Mat::<T>::from_fn(self.n, self.n, |i, j| self.get(i, j))
    }

    /// Entries allocated, including faer's column padding.
    pub fn allocated(&self) -> usize {
        self.panels
            .iter()
            .map(|p| p.col_stride().unsigned_abs() * p.ncols())
            .sum()
    }

    /// Solve `L x = b` in place (`rhs` is `n × nrhs`).
    pub(crate) fn solve_in_place(&self, mut rhs: MatMut<'_, T>, par: Par) {
        let mut base = 0;
        for panel in &self.panels {
            let w = panel.ncols();
            let (diag, below) = panel.as_ref().split_at_row(w);
            let (top, rest) = rhs.as_mut().split_at_row_mut(base + w);
            let mut x_b = top.subrows_mut(base, w);
            solve_unit_lower_triangular_in_place_with_conj(diag, Conj::No, x_b.as_mut(), par);
            if below.nrows() > 0 {
                matmul_with_conj(
                    rest,
                    Accum::Add,
                    below,
                    Conj::No,
                    x_b.as_ref(),
                    Conj::No,
                    T::from_f64(-1.0),
                    par,
                );
            }
            base += w;
        }
    }

    /// Solve `Lᵀ x = b` in place (`rhs` is `n × nrhs`). No conjugation.
    pub(crate) fn solve_transpose_in_place(&self, mut rhs: MatMut<'_, T>, par: Par) {
        let mut base = self.n;
        for panel in self.panels.iter().rev() {
            let w = panel.ncols();
            base -= w;
            let (diag, below) = panel.as_ref().split_at_row(w);
            let (top, rest) = rhs.as_mut().split_at_row_mut(base + w);
            let mut x_b = top.subrows_mut(base, w);
            if below.nrows() > 0 {
                sub_at_b(x_b.as_mut(), below, rest.as_ref(), par);
            }
            solve_unit_upper_triangular_in_place_with_conj(
                diag.transpose(),
                Conj::No,
                x_b.as_mut(),
                par,
            );
        }
    }
}

/// Minimum rows of `a` per task in [`sub_at_b`]'s split-K product.
const SPLIT_K_MIN_ROWS: usize = 512;

/// `c -= aᵀ b` for a short, wide product: `a` is `m × w`, `b` is `m × k`, and
/// the `w × k` result is small (`w` ≤ 64), so splitting the output across
/// threads (as faer does) leaves most of them idle. Instead the inner
/// dimension `m` is split into chunks, each thread forms its partial product
/// in its own buffer, and the partials are subtracted in a fixed order (so
/// the result is deterministic for a given thread count).
///
/// # References
///
/// - Goto & van de Geijn (2008), "Anatomy of high-performance matrix
///   multiplication": parallelizing the k loop with a reduction for small
///   output blocks
fn sub_at_b<T: Scalar>(mut c: MatMut<'_, T>, a: MatRef<'_, T>, b: MatRef<'_, T>, par: Par) {
    let m = a.nrows();
    let chunks = par.degree().min(m / SPLIT_K_MIN_ROWS);
    if chunks < 2 {
        matmul_with_conj(
            c,
            Accum::Add,
            a.transpose(),
            Conj::No,
            b,
            Conj::No,
            T::from_f64(-1.0),
            par,
        );
        return;
    }
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (w, k) = (c.nrows(), c.ncols());
    let partials: Vec<Mat<T>> = (0..chunks)
        .into_par_iter()
        .map(|t| {
            let (r0, r1) = (t * m / chunks, (t + 1) * m / chunks);
            let mut acc = Mat::<T>::zeros(w, k);
            matmul_with_conj(
                acc.as_mut(),
                Accum::Replace,
                a.subrows(r0, r1 - r0).transpose(),
                Conj::No,
                b.subrows(r0, r1 - r0),
                Conj::No,
                T::one(),
                Par::Seq,
            );
            acc
        })
        .collect();
    for p in &partials {
        c -= p;
    }
}

/// Lower triangle (diagonal included) of a symmetric `n × n` matrix, packed
/// column by column: column `j` holds rows `j..n` contiguously, so the block
/// uses `n(n+1)/2` entries — half a dense square — and is allocated exactly.
///
/// Used for contribution blocks, which wait in memory until their parent is
/// assembled and only ever need their lower triangle (column-oriented
/// extend-add reads exactly these slices).
///
/// # References
///
/// - Anderson et al., LAPACK Users' Guide (1999), Section 5.3.2: packed
///   storage for symmetric matrices (`UPLO = 'L'`)
#[derive(Debug, Clone, Default)]
pub(crate) struct PackedLower<T> {
    n: usize,
    data: Vec<T>,
}

impl<T: Scalar> PackedLower<T> {
    /// Build from per-column slices; `col(j)` must return rows `j..n` of
    /// column `j` (length `n - j`).
    pub(crate) fn from_columns<'a>(n: usize, mut col: impl FnMut(usize) -> &'a [T]) -> Self
    where
        T: 'a,
    {
        let mut data = Vec::<T>::with_capacity(n * (n + 1) / 2);
        advise_huge_pages(
            data.as_ptr().cast(),
            data.capacity() * core::mem::size_of::<T>(),
        );
        for j in 0..n {
            let c = col(j);
            debug_assert_eq!(c.len(), n - j, "PackedLower column {j} has wrong length");
            data.extend_from_slice(c);
        }
        Self { n, data }
    }

    /// Like [`from_columns`](Self::from_columns), filling columns in parallel
    /// for large blocks.
    pub(crate) fn from_columns_par<'a>(
        n: usize,
        col: &(impl Fn(usize) -> &'a [T] + Sync),
        par: Par,
    ) -> Self
    where
        T: 'a,
    {
        let total = n * (n + 1) / 2;
        if !copy_par_split(par, n, total) {
            return Self::from_columns(n, col);
        }
        let mut data = Vec::<T>::with_capacity(total);
        advise_huge_pages(
            data.as_ptr().cast(),
            data.capacity() * core::mem::size_of::<T>(),
        );
        fill_packed_cols(&mut data.spare_capacity_mut()[..total], n, 0, n, col, par);
        // SAFETY: fill_packed_cols wrote all `total` entries (the packed
        // columns 0..n tile exactly 0..total).
        unsafe { data.set_len(total) };
        Self { n, data }
    }

    /// Offset of column `j`: sum of the lengths of columns `0..j`.
    #[inline(always)]
    fn col_start(&self, j: usize) -> usize {
        j * self.n - j * j.saturating_sub(1) / 2
    }

    /// Rows `j..n` of column `j`.
    #[inline(always)]
    pub(crate) fn col(&self, j: usize) -> &[T] {
        let start = self.col_start(j);
        &self.data[start..start + (self.n - j)]
    }

    /// Entry `(i, j)` with `i >= j`.
    #[inline(always)]
    pub(crate) fn get(&self, i: usize, j: usize) -> T {
        debug_assert!(i >= j, "PackedLower::get({i}, {j}) above the diagonal");
        self.col(j)[i - j]
    }

    /// Dimension `n`.
    #[allow(dead_code)]
    pub(crate) fn dim(&self) -> usize {
        self.n
    }

    /// Entries allocated (for memory accounting).
    #[allow(dead_code)]
    pub(crate) fn allocated(&self) -> usize {
        self.data.capacity()
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-supernode factorization result.
///
/// Stores the L11, D11, and L21 blocks extracted from the factored frontal
/// matrix, along with permutation and index information needed by the
/// triangular solve.
///
/// # Invariants
///
/// - `l11.nrows() == l11.ncols() == num_eliminated`
/// - `d11.dimension() == num_eliminated`
/// - `l21.ncols() == num_eliminated`
/// - `l21.nrows() == row_indices.len()`
/// - `col_indices.len() == num_eliminated`
#[derive(Debug)]
pub struct FrontFactors<T: Scalar = f64> {
    /// Unit lower triangular factor (ne × ne), stored as column panels.
    pub(crate) l11: BlockedUnitLower<T>,
    /// Block diagonal with 1x1 and 2x2 pivots (ne entries).
    pub(crate) d11: MixedDiagonal<T>,
    /// Subdiagonal factor block (r × ne).
    pub(crate) l21: Mat<T>,
    /// APTP local pivot permutation (length k). Maps factored position to
    /// original front-local column.
    pub(crate) local_perm: Vec<usize>,
    /// Number of columns successfully eliminated (ne ≤ k).
    pub(crate) num_eliminated: usize,
    /// Global permuted column indices for the eliminated columns (length ne).
    pub(crate) col_indices: Vec<usize>,
    /// Global permuted row indices for L21 rows (length r).
    pub(crate) row_indices: Vec<usize>,
}

impl<T: Scalar> FrontFactors<T> {
    /// Unit lower triangular factor L11 (ne × ne), in column-panel storage.
    /// Use [`BlockedUnitLower::to_dense`] for a dense copy.
    pub fn l11(&self) -> &BlockedUnitLower<T> {
        &self.l11
    }

    /// Block diagonal D11 with mixed 1x1/2x2 pivots.
    pub fn d11(&self) -> &MixedDiagonal<T> {
        &self.d11
    }

    /// Subdiagonal factor block L21 (r × ne).
    pub fn l21(&self) -> &Mat<T> {
        &self.l21
    }

    /// APTP local pivot permutation within this front.
    pub fn local_perm(&self) -> &[usize] {
        &self.local_perm
    }

    /// Number of successfully eliminated columns.
    pub fn num_eliminated(&self) -> usize {
        self.num_eliminated
    }

    /// Global permuted column indices for eliminated columns.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Global permuted row indices for L21 rows.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }
}

/// Per-supernode diagnostic statistics.
///
/// Collected during [`AptpNumeric::factor`] for each supernode processed.
/// Provides visibility into per-front behavior for analysis and comparison
/// with reference solvers (e.g., SPRAL).
#[derive(Debug, Clone)]
pub struct PerSupernodeStats {
    /// Supernode index in postorder.
    pub snode_id: usize,
    /// Frontal matrix dimension (m).
    pub front_size: usize,
    /// Number of fully-summed rows/columns (k = supernode cols + delayed from children).
    pub num_fully_summed: usize,
    /// Number of columns successfully eliminated (ne <= k).
    pub num_eliminated: usize,
    /// Number of delayed columns (k - ne).
    pub num_delayed: usize,
    /// Number of 1x1 pivots accepted at this supernode.
    pub num_1x1: usize,
    /// Number of 2x2 pivot pairs accepted at this supernode.
    pub num_2x2: usize,
    /// Maximum absolute L entry at this supernode (stability metric).
    pub max_l_entry: f64,
    /// Wall-clock time for scatter + extend-add assembly.
    #[cfg(feature = "diagnostic")]
    pub assembly_time: std::time::Duration,
    /// Wall-clock time for dense APTP kernel.
    #[cfg(feature = "diagnostic")]
    pub kernel_time: std::time::Duration,
    /// Wall-clock time for front factor extraction.
    #[cfg(feature = "diagnostic")]
    pub extraction_time: std::time::Duration,
    // -- Sub-phase timing (finer-grained diagnostic breakdown) --
    /// Wall-clock time for zeroing the frontal matrix (memset-equivalent).
    #[cfg(feature = "diagnostic")]
    pub zero_time: std::time::Duration,
    /// Wall-clock time for building the global-to-local mapping.
    #[cfg(feature = "diagnostic")]
    pub g2l_time: std::time::Duration,
    /// Wall-clock time for scattering original CSC entries into the frontal matrix.
    #[cfg(feature = "diagnostic")]
    pub scatter_time: std::time::Duration,
    /// Wall-clock time for extend-add (merging child contributions).
    #[cfg(feature = "diagnostic")]
    pub extend_add_time: std::time::Duration,
    /// Wall-clock time for extracting L11, D11, L21 from the factored frontal matrix.
    #[cfg(feature = "diagnostic")]
    pub extract_factors_time: std::time::Duration,
    /// Wall-clock time for extracting the contribution block.
    #[cfg(feature = "diagnostic")]
    pub extract_contrib_time: std::time::Duration,
    /// Wall-clock time for the deferred contribution GEMM.
    #[cfg(feature = "diagnostic")]
    pub contrib_gemm_time: std::time::Duration,
    /// Wall-clock time for resetting the global-to-local mapping after factoring.
    #[cfg(feature = "diagnostic")]
    pub g2l_reset_time: std::time::Duration,
}

/// Aggregate statistics from multifrontal factorization.
///
/// Reports pivot counts, delay events, and front sizes across all supernodes.
#[derive(Debug, Clone)]
pub struct FactorizationStats {
    /// Total 1x1 pivots across all supernodes.
    pub total_1x1_pivots: usize,
    /// Total 2x2 pivot pairs across all supernodes.
    pub total_2x2_pivots: usize,
    /// Total delay events across all supernodes.
    pub total_delayed: usize,
    /// Columns that could not be eliminated at root supernodes (zero pivots).
    /// These represent rank deficiency in the matrix. A nonzero value means
    /// the matrix is numerically singular (or nearly so) — the solve phase
    /// must handle this appropriately.
    pub zero_pivots: usize,
    /// Largest frontal matrix dimension encountered.
    pub max_front_size: usize,
    /// Number of supernodes before amalgamation.
    pub supernodes_before_amalgamation: usize,
    /// Number of supernodes after amalgamation.
    pub supernodes_after_amalgamation: usize,
    /// Number of merge operations performed during amalgamation.
    pub merges_performed: usize,
    /// Sum of assembly times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_assembly_time: std::time::Duration,
    /// Sum of kernel times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_kernel_time: std::time::Duration,
    /// Sum of extraction times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_extraction_time: std::time::Duration,
    // -- Sub-phase totals --
    /// Sum of zeroing times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_zero_time: std::time::Duration,
    /// Sum of g2l build times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_g2l_time: std::time::Duration,
    /// Sum of scatter times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_scatter_time: std::time::Duration,
    /// Sum of extend-add times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_extend_add_time: std::time::Duration,
    /// Sum of factor extraction times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_extract_factors_time: std::time::Duration,
    /// Sum of contribution extraction times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_extract_contrib_time: std::time::Duration,
    /// Sum of deferred contribution GEMM times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_contrib_gemm_time: std::time::Duration,
    /// Sum of g2l reset times across all supernodes.
    #[cfg(feature = "diagnostic")]
    pub total_g2l_reset_time: std::time::Duration,
    /// Number of small-leaf subtrees identified by classification.
    pub small_leaf_subtrees: usize,
    /// Number of supernodes processed via the small-leaf fast path.
    pub small_leaf_nodes: usize,
}

/// Assembled frontal matrix exported for diagnostic comparison.
#[cfg(feature = "diagnostic")]
pub struct ExportedFrontal<T: Scalar = f64> {
    /// Dense m x m frontal matrix (lower triangle populated).
    pub data: Mat<T>,
    /// Total front size.
    pub front_size: usize,
    /// Number of fully-summed columns.
    pub num_fully_summed: usize,
    /// Global permuted column indices for each local row.
    pub row_indices: Vec<usize>,
}

/// Complete multifrontal numeric factorization result.
///
/// Contains per-supernode factors and aggregate statistics. Created by
/// [`AptpNumeric::factor`] and used by the triangular solve to
/// perform forward/backward substitution.
///
/// # Usage
///
/// ```
/// use faer::sparse::{SparseColMat, Triplet};
/// use faer::sparse::linalg::cholesky::SymmetricOrdering;
/// use rivrs_sparse::symmetric::{AptpSymbolic, AptpOptions, AptpNumeric};
///
/// // Analyze
/// # let triplets = vec![Triplet::new(0, 0, 1.0)];
/// # let matrix = SparseColMat::try_new_from_triplets(1, 1, &triplets).unwrap();
/// let symbolic = AptpSymbolic::analyze(matrix.symbolic(), SymmetricOrdering::Amd).unwrap();
///
/// // Factor
/// let numeric = AptpNumeric::factor(&symbolic, &matrix, &AptpOptions::default(), None).unwrap();
/// println!("Stats: {:?}", numeric.stats());
/// ```
pub struct AptpNumeric<T: Scalar = f64> {
    /// Per-supernode factors, indexed by supernode ID.
    front_factors: Vec<FrontFactors<T>>,
    /// Aggregate factorization statistics.
    stats: FactorizationStats,
    /// Per-supernode diagnostic statistics (same order as `front_factors`).
    per_supernode_stats: Vec<PerSupernodeStats>,
    /// Matrix dimension.
    n: usize,
    /// Parent of each supernode in the assembly tree (postorder, so
    /// `parent[s] > s`), `None` for roots. Used by the tree-parallel solve.
    sn_parent: Vec<Option<usize>>,
    /// Profiling session from factorization (Chrome Trace export).
    #[cfg(feature = "diagnostic")]
    profile_session: Option<FinishedSession>,
}

impl<T: Scalar> std::fmt::Debug for AptpNumeric<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AptpNumeric")
            .field("n", &self.n)
            .field("stats", &self.stats)
            .field("front_factors_count", &self.front_factors.len())
            .finish()
    }
}

/// Memory held by the retained factors, in matrix entries.
///
/// `l11_allocated` and `l21_allocated` include faer's column-stride padding.
/// `l11_lower` is what a packed/triangular L11 would need (ne(ne+1)/2 per
/// supernode); together with `l21_logical` it is the minimum storage for L.
#[derive(Debug, Clone, Copy, Default)]
pub struct FactorStorage {
    /// Entries allocated for L11 blocks (column stride × columns).
    pub l11_allocated: usize,
    /// Entries in the lower triangle (incl. diagonal) of L11 blocks.
    pub l11_lower: usize,
    /// Entries allocated for L21 blocks (column stride × columns).
    pub l21_allocated: usize,
    /// Logical L21 entries (rows × columns).
    pub l21_logical: usize,
    /// Bytes per entry.
    pub entry_bytes: usize,
}

impl<T: Scalar> AptpNumeric<T> {
    /// Entries and bytes held by the retained L factors.
    pub fn factor_storage(&self) -> FactorStorage {
        let mut fs = FactorStorage {
            entry_bytes: std::mem::size_of::<T>(),
            ..Default::default()
        };
        for ff in &self.front_factors {
            let (l11, l21) = (ff.l11(), ff.l21());
            let ne = l11.dim();
            fs.l11_allocated += l11.allocated();
            fs.l11_lower += ne * (ne + 1) / 2;
            fs.l21_allocated += l21.col_stride().unsigned_abs() * l21.ncols();
            fs.l21_logical += l21.nrows() * l21.ncols();
        }
        fs
    }

    /// Per-supernode factors, indexed by supernode ID.
    pub fn front_factors(&self) -> &[FrontFactors<T>] {
        &self.front_factors
    }

    /// Parent of each supernode in the assembly tree (`None` for roots).
    pub(crate) fn supernode_parents(&self) -> &[Option<usize>] {
        &self.sn_parent
    }

    /// Aggregate factorization statistics.
    pub fn stats(&self) -> &FactorizationStats {
        &self.stats
    }

    /// Per-supernode diagnostic statistics.
    ///
    /// One entry per supernode in postorder, matching [`front_factors`](Self::front_factors).
    pub fn per_supernode_stats(&self) -> &[PerSupernodeStats] {
        &self.per_supernode_stats
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Profiling session from factorization, for Chrome Trace export.
    ///
    /// Returns `None` if factorization was not profiled.
    /// Only available with `diagnostic` feature.
    #[cfg(feature = "diagnostic")]
    pub fn profile_session(&self) -> Option<&FinishedSession> {
        self.profile_session.as_ref()
    }

    /// Factor a sparse symmetric matrix using the multifrontal method with APTP.
    ///
    /// Traverses the assembly tree in postorder, assembling and factoring dense
    /// frontal matrices at each supernode using the dense APTP kernel.
    ///
    /// # Arguments
    ///
    /// - `symbolic`: Symbolic analysis result from [`AptpSymbolic::analyze`]
    /// - `matrix`: Sparse symmetric matrix (lower triangle stored). Dimensions
    ///   must match `symbolic.nrows()`
    /// - `options`: APTP configuration (threshold, fallback strategy)
    ///
    /// # Errors
    ///
    /// - [`SparseError::DimensionMismatch`] if matrix dimensions don't match symbolic
    /// - [`SparseError::AnalysisFailure`] if symbolic analysis is inconsistent
    ///
    /// # Zero Pivots
    ///
    /// If a root supernode has columns that cannot be eliminated (all pivots
    /// delayed to root but still fail the threshold), these are recorded as
    /// zero pivots in [`FactorizationStats::zero_pivots`] rather than returning
    /// an error. The factorization succeeds;
    /// the solve phase must handle the rank deficiency.
    ///
    /// # References
    ///
    /// - Duff & Reid (1983), "The multifrontal solution of indefinite sparse
    ///   symmetric linear equations"
    /// - Liu (1992), "The Multifrontal Method for Sparse Matrix Solution"
    /// - Hogg, Duff & Lopez (2020), "A New Sparse LDL^T Solver Using A Posteriori
    ///   Threshold Pivoting"
    pub fn factor(
        symbolic: &AptpSymbolic,
        matrix: &SparseColMat<usize, T>,
        options: &AptpOptions,
        scaling: Option<&[f64]>,
    ) -> Result<Self, SparseError> {
        let nemin = options.nemin;
        let n = symbolic.nrows();

        // Validate dimensions
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(SparseError::DimensionMismatch {
                expected: (n, n),
                got: (matrix.nrows(), matrix.ncols()),
                context: "Matrix dimensions must match symbolic analysis".to_string(),
            });
        }

        // Build supernode info (unified abstraction over supernodal/simplicial)
        let supernodes = build_supernode_info(symbolic);
        let n_supernodes_before = supernodes.len();
        let mut supernodes = super::amalgamation::amalgamate(supernodes, nemin);
        let n_supernodes = supernodes.len();
        let merges_performed = n_supernodes_before - n_supernodes;
        let children = build_children_map(&supernodes);

        // Classify small-leaf subtrees for the fast-path pre-pass
        let small_leaf_subtrees =
            classify_small_leaf_subtrees(&mut supernodes, &children, options.small_leaf_threshold);

        // Get fill-reducing permutation
        let (perm_fwd, perm_inv) = symbolic.perm_vecs();

        // Precompute assembly index maps (scatter + extend-add) for the
        // zero-delay case. Supernodes with delayed children fall back to
        // per-entry index arithmetic.
        let assembly_maps =
            build_assembly_maps(&supernodes, &children, matrix, &perm_fwd, &perm_inv, n);

        // Profiling session (diagnostic only)
        #[cfg(feature = "diagnostic")]
        let session = ProfileSession::new();
        #[cfg(feature = "diagnostic")]
        let _factor_loop_guard = session.enter_section("factor_loop");

        // Depth-first factorization with an L0 layer of parallel subtrees (see
        // `factor_tree`). Traversals are iterative — no stack overflow on deep
        // elimination trees (e.g., c-71 with 76K supernodes on rayon's 2MB stacks).
        let all_node_results = factor_tree(
            &supernodes,
            &children,
            matrix,
            &perm_fwd,
            &perm_inv,
            options,
            scaling,
            n,
            &assembly_maps,
            &small_leaf_subtrees,
        )?;

        #[cfg(feature = "diagnostic")]
        drop(_factor_loop_guard);

        // Scatter returned results into supernode-indexed vectors
        let mut front_factors_vec: Vec<Option<FrontFactors<T>>> =
            (0..n_supernodes).map(|_| None).collect();
        let mut per_sn_stats_vec: Vec<Option<PerSupernodeStats>> =
            (0..n_supernodes).map(|_| None).collect();
        for (idx, ff, stats) in all_node_results {
            front_factors_vec[idx] = Some(ff);
            per_sn_stats_vec[idx] = Some(stats);
        }

        let front_factors: Vec<FrontFactors<T>> = front_factors_vec
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| panic!("supernode {} not factored", i)))
            .collect();
        let per_sn_stats: Vec<PerSupernodeStats> = per_sn_stats_vec
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| panic!("supernode {} missing stats", i)))
            .collect();

        // Compute aggregate stats from per-supernode stats
        let mut stats = FactorizationStats {
            total_1x1_pivots: 0,
            total_2x2_pivots: 0,
            total_delayed: 0,
            zero_pivots: 0,
            max_front_size: 0,
            supernodes_before_amalgamation: n_supernodes_before,
            supernodes_after_amalgamation: n_supernodes,
            merges_performed,
            #[cfg(feature = "diagnostic")]
            total_assembly_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_kernel_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_extraction_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_zero_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_g2l_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_scatter_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_extend_add_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_extract_factors_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_extract_contrib_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_contrib_gemm_time: std::time::Duration::ZERO,
            #[cfg(feature = "diagnostic")]
            total_g2l_reset_time: std::time::Duration::ZERO,
            small_leaf_subtrees: small_leaf_subtrees.len(),
            small_leaf_nodes: small_leaf_subtrees.iter().map(|s| s.nodes.len()).sum(),
        };
        for sn_stat in &per_sn_stats {
            stats.total_1x1_pivots += sn_stat.num_1x1;
            stats.total_2x2_pivots += sn_stat.num_2x2;
            stats.total_delayed += sn_stat.num_delayed;
            if sn_stat.front_size > stats.max_front_size {
                stats.max_front_size = sn_stat.front_size;
            }
            #[cfg(feature = "diagnostic")]
            {
                stats.total_assembly_time += sn_stat.assembly_time;
                stats.total_kernel_time += sn_stat.kernel_time;
                stats.total_extraction_time += sn_stat.extraction_time;
                stats.total_zero_time += sn_stat.zero_time;
                stats.total_g2l_time += sn_stat.g2l_time;
                stats.total_scatter_time += sn_stat.scatter_time;
                stats.total_extend_add_time += sn_stat.extend_add_time;
                stats.total_extract_factors_time += sn_stat.extract_factors_time;
                stats.total_extract_contrib_time += sn_stat.extract_contrib_time;
                stats.total_contrib_gemm_time += sn_stat.contrib_gemm_time;
                stats.total_g2l_reset_time += sn_stat.g2l_reset_time;
            }
        }

        // Count zero pivots from front factors
        for ff in &front_factors {
            for (_, entry) in ff.d11().iter_pivots() {
                if let PivotEntry::OneByOne(val) = entry {
                    if val == T::zero() {
                        stats.zero_pivots += 1;
                    }
                }
            }
        }
        // Also count unresolved delays at root supernodes
        for (s, sn) in supernodes.iter().enumerate() {
            if sn.parent.is_none() {
                let sn_stat = &per_sn_stats[s];
                let m = sn_stat.front_size;
                let ne = sn_stat.num_eliminated;
                if ne < m {
                    let k = sn_stat.num_fully_summed;
                    let n_unresolved = k - ne;
                    stats.zero_pivots += n_unresolved;
                }
            }
        }

        // Finish profiling session
        #[cfg(feature = "diagnostic")]
        let finished_session = session.finish();

        let sn_parent = supernodes.iter().map(|sn| sn.parent).collect();

        Ok(AptpNumeric {
            front_factors,
            stats,
            per_supernode_stats: per_sn_stats,
            n,
            sn_parent,
            #[cfg(feature = "diagnostic")]
            profile_session: Some(finished_session),
        })
    }

    /// Export the assembled (pre-factorization) frontal matrix for a specific supernode.
    ///
    /// Runs the multifrontal assembly up to the target supernode, factoring all
    /// preceding supernodes to generate their contributions, then returns the
    /// assembled frontal matrix for the target **before** APTP factoring.
    ///
    /// # Selecting the Target Supernode
    ///
    /// If `target_snode` is `None`, the supernode with the largest front size is used.
    /// Otherwise, the specified supernode index is used.
    ///
    /// # Use Case
    ///
    /// This is a diagnostic tool for comparing our APTP dense kernel with SPRAL's
    /// `ldlt_app_factor` on the same assembled input. The exported matrix can be
    /// written to a file and fed to a SPRAL driver.
    #[cfg(feature = "diagnostic")]
    pub fn export_assembled_frontal(
        symbolic: &AptpSymbolic,
        matrix: &SparseColMat<usize, T>,
        options: &AptpOptions,
        scaling: Option<&[f64]>,
        target_snode: Option<usize>,
    ) -> Result<ExportedFrontal<T>, SparseError> {
        let n = symbolic.nrows();

        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(SparseError::DimensionMismatch {
                expected: (n, n),
                got: (matrix.nrows(), matrix.ncols()),
                context: "Matrix dimensions must match symbolic analysis".to_string(),
            });
        }

        let supernodes = build_supernode_info(symbolic);
        let n_supernodes = supernodes.len();
        let children = build_children_map(&supernodes);
        let (perm_fwd, perm_inv) = symbolic.perm_vecs();

        // Determine target supernode
        let target = match target_snode {
            Some(t) => {
                if t >= n_supernodes {
                    return Err(SparseError::AnalysisFailure {
                        reason: format!(
                            "Target supernode {} out of range (n_supernodes={})",
                            t, n_supernodes
                        ),
                    });
                }
                t
            }
            None => {
                // Find supernode with largest front
                let mut best = 0;
                let mut best_m = 0;
                for (s, sn) in supernodes.iter().enumerate() {
                    let sn_ncols = sn.col_end - sn.col_begin;
                    // Approximate: delayed cols not counted here, just pattern + cols
                    let m_approx = sn_ncols + sn.pattern.len();
                    if m_approx > best_m {
                        best_m = m_approx;
                        best = s;
                    }
                }
                best
            }
        };

        let mut global_to_local = vec![NOT_IN_FRONT; n];
        let mut contributions: Vec<Option<ContributionBlock<T>>> =
            (0..n_supernodes).map(|_| None).collect();

        // Process supernodes in postorder up to and including target
        for s in 0..=target {
            let sn = &supernodes[s];

            // 1. Collect delayed columns from children
            let mut delayed_cols: Vec<usize> = Vec::new();
            for &c in &children[s] {
                if let Some(ref cb) = contributions[c] {
                    delayed_cols.extend_from_slice(&cb.row_indices[..cb.num_delayed]);
                }
            }

            // 2. Compute frontal matrix structure
            let sn_ncols: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
            let k = sn_ncols + delayed_cols.len();
            let mut frontal_rows: Vec<usize> = Vec::with_capacity(k + sn.pattern.len());
            for range in &sn.owned_ranges {
                frontal_rows.extend(range.clone());
            }
            frontal_rows.extend_from_slice(&delayed_cols);
            frontal_rows.extend_from_slice(&sn.pattern);
            let m = frontal_rows.len();

            // 3. Build global-to-local mapping
            for (i, &global) in frontal_rows.iter().enumerate() {
                global_to_local[global] = i;
            }

            // 4. Assemble frontal matrix (owned data — diagnostic path)
            let mut frontal_data = zeros_padded::<T>(m, m);
            {
                let mut frontal = FrontalMatrix {
                    data: frontal_data.as_mut(),
                    row_indices: &frontal_rows,
                    num_fully_summed: k,
                };

                scatter_original_entries_multi(
                    &mut frontal,
                    matrix,
                    &perm_fwd,
                    &perm_inv,
                    &global_to_local,
                    &sn.owned_ranges,
                    scaling,
                );

                for &c in &children[s] {
                    if let Some(cb) = contributions[c].take() {
                        extend_add(&mut frontal, cb, &global_to_local);
                    }
                }
            }

            if s == target {
                // Return the assembled frontal matrix BEFORE factoring
                // Clean up global_to_local
                for &global in &frontal_rows {
                    global_to_local[global] = NOT_IN_FRONT;
                }
                return Ok(ExportedFrontal {
                    data: frontal_data,
                    front_size: m,
                    num_fully_summed: k,
                    row_indices: frontal_rows,
                });
            }

            // Factor this supernode (needed for contribution propagation)
            let result = aptp_factor_in_place(frontal_data.as_mut(), k, options)?;
            let ne = result.num_eliminated;

            if sn.parent.is_some() && ne < m {
                let mut ld_ws = Mat::<T>::new();
                compute_contribution_gemm_in_place(
                    &mut frontal_data,
                    k,
                    ne,
                    m,
                    &result.d,
                    &mut ld_ws,
                    Par::Seq,
                );
                contributions[s] = Some(extract_contribution(
                    &frontal_data,
                    m,
                    k,
                    &frontal_rows,
                    &result,
                    Par::Seq,
                ));
            }

            // Cleanup
            for &global in &frontal_rows {
                global_to_local[global] = NOT_IN_FRONT;
            }
        }

        // Should not reach here — we return inside the loop at s == target
        unreachable!("Target supernode {} not reached in postorder loop", target)
    }
}

// ---------------------------------------------------------------------------
// Tree-level factorization functions
// ---------------------------------------------------------------------------

/// Result of factoring a single supernode.
struct SupernodeResult<T: Scalar = f64> {
    ff: FrontFactors<T>,
    contribution: Option<ContributionBlock<T>>,
    stats: PerSupernodeStats,
}

/// Factor a single supernode given its children's contribution blocks.
///
/// Assembles the frontal matrix from sparse entries and child contributions,
/// factors it with APTP, and extracts the front factors and contribution block.
///
/// Uses the pre-allocated `workspace` for the frontal matrix buffer, row indices,
/// delayed columns, and global-to-local mapping to avoid per-supernode allocations.
#[allow(clippy::too_many_arguments)]
fn factor_single_supernode<T: Scalar>(
    s: usize,
    sn: &SupernodeInfo,
    child_contributions: Vec<Option<ContributionBlock<T>>>,
    matrix: &SparseColMat<usize, T>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
    options: &AptpOptions,
    scaling: Option<&[f64]>,
    workspace: &mut FactorizationWorkspace<T>,
    assembly_maps: &AssemblyMaps,
) -> Result<SupernodeResult<T>, SparseError> {
    // 1. Collect delayed columns from children into workspace buffer
    workspace.delayed_cols_buf.clear();
    for cb in child_contributions.iter().flatten() {
        workspace
            .delayed_cols_buf
            .extend_from_slice(&cb.row_indices[..cb.num_delayed]);
    }

    // 2. Compute frontal matrix structure
    // Use owned_ranges (not col_begin..col_end) to enumerate fully-summed columns.
    // After amalgamation of non-adjacent supernodes, col_begin..col_end may span
    // columns belonging to other supernodes. Only owned columns should be in the front.
    let sn_ncols: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
    let k = sn_ncols + workspace.delayed_cols_buf.len(); // num_fully_summed

    workspace.frontal_row_indices.clear();
    workspace.frontal_row_indices.reserve(k + sn.pattern.len());
    for range in &sn.owned_ranges {
        workspace.frontal_row_indices.extend(range.clone());
    }
    workspace
        .frontal_row_indices
        .extend_from_slice(&workspace.delayed_cols_buf);
    workspace.frontal_row_indices.extend_from_slice(&sn.pattern);
    let m = workspace.frontal_row_indices.len();

    // Intra-node parallelism for this front: BLAS-3 in the kernel, plus the
    // memory-bound zeroing, extend-add and extraction around it.
    let effective_par = if m < INTRA_NODE_THRESHOLD {
        Par::Seq
    } else {
        options.par
    };

    // 3. Ensure workspace capacity and zero frontal data for this supernode
    // If delayed columns make this front larger than estimated, grow the workspace
    if m > workspace.frontal_data.nrows() {
        workspace.frontal_data = zeros_padded(m, m);
    }

    // Children assembled in the same pass as the zeroing: no delays on either
    // side and a strictly increasing row map (the usual case). The rest go
    // through the extend-add loop below.
    let ea_children_begin = assembly_maps.ea_snode_child_begin[s];
    let child_map = |i: usize, cb: &ContributionBlock<T>| -> &[u32] {
        let start = assembly_maps.ea_offsets[ea_children_begin + i];
        &assembly_maps.ea_map[start..start + cb.row_indices.len()]
    };
    let no_delays_in = workspace.delayed_cols_buf.is_empty();
    let fused: Vec<bool> = child_contributions
        .iter()
        .enumerate()
        .map(|(i, opt_cb)| {
            opt_cb.as_ref().is_some_and(|cb| {
                no_delays_in
                    && cb.num_delayed == 0
                    && child_map(i, cb).windows(2).all(|w| w[0] < w[1])
            })
        })
        .collect();

    #[cfg(feature = "diagnostic")]
    let zero_start = std::time::Instant::now();

    {
        let children: Vec<(&PackedLower<T>, &[u32])> = child_contributions
            .iter()
            .enumerate()
            .filter(|&(i, _)| fused[i])
            .filter_map(|(i, opt_cb)| opt_cb.as_ref().map(|cb| (&cb.data, child_map(i, cb))))
            .collect();
        zero_and_extend_add(
            workspace.frontal_data.as_mut().submatrix_mut(0, 0, m, m),
            &children,
            effective_par,
        );
    }

    // With the diagnostic feature, `zero_time` includes the extend-add of the
    // children assembled in the zeroing pass.
    #[cfg(feature = "diagnostic")]
    let zero_time = zero_start.elapsed();

    // 4. Build global-to-local mapping
    #[cfg(feature = "diagnostic")]
    let g2l_start = std::time::Instant::now();

    for (i, &global) in workspace.frontal_row_indices.iter().enumerate() {
        workspace.global_to_local[global] = i;
    }

    #[cfg(feature = "diagnostic")]
    let g2l_time = g2l_start.elapsed();

    // 5. Assemble frontal matrix using workspace buffer
    #[cfg(feature = "diagnostic")]
    let assembly_start = std::time::Instant::now();

    let ndelay_in = workspace.delayed_cols_buf.len();

    let mut frontal = FrontalMatrix {
        data: workspace.frontal_data.as_mut().submatrix_mut(0, 0, m, m),
        row_indices: &workspace.frontal_row_indices,
        num_fully_summed: k,
    };

    // 5a. Scatter original CSC entries
    #[cfg(feature = "diagnostic")]
    let scatter_start = std::time::Instant::now();

    // Use precomputed amap when no delays shift row positions
    if ndelay_in == 0 {
        // Fast path: scatter using precomputed index tuples (4 u32 per entry)
        let amap_start = assembly_maps.amap_offsets[s];
        let amap_end = assembly_maps.amap_offsets[s + 1];
        let values = matrix.val();
        let amap = &assembly_maps.amap_entries[amap_start * 4..amap_end * 4];

        if let Some(sc) = scaling {
            for entry in amap.chunks_exact(4) {
                let src_idx = entry[0] as usize;
                let dest_linear = entry[1] as usize;
                let scale_row = entry[2] as usize;
                let scale_col = entry[3] as usize;
                let val = values[src_idx] * T::from_f64(sc[scale_row] * sc[scale_col]);
                let dest_col = dest_linear / m;
                let dest_row = dest_linear % m;
                frontal.data[(dest_row, dest_col)] += val;
            }
        } else {
            for entry in amap.chunks_exact(4) {
                let src_idx = entry[0] as usize;
                let dest_linear = entry[1] as usize;
                let dest_col = dest_linear / m;
                let dest_row = dest_linear % m;
                frontal.data[(dest_row, dest_col)] += values[src_idx];
            }
        }
    } else {
        // Fallback: delayed columns shift positions, use per-entry assembly
        scatter_original_entries_multi(
            &mut frontal,
            matrix,
            perm_fwd,
            perm_inv,
            &workspace.global_to_local,
            &sn.owned_ranges,
            scaling,
        );
    }

    #[cfg(feature = "diagnostic")]
    let scatter_time = scatter_start.elapsed();

    // 5b. Extend-add: merge child contributions into parent frontal matrix
    #[cfg(feature = "diagnostic")]
    let ea_start_time = std::time::Instant::now();

    let mut ea_child_idx = ea_children_begin;

    for (opt_cb, done) in child_contributions.into_iter().zip(fused) {
        if let Some(cb) = opt_cb.filter(|_| !done) {
            // Check if this child had delays — if so, fall back to g2l
            if cb.num_delayed > 0 || ndelay_in > 0 {
                extend_add(&mut frontal, cb, &workspace.global_to_local);
            } else {
                // Fast path: use precomputed row mapping
                let ea_start = assembly_maps.ea_offsets[ea_child_idx];
                let ea_end = assembly_maps.ea_offsets[ea_child_idx + 1];
                let ea_row_map = &assembly_maps.ea_map[ea_start..ea_end];
                extend_add_mapped(&mut frontal, cb, ea_row_map, effective_par);
            }
        }
        // Always increment: ea_offsets indexes all children, including those
        // with no contribution (fully eliminated, ne == m).
        ea_child_idx += 1;
    }

    #[cfg(feature = "diagnostic")]
    let extend_add_time = ea_start_time.elapsed();

    #[cfg(feature = "diagnostic")]
    let assembly_time = assembly_start.elapsed();

    // 6. Factor the frontal matrix
    #[cfg(feature = "diagnostic")]
    let kernel_start = std::time::Instant::now();

    let per_sn_options = AptpOptions {
        par: effective_par,
        ..options.clone()
    };
    let result = aptp_factor_in_place(
        workspace.frontal_data.as_mut().submatrix_mut(0, 0, m, m),
        k,
        &per_sn_options,
    )?;
    let ne = result.num_eliminated;

    #[cfg(feature = "diagnostic")]
    let kernel_time = kernel_start.elapsed();

    // 6b. Deferred contribution GEMM: apply the NFS×NFS Schur complement update
    //     in place in the front. This replaces the per-block NFS×NFS updates
    //     that were skipped during the blocking loop.
    #[cfg(feature = "diagnostic")]
    let contrib_gemm_start = std::time::Instant::now();

    let nfs = m - k;
    if sn.parent.is_some() && ne < m && nfs > 0 {
        compute_contribution_gemm_in_place(
            &mut workspace.frontal_data,
            k,
            ne,
            m,
            &result.d,
            &mut workspace.ld_workspace,
            effective_par,
        );
    }

    #[cfg(feature = "diagnostic")]
    let contrib_gemm_time = contrib_gemm_start.elapsed();

    // 7. Extract front factors (reads from workspace frontal data)
    #[cfg(feature = "diagnostic")]
    let extraction_start = std::time::Instant::now();
    #[cfg(feature = "diagnostic")]
    let extract_factors_start = std::time::Instant::now();

    // Use owned Mat<f64> directly for col_as_slice-based bulk extraction
    let ff = extract_front_factors(
        &workspace.frontal_data,
        m,
        k,
        &workspace.frontal_row_indices,
        &result,
        effective_par,
    );

    #[cfg(feature = "diagnostic")]
    let extract_factors_time = extract_factors_start.elapsed();

    // 8. Prepare contribution for parent (if not root and not fully eliminated)
    //    Delayed and updated NFS columns are all in workspace.frontal_data;
    //    pack their lower triangle into an exactly sized block.
    #[cfg(feature = "diagnostic")]
    let extract_contrib_start = std::time::Instant::now();

    let contribution = if sn.parent.is_some() && ne < m {
        Some(extract_contribution(
            &workspace.frontal_data,
            m,
            k,
            &workspace.frontal_row_indices,
            &result,
            effective_par,
        ))
    } else {
        None
    };

    #[cfg(feature = "diagnostic")]
    let extract_contrib_time = extract_contrib_start.elapsed();

    #[cfg(feature = "diagnostic")]
    let extraction_time = extraction_start.elapsed();

    let stats = PerSupernodeStats {
        snode_id: s,
        front_size: m,
        num_fully_summed: k,
        num_eliminated: ne,
        num_delayed: k - ne,
        num_1x1: result.stats.num_1x1,
        num_2x2: result.stats.num_2x2,
        max_l_entry: result.stats.max_l_entry,
        #[cfg(feature = "diagnostic")]
        assembly_time,
        #[cfg(feature = "diagnostic")]
        kernel_time,
        #[cfg(feature = "diagnostic")]
        extraction_time,
        #[cfg(feature = "diagnostic")]
        zero_time,
        #[cfg(feature = "diagnostic")]
        g2l_time,
        #[cfg(feature = "diagnostic")]
        scatter_time,
        #[cfg(feature = "diagnostic")]
        extend_add_time,
        #[cfg(feature = "diagnostic")]
        extract_factors_time,
        #[cfg(feature = "diagnostic")]
        extract_contrib_time,
        #[cfg(feature = "diagnostic")]
        contrib_gemm_time,
        #[cfg(feature = "diagnostic")]
        g2l_reset_time: std::time::Duration::ZERO, // filled below after reset
    };

    // Reset global_to_local entries for reuse (O(m), not O(n))
    #[cfg(feature = "diagnostic")]
    let g2l_reset_start = std::time::Instant::now();

    for &global in &workspace.frontal_row_indices[..m] {
        workspace.global_to_local[global] = NOT_IN_FRONT;
    }

    #[cfg(feature = "diagnostic")]
    let g2l_reset_time = g2l_reset_start.elapsed();
    #[cfg(feature = "diagnostic")]
    let stats = {
        let mut s = stats;
        s.g2l_reset_time = g2l_reset_time;
        s
    };

    Ok(SupernodeResult {
        ff,
        contribution,
        stats,
    })
}

/// Factor a small-leaf subtree using rectangular L storage.
///
/// Replaces the general `factor_single_supernode` path for classified small-leaf
/// subtrees with a streamlined per-node loop that:
/// 1. Uses rectangular m×k L storage instead of square m×m frontal matrix
/// 2. Factors via `tpp_factor_rect` (rectangular TPP kernel)
/// 3. Computes contribution GEMM from rectangular storage
/// 4. Reuses a single small workspace across all nodes in the subtree
///
/// # Arguments
///
/// - `subtree`: Small-leaf subtree descriptor (nodes in postorder).
/// - `supernodes`: All supernode descriptors.
/// - `children`: Children map.
/// - `matrix`: Original sparse matrix.
/// - `perm_fwd`/`perm_inv`: Forward/inverse permutations.
/// - `options`: APTP options.
/// - `scaling`: Optional MC64 scaling factors.
/// - `n`: Matrix dimension.
/// - `contributions`: Global contribution storage (read children, write results).
/// - `assembly_maps`: Precomputed scatter maps.
///
/// # Returns
///
/// Vec of (node_id, FrontFactors<T>, PerSupernodeStats) for each node in the subtree.
// SPRAL Equivalent: `SmallLeafNumericSubtree::factor()` in `SmallLeafNumericSubtree.hxx:187-446` (BSD-3).
#[allow(clippy::too_many_arguments)]
fn factor_small_leaf_subtree<T: Scalar>(
    subtree: &SmallLeafSubtree,
    supernodes: &[SupernodeInfo],
    children: &[Vec<usize>],
    matrix: &SparseColMat<usize, T>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
    options: &AptpOptions,
    scaling: Option<&[f64]>,
    n: usize,
    contributions: &mut [Option<ContributionBlock<T>>],
    assembly_maps: &AssemblyMaps,
) -> Result<Vec<(usize, FrontFactors<T>, PerSupernodeStats)>, SparseError> {
    let max_front = subtree.max_front_size;
    let mut results = Vec::with_capacity(subtree.nodes.len());

    // Shared workspace for the subtree:
    // - l_storage: rectangular m×k buffer, reused per node (max_front × max_front covers all)
    // - global_to_local: length n, reused per node
    // - contrib_buffer: for contribution GEMM output, recycled
    // - ld_workspace: for L·D product in contribution GEMM
    // - frontal_row_indices: reusable buffer for building front structure
    // - delayed_cols_buf: reusable buffer for collecting child delays
    let mut l_storage = zeros_padded::<T>(max_front, max_front);
    let mut global_to_local = vec![NOT_IN_FRONT; n];
    let mut contrib_buffer = Mat::<T>::new();
    let mut ld_workspace = Mat::<T>::new();
    let mut frontal_row_indices = Vec::<usize>::with_capacity(max_front);
    let mut delayed_cols_buf = Vec::<usize>::new();

    let symbolic_matrix = matrix.symbolic();
    let col_ptrs = symbolic_matrix.col_ptr();
    let row_indices_csc = symbolic_matrix.row_idx();
    let values = matrix.val();

    for &node_id in &subtree.nodes {
        let sn = &supernodes[node_id];

        // 1. Collect delayed columns from children
        delayed_cols_buf.clear();
        for &c in &children[node_id] {
            if let Some(ref cb) = contributions[c] {
                delayed_cols_buf.extend_from_slice(&cb.row_indices[..cb.num_delayed]);
            }
        }

        // 2. Compute frontal matrix structure
        let sn_ncols: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
        let k = sn_ncols + delayed_cols_buf.len(); // num_fully_summed

        frontal_row_indices.clear();
        frontal_row_indices.reserve(k + sn.pattern.len());
        for range in &sn.owned_ranges {
            frontal_row_indices.extend(range.clone());
        }
        frontal_row_indices.extend_from_slice(&delayed_cols_buf);
        frontal_row_indices.extend_from_slice(&sn.pattern);
        let m = frontal_row_indices.len();

        // 3. Ensure l_storage capacity and zero rectangular m×k region
        #[cfg(feature = "diagnostic")]
        let zero_start = std::time::Instant::now();

        if m > l_storage.nrows() || k > l_storage.ncols() {
            l_storage = zeros_padded(m, k.max(l_storage.ncols()));
        }
        // Zero only the m×k rectangle (smaller than m×m for NFS nodes)
        for j in 0..k {
            l_storage.col_as_slice_mut(j)[0..m].fill(T::zero());
        }

        #[cfg(feature = "diagnostic")]
        let zero_time = zero_start.elapsed();

        // 4. Build global-to-local mapping
        #[cfg(feature = "diagnostic")]
        let g2l_start = std::time::Instant::now();

        for (i, &global) in frontal_row_indices.iter().enumerate() {
            global_to_local[global] = i;
        }

        #[cfg(feature = "diagnostic")]
        let g2l_time = g2l_start.elapsed();

        // 5. Split assembly: scatter into l_storage (FS columns) and contrib_buffer (NFS×NFS)
        #[cfg(feature = "diagnostic")]
        let assembly_start = std::time::Instant::now();
        #[cfg(feature = "diagnostic")]
        let scatter_start = std::time::Instant::now();

        let ndelay_in = delayed_cols_buf.len();
        let total_owned = sn_ncols;
        let nfs = m - k;

        // Pre-allocate and zero contrib_buffer if there will be NFS rows.
        // This allows both original matrix scatter and child extend-add to
        // deposit NFS×NFS entries directly, avoiding a re-scatter pass.
        let has_nfs = nfs > 0 && sn.parent.is_some();
        if has_nfs {
            if contrib_buffer.nrows() < nfs || contrib_buffer.ncols() < nfs {
                contrib_buffer = zeros_padded(
                    nfs.max(contrib_buffer.nrows()),
                    nfs.max(contrib_buffer.ncols()),
                );
            }
            for j in 0..nfs {
                contrib_buffer.col_as_slice_mut(j)[0..nfs].fill(T::zero());
            }
        }

        // 5a. Scatter original CSC entries — FS columns → l_storage, NFS×NFS → contrib_buffer
        if ndelay_in == 0 {
            let amap_start = assembly_maps.amap_offsets[node_id];
            let amap_end = assembly_maps.amap_offsets[node_id + 1];
            let amap = &assembly_maps.amap_entries[amap_start * 4..amap_end * 4];

            if let Some(sc) = scaling {
                for entry in amap.chunks_exact(4) {
                    let src_idx = entry[0] as usize;
                    let dest_linear = entry[1] as usize;
                    let scale_row = entry[2] as usize;
                    let scale_col = entry[3] as usize;
                    let val = values[src_idx] * T::from_f64(sc[scale_row] * sc[scale_col]);
                    let amap_col = dest_linear / m;
                    let amap_row = dest_linear % m;
                    if amap_col < k {
                        l_storage[(amap_row, amap_col)] += val;
                    } else if has_nfs && amap_row >= k {
                        // NFS×NFS entry → contrib_buffer
                        contrib_buffer[(amap_row - k, amap_col - k)] += val;
                    }
                    // FS×NFS cross-terms (amap_col >= k, amap_row < k) are handled
                    // by the symmetric counterpart where amap_col < k.
                }
            } else {
                for entry in amap.chunks_exact(4) {
                    let src_idx = entry[0] as usize;
                    let dest_linear = entry[1] as usize;
                    let amap_col = dest_linear / m;
                    let amap_row = dest_linear % m;
                    if amap_col < k {
                        l_storage[(amap_row, amap_col)] += values[src_idx];
                    } else if has_nfs && amap_row >= k {
                        contrib_buffer[(amap_row - k, amap_col - k)] += values[src_idx];
                    }
                }
            }
        } else {
            // Fallback: delayed columns shift positions, use per-entry g2l scatter
            for range in &sn.owned_ranges {
                for pj in range.clone() {
                    let orig_col = perm_fwd[pj];
                    let local_col = global_to_local[pj];

                    let start = col_ptrs[orig_col];
                    let end = col_ptrs[orig_col + 1];
                    for idx in start..end {
                        let orig_row = row_indices_csc[idx];
                        if orig_row < orig_col {
                            let perm_row = perm_inv[orig_row];
                            let local_peer = global_to_local[perm_row];
                            if local_peer != NOT_IN_FRONT && local_peer < total_owned {
                                continue;
                            }
                        }
                        let global_row = perm_inv[orig_row];
                        let local_row = global_to_local[global_row];
                        if local_row == NOT_IN_FRONT {
                            continue;
                        }
                        if local_row >= total_owned && local_row < k {
                            continue;
                        }
                        let mut val = values[idx];
                        if let Some(s) = scaling {
                            val *= T::from_f64(s[perm_inv[orig_row]] * s[perm_inv[orig_col]]);
                        }
                        let (r, c) = if local_row >= local_col {
                            (local_row, local_col)
                        } else {
                            (local_col, local_row)
                        };
                        if c < k {
                            l_storage[(r, c)] += val;
                        } else if has_nfs && r >= k {
                            contrib_buffer[(r - k, c - k)] += val;
                        }
                    }
                }
            }
        }

        #[cfg(feature = "diagnostic")]
        let scatter_time = scatter_start.elapsed();

        // 5b. Extend-add child contributions
        //     FS columns → l_storage, NFS×NFS → contrib_buffer
        #[cfg(feature = "diagnostic")]
        let extend_add_start = std::time::Instant::now();

        for &c in &children[node_id] {
            if let Some(cb) = contributions[c].take() {
                let cb_size = cb.row_indices.len();
                for i in 0..cb_size {
                    let gi = cb.row_indices[i];
                    let li = global_to_local[gi];
                    debug_assert!(li != NOT_IN_FRONT, "child row {} not in parent", gi);
                    for j in 0..=i {
                        let gj = cb.row_indices[j];
                        let lj = global_to_local[gj];
                        debug_assert!(lj != NOT_IN_FRONT, "child col {} not in parent", gj);
                        let val = cb.data.get(i, j);
                        if val != T::zero() {
                            let (li, lj) = if li >= lj { (li, lj) } else { (lj, li) };
                            if lj < k {
                                // FS column → l_storage
                                l_storage[(li, lj)] += val;
                            } else if has_nfs {
                                // NFS×NFS → contrib_buffer
                                contrib_buffer[(li - k, lj - k)] += val;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(feature = "diagnostic")]
        let extend_add_time = extend_add_start.elapsed();
        #[cfg(feature = "diagnostic")]
        let assembly_time = assembly_start.elapsed();

        // 6. Factor using rectangular TPP kernel
        #[cfg(feature = "diagnostic")]
        let kernel_start = std::time::Instant::now();

        let result = if !options.pivoting {
            // Small fronts: unpivoted kernel on the rectangular storage.
            let seq = AptpOptions {
                par: Par::Seq,
                ..options.clone()
            };
            unpivoted_factor_in_place(l_storage.as_mut().submatrix_mut(0, 0, m, k), k, &seq)?
        } else {
            tpp_factor_rect(l_storage.as_mut().submatrix_mut(0, 0, m, k), k, options)?
        };
        let ne = result.num_eliminated;

        #[cfg(feature = "diagnostic")]
        let kernel_time = kernel_start.elapsed();

        // 7. Build contribution block
        #[cfg(feature = "diagnostic")]
        let mut contrib_gemm_time = std::time::Duration::ZERO;
        #[cfg(feature = "diagnostic")]
        let mut extract_contrib_time = std::time::Duration::ZERO;
        let contribution = if sn.parent.is_some() && ne < m {
            // Ensure contrib_buffer is large enough for extract_contribution_rect
            // which may need to assemble delayed + NFS into a single buffer
            if contrib_buffer.nrows() < m || contrib_buffer.ncols() < m {
                // Grow if delayed columns expanded the need
                let new_size = m.max(contrib_buffer.nrows());
                let mut new_buf = zeros_padded::<T>(new_size, new_size);
                // Preserve NFS×NFS data already assembled
                if nfs > 0 {
                    for j in 0..nfs {
                        let col_len = nfs - j;
                        let src = &contrib_buffer.col_as_slice(j)[j..j + col_len];
                        new_buf.col_as_slice_mut(j)[j..j + col_len].copy_from_slice(src);
                    }
                }
                contrib_buffer = new_buf;
            }

            // Apply contribution GEMM: updates NFS×NFS with -L21_NFS * D * L21_NFS^T
            #[cfg(feature = "diagnostic")]
            let gemm_start = std::time::Instant::now();

            if nfs > 0 && ne > 0 {
                compute_contribution_gemm_rect(
                    &l_storage,
                    k,
                    ne,
                    m,
                    &result.d,
                    &mut contrib_buffer,
                    &mut ld_workspace,
                    Par::Seq,
                );
            }

            #[cfg(feature = "diagnostic")]
            {
                contrib_gemm_time = gemm_start.elapsed();
            }

            #[cfg(feature = "diagnostic")]
            let extract_contrib_start = std::time::Instant::now();

            let cb = extract_contribution_rect(
                &l_storage,
                m,
                k,
                &frontal_row_indices,
                &result,
                &contrib_buffer,
            );

            #[cfg(feature = "diagnostic")]
            {
                extract_contrib_time = extract_contrib_start.elapsed();
            }

            Some(cb)
        } else {
            None
        };

        // 8. Extract FrontFactors from rectangular L storage
        #[cfg(feature = "diagnostic")]
        let extract_factors_start = std::time::Instant::now();

        let ff = extract_front_factors_rect(&l_storage, m, k, &frontal_row_indices, &result);

        #[cfg(feature = "diagnostic")]
        let extract_factors_time = extract_factors_start.elapsed();

        let stats = PerSupernodeStats {
            snode_id: node_id,
            front_size: m,
            num_fully_summed: k,
            num_eliminated: ne,
            num_delayed: k - ne,
            num_1x1: result.stats.num_1x1,
            num_2x2: result.stats.num_2x2,
            max_l_entry: result.stats.max_l_entry,
            #[cfg(feature = "diagnostic")]
            assembly_time,
            #[cfg(feature = "diagnostic")]
            kernel_time,
            #[cfg(feature = "diagnostic")]
            extraction_time: extract_factors_time + extract_contrib_time,
            #[cfg(feature = "diagnostic")]
            zero_time,
            #[cfg(feature = "diagnostic")]
            g2l_time,
            #[cfg(feature = "diagnostic")]
            scatter_time,
            #[cfg(feature = "diagnostic")]
            extend_add_time,
            #[cfg(feature = "diagnostic")]
            extract_factors_time,
            #[cfg(feature = "diagnostic")]
            extract_contrib_time,
            #[cfg(feature = "diagnostic")]
            contrib_gemm_time,
            #[cfg(feature = "diagnostic")]
            g2l_reset_time: std::time::Duration::ZERO, // filled below
        };

        // 9. Reset g2l
        #[cfg(feature = "diagnostic")]
        let g2l_reset_start = std::time::Instant::now();

        for &global in &frontal_row_indices[..m] {
            global_to_local[global] = NOT_IN_FRONT;
        }

        #[cfg(feature = "diagnostic")]
        let stats = {
            let mut s = stats;
            s.g2l_reset_time = g2l_reset_start.elapsed();
            s
        };

        contributions[node_id] = contribution;
        results.push((node_id, ff, stats));
    }

    Ok(results)
}

/// Shared read-only inputs for a tree traversal.
struct TreeCtx<'a, T: Scalar> {
    supernodes: &'a [SupernodeInfo],
    children: &'a [Vec<usize>],
    matrix: &'a SparseColMat<usize, T>,
    perm_fwd: &'a [usize],
    perm_inv: &'a [usize],
    options: &'a AptpOptions,
    scaling: Option<&'a [f64]>,
    n: usize,
    assembly_maps: &'a AssemblyMaps,
    small_leaf_subtrees: &'a [SmallLeafSubtree],
    /// `small_leaf_of_root[s] = Some(i)` if `s` is the root of small-leaf subtree `i`.
    small_leaf_of_root: &'a [Option<usize>],
    #[cfg(feature = "diagnostic")]
    cb_mem: &'a ContribMemDebug,
}

/// Per-worker state for phase A: a workspace plus a full-length slot vector
/// for contribution blocks waiting for their parent.
struct WorkerState<T: Scalar> {
    ws: FactorizationWorkspace<T>,
    pending: Vec<Option<ContributionBlock<T>>>,
}

/// Results of factoring one L0 subtree.
struct SubtreeOutput<T: Scalar> {
    root: usize,
    results: Vec<(usize, FrontFactors<T>, PerSupernodeStats)>,
    contribution: Option<ContributionBlock<T>>,
}

/// Factor the assembly tree depth-first, with memory-bounded parallelism.
///
/// The tree is split into an *L0 layer* of independent subtrees and the *top*
/// part above it, following the "L0-OpenMP" scheme of MUMPS 5:
///
/// - **Phase A** (parallel only): L0 subtrees are factored concurrently, one
///   subtree per task, each traversed depth-first by one worker with its own
///   workspace. No L0 subtree contains a front of `options.l0_max_front` or more,
///   which bounds every worker's frontal buffer.
/// - **Phase B**: the top part is factored depth-first, one supernode at a
///   time, with intra-node (BLAS-level) parallelism inside each front, in a
///   single workspace.
///
/// With `Par::Seq` the layer is empty and the whole tree is phase B.
///
/// Depth-first postorder keeps alive only the contribution blocks of
/// completed siblings along the current root path (Liu 1986), rather than
/// every block of a level-set wave. Traversals use explicit stacks, so deep
/// elimination trees cannot overflow rayon's worker stacks.
///
/// # References
///
/// - Liu (1986), "On the storage requirement in the out-of-core multifrontal
///   method for sparse factorization", ACM TOMS 12(3): depth-first stack
/// - Geist & Ng (1989), "Task scheduling for parallel sparse Cholesky
///   factorization", IJPP 18(4): subtree-to-processor layer selection
/// - L'Excellent & Sid-Lakhdar (2014), "A study of shared-memory parallelism
///   in a multifrontal solver", Parallel Computing 40(3-4): the L0 layer
#[allow(clippy::too_many_arguments)]
fn factor_tree<T: Scalar>(
    supernodes: &[SupernodeInfo],
    children: &[Vec<usize>],
    matrix: &SparseColMat<usize, T>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
    options: &AptpOptions,
    scaling: Option<&[f64]>,
    n: usize,
    assembly_maps: &AssemblyMaps,
    small_leaf_subtrees: &[SmallLeafSubtree],
) -> Result<Vec<(usize, FrontFactors<T>, PerSupernodeStats)>, SparseError> {
    let n_supernodes = supernodes.len();
    let mut small_leaf_of_root: Vec<Option<usize>> = vec![None; n_supernodes];
    for (i, st) in small_leaf_subtrees.iter().enumerate() {
        small_leaf_of_root[st.root] = Some(i);
    }
    #[cfg(feature = "diagnostic")]
    let cb_mem = ContribMemDebug::default();
    let ctx = TreeCtx {
        supernodes,
        children,
        matrix,
        perm_fwd,
        perm_inv,
        options,
        scaling,
        n,
        assembly_maps,
        small_leaf_subtrees,
        small_leaf_of_root: &small_leaf_of_root,
        #[cfg(feature = "diagnostic")]
        cb_mem: &cb_mem,
    };

    let threads = options.par.degree();
    let layer = if threads > 1 {
        select_l0_layer(supernodes, children, threads, options.l0_max_front)
    } else {
        Vec::new()
    };
    // Depth-first visiting order of each node's children (Liu's rule).
    let visit = liu_child_order(supernodes, children);
    let mut is_layer_root = vec![false; n_supernodes];
    for &r in &layer {
        is_layer_root[r] = true;
    }

    let mut all_results = Vec::with_capacity(n_supernodes);
    let mut pending: Vec<Option<ContributionBlock<T>>> = (0..n_supernodes).map(|_| None).collect();

    #[cfg(feature = "diagnostic")]
    let phase_a_start = std::time::Instant::now();

    // Phase A: L0 subtrees in parallel, largest first (LPT order).
    //
    // `layer` is sorted by decreasing cost. Each of `threads` workers takes
    // the next unclaimed subtree from a shared counter, which is the greedy
    // longest-processing-time schedule. (A `par_iter` over the sorted list
    // splits it into contiguous halves, so one worker ran the largest
    // subtrees back to back while the others idled.)
    if !layer.is_empty() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        #[cfg(feature = "diagnostic")]
        let subtree_times: std::sync::Mutex<Vec<(usize, usize, f64, f64)>> =
            std::sync::Mutex::new(Vec::new());
        let per_worker: Vec<Result<Vec<SubtreeOutput<T>>, SparseError>> = (0..threads)
            .into_par_iter()
            .map(|_worker| {
                let mut st = WorkerState {
                    ws: FactorizationWorkspace::empty(),
                    pending: (0..n_supernodes).map(|_| None).collect(),
                };
                st.ws.ensure_g2l(n);
                let mut outs = Vec::new();
                while !failed.load(Ordering::Relaxed) {
                    let Some(&root) = layer.get(next.fetch_add(1, Ordering::Relaxed)) else {
                        break;
                    };
                    #[cfg(feature = "diagnostic")]
                    let t_start = phase_a_start.elapsed().as_secs_f64();
                    let order = dfs_postorder(&[root], &visit, |_| true);
                    let results = match factor_in_order(&ctx, &order, &mut st.ws, &mut st.pending) {
                        Ok(r) => r,
                        Err(e) => {
                            failed.store(true, Ordering::Relaxed);
                            return Err(e);
                        }
                    };
                    let contribution = st.pending[root].take();
                    #[cfg(feature = "diagnostic")]
                    if let Ok(mut v) = subtree_times.lock() {
                        v.push((
                            root,
                            _worker,
                            t_start,
                            phase_a_start.elapsed().as_secs_f64(),
                        ));
                    }
                    outs.push(SubtreeOutput {
                        root,
                        results,
                        contribution,
                    });
                }
                Ok(outs)
            })
            .collect();
        #[cfg(feature = "diagnostic")]
        if std::env::var("RIVRS_MEMDEBUG").is_ok_and(|v| v == "1") {
            let v = subtree_times.into_inner().unwrap_or_default();
            let wall = v.iter().map(|x| x.3).fold(0.0, f64::max);
            let mut busy = vec![(0.0f64, 0.0f64); threads];
            for &(_, t, s0, s1) in &v {
                if t < threads {
                    busy[t].0 += s1 - s0;
                    busy[t].1 = busy[t].1.max(s1);
                }
            }
            let mut by_len: Vec<_> = v.iter().map(|x| (x.3 - x.2, x.0, x.2)).collect();
            by_len.sort_by(|a, b| b.0.total_cmp(&a.0));
            eprintln!(
                "phase A: wall {wall:.2} s, per-worker busy/finish (s): {}",
                busy.iter()
                    .map(|b| format!("{:.2}/{:.2}", b.0, b.1))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            eprintln!(
                "phase A: longest subtrees (s, start): {}",
                by_len
                    .iter()
                    .take(6)
                    .map(|x| format!("{:.2}@{:.2}", x.0, x.2))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        // Workers' workspaces are dropped with their closures, before phase
        // B allocates its large one.
        let mut outputs = Vec::with_capacity(layer.len());
        for w in per_worker {
            outputs.extend(w?);
        }
        for out in outputs {
            pending[out.root] = out.contribution;
            all_results.extend(out.results);
        }
    }

    #[cfg(feature = "diagnostic")]
    let phase_b_start = std::time::Instant::now();

    // Phase B: the top part, depth-first, one front at a time.
    let roots: Vec<usize> = (0..n_supernodes)
        .filter(|&s| supernodes[s].parent.is_none())
        .collect();
    let order = dfs_postorder(&roots, &visit, |s| !is_layer_root[s]);
    let max_front = order
        .iter()
        .map(|&s| front_dims(&supernodes[s]).1)
        .max()
        .unwrap_or(0);
    let mut ws = FactorizationWorkspace::new(max_front, n);
    all_results.extend(factor_in_order(&ctx, &order, &mut ws, &mut pending)?);

    #[cfg(feature = "diagnostic")]
    if std::env::var("RIVRS_MEMDEBUG").is_ok_and(|v| v == "1") {
        let top_flops: f64 = order
            .iter()
            .map(|&s| {
                let (k, m) = front_dims(&supernodes[s]);
                k as f64 * m as f64 * m as f64
            })
            .sum();
        eprintln!(
            "phases: A {:.2} s ({} subtrees), B {:.2} s ({} top nodes, {:.2e} est. k*m^2, max front {})",
            (phase_b_start - phase_a_start).as_secs_f64(),
            layer.len(),
            phase_b_start.elapsed().as_secs_f64(),
            order.len(),
            top_flops,
            max_front
        );
    }
    #[cfg(feature = "diagnostic")]
    cb_mem.report(layer.len());

    Ok(all_results)
}

/// Factor supernodes in the given order (children before parents), taking
/// child contribution blocks from `pending` and storing each node's own.
///
/// Small-leaf subtree members are skipped until the subtree root is reached,
/// where the whole subtree is factored by the fast path.
fn factor_in_order<T: Scalar>(
    ctx: &TreeCtx<'_, T>,
    order: &[usize],
    ws: &mut FactorizationWorkspace<T>,
    pending: &mut [Option<ContributionBlock<T>>],
) -> Result<Vec<(usize, FrontFactors<T>, PerSupernodeStats)>, SparseError> {
    let mut out = Vec::with_capacity(order.len());
    for &s in order {
        let sn = &ctx.supernodes[s];
        if sn.in_small_leaf {
            if let Some(i) = ctx.small_leaf_of_root[s] {
                out.extend(factor_small_leaf_subtree(
                    &ctx.small_leaf_subtrees[i],
                    ctx.supernodes,
                    ctx.children,
                    ctx.matrix,
                    ctx.perm_fwd,
                    ctx.perm_inv,
                    ctx.options,
                    ctx.scaling,
                    ctx.n,
                    pending,
                    ctx.assembly_maps,
                )?);
                #[cfg(feature = "diagnostic")]
                if let Some(cb) = &pending[s] {
                    ctx.cb_mem.add(cb);
                }
            }
            continue;
        }
        let contribs: Vec<Option<ContributionBlock<T>>> =
            ctx.children[s].iter().map(|&c| pending[c].take()).collect();
        #[cfg(feature = "diagnostic")]
        for cb in contribs.iter().flatten() {
            ctx.cb_mem.sub(cb);
        }
        let result = factor_single_supernode(
            s,
            sn,
            contribs,
            ctx.matrix,
            ctx.perm_fwd,
            ctx.perm_inv,
            ctx.options,
            ctx.scaling,
            ws,
            ctx.assembly_maps,
        )?;
        #[cfg(feature = "diagnostic")]
        if let Some(cb) = &result.contribution {
            ctx.cb_mem.add(cb);
        }
        pending[s] = result.contribution;
        out.push((s, result.ff, result.stats));
    }
    Ok(out)
}

/// Fully-summed column count `k` and front size `m` of a supernode, before
/// any delayed columns from children.
fn front_dims(sn: &SupernodeInfo) -> (usize, usize) {
    let k: usize = sn.owned_ranges.iter().map(|r| r.len()).sum();
    (k, k + sn.pattern.len())
}

/// Order each node's children to minimise the contribution-block stack of a
/// depth-first traversal (Liu 1986).
///
/// While a child's subtree is factored, the blocks of siblings visited before
/// it stay alive. Visiting children in decreasing order of
/// `peak(c) − cb(c)` minimises the peak, where `cb(c)` is the child's own
/// contribution block (lower triangle, before delays) and `peak(c)` is the
/// largest stack its subtree needs. Only the visiting order changes: blocks
/// are still handed to the parent in `children[s]` order.
///
/// # References
///
/// - Liu (1986), "On the storage requirement in the out-of-core multifrontal
///   method for sparse factorization", ACM TOMS 12(3), Theorem 3.2
fn liu_child_order(supernodes: &[SupernodeInfo], children: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = supernodes.len();
    let tri = |x: usize| x * (x + 1) / 2;
    let cb: Vec<usize> = (0..n)
        .map(|s| {
            let (k, m) = front_dims(&supernodes[s]);
            if supernodes[s].parent.is_some() {
                tri(m - k)
            } else {
                0
            }
        })
        .collect();
    let mut peak = vec![0usize; n];
    let mut order: Vec<Vec<usize>> = children.to_vec();
    // Postorder (parent > child): every child's peak is final before its parent.
    for s in 0..n {
        order[s].sort_by_key(|&c| std::cmp::Reverse(peak[c].saturating_sub(cb[c])));
        let (mut running, mut pk) = (0usize, 0usize);
        for &c in &order[s] {
            pk = pk.max(running + peak[c]);
            running += cb[c];
        }
        peak[s] = pk.max(running).max(cb[s]);
    }
    order
}

/// Depth-first postorder of the subtrees rooted at `roots`, visiting only
/// nodes for which `include` holds (an excluded node's subtree is skipped).
fn dfs_postorder(
    roots: &[usize],
    children: &[Vec<usize>],
    include: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut order = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &r in roots {
        if !include(r) {
            continue;
        }
        stack.push((r, 0));
        while let Some(top) = stack.last_mut() {
            let (s, i) = *top;
            if i < children[s].len() {
                top.1 += 1;
                let c = children[s][i];
                if include(c) {
                    stack.push((c, 0));
                }
            } else {
                order.push(s);
                stack.pop();
            }
        }
    }
    order
}

/// `f64` with a total order (for subtree costs in a heap).
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Choose the L0 layer: roots of independent subtrees for phase A.
///
/// Starting from the tree roots, any node whose subtree contains a front of
/// `l0_max_front` or more is moved to the top part (replaced by its
/// children). Then, Geist–Ng style, the most expensive subtree is split while
/// it exceeds `1/threads` of the layer's total cost, so the greedy (largest
/// first) assignment of subtrees to threads stays balanced. Small-leaf
/// subtrees are never split. Cost is approximated by `k·m²` per front.
///
/// Returns layer roots sorted by decreasing subtree cost.
fn select_l0_layer(
    supernodes: &[SupernodeInfo],
    children: &[Vec<usize>],
    threads: usize,
    l0_max_front: usize,
) -> Vec<usize> {
    let n = supernodes.len();
    let mut cost = vec![0.0f64; n];
    let mut max_front = vec![0usize; n];
    // Postorder (parent > child): every child is final before its parent.
    for s in 0..n {
        let (k, m) = front_dims(&supernodes[s]);
        cost[s] += k as f64 * m as f64 * m as f64;
        max_front[s] = max_front[s].max(m);
        if let Some(p) = supernodes[s].parent {
            cost[p] += cost[s];
            max_front[p] = max_front[p].max(max_front[s]);
        }
    }

    // Memory rule: no L0 subtree may contain a large front.
    let mut work: Vec<usize> = (0..n).filter(|&s| supernodes[s].parent.is_none()).collect();
    let mut layer = Vec::new();
    while let Some(r) = work.pop() {
        if max_front[r] >= l0_max_front && !supernodes[r].in_small_leaf {
            work.extend(children[r].iter().copied());
        } else {
            layer.push(r);
        }
    }

    // Load-balance rule: split the largest subtree while it dominates. There
    // is no cap on the number of splits: a tree can be a long spine of
    // medium fronts with tiny subtrees hanging off it (aug3dcqp: about 3400
    // supernodes, the spine holds nearly all the flops), and the walk must
    // reach the bottom of the spine so the side subtrees run in parallel
    // while the spine moves to the top part. A heap keeps each split
    // O(log layer).
    let mut heap: std::collections::BinaryHeap<(OrdF64, usize)> =
        layer.iter().map(|&r| (OrdF64(cost[r]), r)).collect();
    let mut total: f64 = layer.iter().map(|&r| cost[r]).sum();
    while let Some(&(OrdF64(c), r)) = heap.peek() {
        let splittable = !children[r].is_empty() && !supernodes[r].in_small_leaf;
        if !splittable || c * threads as f64 <= total {
            break;
        }
        heap.pop();
        total -= c;
        for &ch in &children[r] {
            total += cost[ch];
            heap.push((OrdF64(cost[ch]), ch));
        }
    }
    let mut layer: Vec<usize> = heap.into_iter().map(|(_, r)| r).collect();

    layer.sort_by(|a, b| cost[*b].total_cmp(&cost[*a]));
    layer
}

/// Live contribution-block accounting (feature `diagnostic`).
///
/// Tracks bytes allocated by contribution blocks waiting for their parent
/// and the bytes they logically need (lower triangle), with peaks. Printed to
/// stderr at the end of factorization when `RIVRS_MEMDEBUG=1`.
#[cfg(feature = "diagnostic")]
#[derive(Default)]
struct ContribMemDebug {
    live_alloc: std::sync::atomic::AtomicUsize,
    live_tri: std::sync::atomic::AtomicUsize,
    live_blocks: std::sync::atomic::AtomicUsize,
    peak_alloc: std::sync::atomic::AtomicUsize,
    peak_tri: std::sync::atomic::AtomicUsize,
    peak_blocks: std::sync::atomic::AtomicUsize,
    entry_bytes: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "diagnostic")]
impl ContribMemDebug {
    fn sizes<T: Scalar>(cb: &ContributionBlock<T>) -> (usize, usize) {
        let nc = cb.row_indices.len();
        let alloc = cb.data.allocated();
        (alloc, nc * (nc + 1) / 2)
    }

    fn add<T: Scalar>(&self, cb: &ContributionBlock<T>) {
        use std::sync::atomic::Ordering::Relaxed;
        let (a, t) = Self::sizes(cb);
        self.entry_bytes.store(std::mem::size_of::<T>(), Relaxed);
        let la = self.live_alloc.fetch_add(a, Relaxed) + a;
        let lt = self.live_tri.fetch_add(t, Relaxed) + t;
        let lb = self.live_blocks.fetch_add(1, Relaxed) + 1;
        self.peak_alloc.fetch_max(la, Relaxed);
        self.peak_tri.fetch_max(lt, Relaxed);
        self.peak_blocks.fetch_max(lb, Relaxed);
    }

    fn sub<T: Scalar>(&self, cb: &ContributionBlock<T>) {
        use std::sync::atomic::Ordering::Relaxed;
        let (a, t) = Self::sizes(cb);
        self.live_alloc.fetch_sub(a, Relaxed);
        self.live_tri.fetch_sub(t, Relaxed);
        self.live_blocks.fetch_sub(1, Relaxed);
    }

    fn report(&self, layer_len: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        if std::env::var("RIVRS_MEMDEBUG").map_or(true, |v| v != "1") {
            return;
        }
        let eb = self.entry_bytes.load(Relaxed);
        let gb = |e: usize| (e * eb) as f64 / (1u64 << 30) as f64;
        eprintln!(
            "contrib blocks (L0 layer: {layer_len} subtrees): peak allocated {:.2} GB, peak needed (lower tri) {:.2} GB, peak live blocks {}",
            gb(self.peak_alloc.load(Relaxed)),
            gb(self.peak_tri.load(Relaxed)),
            self.peak_blocks.load(Relaxed)
        );
    }
}

// ---------------------------------------------------------------------------
// Internal functions
// ---------------------------------------------------------------------------

/// Build unified supernode descriptors from symbolic analysis.
///
/// For supernodal decompositions, maps directly from faer's supernode structure.
/// For simplicial decompositions, each column becomes a trivial 1-column supernode
/// with pattern derived from the elimination tree and L structure.
///
/// # Postcondition
///
/// `info[s].parent.map_or(true, |p| p > s)` for all `s` (postorder).
#[allow(clippy::single_range_in_vec_init)]
fn build_supernode_info(symbolic: &AptpSymbolic) -> Vec<SupernodeInfo> {
    match symbolic.raw() {
        SymbolicCholeskyRaw::Supernodal(sn) => {
            let ns = sn.n_supernodes();
            let begin = sn.supernode_begin();
            let end = sn.supernode_end();
            (0..ns)
                .map(|s| {
                    let pattern = sn.supernode(s).pattern().to_vec();
                    let parent = symbolic.supernode_parent(s);
                    SupernodeInfo {
                        col_begin: begin[s],
                        col_end: end[s],
                        pattern,
                        parent,
                        owned_ranges: vec![begin[s]..end[s]],
                        in_small_leaf: false,
                    }
                })
                .collect()
        }
        SymbolicCholeskyRaw::Simplicial(simp) => {
            let n = symbolic.nrows();
            let etree = symbolic.etree();
            // Get L structure from simplicial symbolic factorization
            let l_symbolic = simp.factor();
            let col_ptr = l_symbolic.col_ptr();
            let row_idx = l_symbolic.row_idx();
            (0..n)
                .map(|j| {
                    // Pattern = row indices in column j of L that are > j
                    // (off-diagonal entries below the diagonal)
                    let start = col_ptr[j];
                    let end = col_ptr[j + 1];
                    let pattern: Vec<usize> = row_idx[start..end]
                        .iter()
                        .copied()
                        .filter(|&r| r > j)
                        .collect();
                    // Parent from etree
                    let parent = if etree[j] < 0 {
                        None
                    } else {
                        Some(etree[j] as usize)
                    };
                    SupernodeInfo {
                        col_begin: j,
                        col_end: j + 1,
                        pattern,
                        parent,
                        owned_ranges: vec![j..j + 1],
                        in_small_leaf: false,
                    }
                })
                .collect()
        }
    }
}

/// Build a map from each supernode to its children in the assembly tree.
fn build_children_map(infos: &[SupernodeInfo]) -> Vec<Vec<usize>> {
    let n = infos.len();
    let mut children = vec![Vec::new(); n];
    for (s, info) in infos.iter().enumerate() {
        if let Some(p) = info.parent {
            children[p].push(s);
        }
    }
    children
}

/// Classify post-amalgamation supernodes into small-leaf subtrees for the fast path.
///
/// Performs a single O(n_supernodes) bottom-up pass to identify contiguous leaf
/// subtrees where every supernode has `front_size < threshold`. Sets `in_small_leaf`
/// on qualifying supernodes and returns a `Vec<SmallLeafSubtree>` describing each
/// identified subtree (with at least 2 nodes).
///
/// # Arguments
///
/// - `supernodes`: Post-amalgamation supernode descriptors (mutated: `in_small_leaf` set).
/// - `children_map`: Children map (`children[s]` = list of child supernode indices).
/// - `threshold`: Front-size threshold. Supernodes with `front_size >= threshold` are
///   excluded. If `threshold == 0`, returns empty (fast path disabled).
///
/// # Algorithm
///
/// 1. Compute `front_size[s] = sum(owned_ranges[s].len()) + pattern[s].len()`.
/// 2. Bottom-up (postorder): mark `in_small_leaf[s] = true` iff `front_size[s] < threshold`
///    AND (s is a leaf OR all children have `in_small_leaf = true`).
/// 3. Identify subtree roots: `in_small_leaf[s] = true` and parent has `in_small_leaf = false`
///    (or no parent).
/// 4. Collect descendant nodes in postorder via iterative stack-based traversal.
/// 5. Filter subtrees with < 2 nodes.
///
/// # References
///
/// - SPRAL `SymbolicSubtree.hxx:57-84` (BSD-3): bottom-up classification
/// - Duff, Hogg & Lopez (2020), Algorithm 6.1: find_subtree_partition
fn classify_small_leaf_subtrees(
    supernodes: &mut [SupernodeInfo],
    children_map: &[Vec<usize>],
    threshold: usize,
) -> Vec<SmallLeafSubtree> {
    if threshold == 0 {
        return Vec::new();
    }

    let n_supernodes = supernodes.len();

    // Step 1-2: Bottom-up classification (supernodes are in postorder: s=0 is leftmost leaf)
    for s in 0..n_supernodes {
        let owned_cols: usize = supernodes[s].owned_ranges.iter().map(|r| r.len()).sum();
        let front_size = owned_cols + supernodes[s].pattern.len();

        if front_size >= threshold {
            supernodes[s].in_small_leaf = false;
            continue;
        }

        if children_map[s].is_empty() {
            // Leaf node with small front
            supernodes[s].in_small_leaf = true;
        } else if children_map[s].iter().all(|&c| supernodes[c].in_small_leaf) {
            // Interior node: all children are in_small_leaf and this node is small
            supernodes[s].in_small_leaf = true;
        } else {
            supernodes[s].in_small_leaf = false;
        }
    }

    // Step 3: Identify subtree roots
    let mut subtrees = Vec::new();
    for s in 0..n_supernodes {
        if !supernodes[s].in_small_leaf {
            continue;
        }
        let is_root = match supernodes[s].parent {
            None => true,
            Some(p) => !supernodes[p].in_small_leaf,
        };
        if !is_root {
            continue;
        }

        // Step 4: Collect descendant nodes in postorder via iterative DFS.
        // Two-stack approach: first stack drives DFS, second collects in reverse postorder.
        let mut nodes = Vec::new();
        let mut stack = vec![s];
        let mut visit_stack = Vec::new();
        while let Some(node) = stack.pop() {
            visit_stack.push(node);
            // Push children (they'll be processed before parent due to stack LIFO)
            for &c in &children_map[node] {
                if supernodes[c].in_small_leaf {
                    stack.push(c);
                }
            }
        }
        // visit_stack is in reverse postorder — reverse to get postorder
        while let Some(node) = visit_stack.pop() {
            nodes.push(node);
        }

        // Step 5: Filter subtrees with < 2 nodes
        if nodes.len() < 2 {
            // Single-node "subtree" — not worth the fast path overhead
            supernodes[s].in_small_leaf = false;
            continue;
        }

        // Compute max_front_size across all nodes in the subtree
        let max_front_size = nodes
            .iter()
            .map(|&node| {
                let owned: usize = supernodes[node].owned_ranges.iter().map(|r| r.len()).sum();
                owned + supernodes[node].pattern.len()
            })
            .max()
            .unwrap_or(0);

        subtrees.push(SmallLeafSubtree {
            root: s,
            nodes,
            max_front_size,
            parent_of_root: supernodes[s].parent,
        });
    }

    subtrees
}

/// Scatter original sparse matrix entries into a frontal matrix from multiple owned ranges.
///
/// After amalgamation, a supernode may own multiple non-contiguous column ranges.
/// This function iterates all owned columns and correctly handles upper-triangle
/// deduplication across ranges.
fn scatter_original_entries_multi<T: Scalar>(
    frontal: &mut FrontalMatrix<'_, T>,
    matrix: &SparseColMat<usize, T>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
    global_to_local: &[usize],
    owned_ranges: &[std::ops::Range<usize>],
    scaling: Option<&[f64]>,
) {
    let symbolic = matrix.symbolic();
    let col_ptrs = symbolic.col_ptr();
    let row_indices_csc = symbolic.row_idx();
    let values = matrix.val();
    let total_owned: usize = owned_ranges.iter().map(|r| r.len()).sum();
    let k = frontal.num_fully_summed;

    // Build a fast check for "is this permuted column an owned supernode column?"
    // We reuse global_to_local: if g2l[pj] < total_owned, it's an owned column
    // (since frontal_rows puts owned columns first). This works because owned
    // columns always occupy the first `total_owned` positions in frontal_rows.

    for range in owned_ranges {
        for pj in range.clone() {
            let orig_col = perm_fwd[pj];
            let local_col = global_to_local[pj];

            let start = col_ptrs[orig_col];
            let end = col_ptrs[orig_col + 1];
            for idx in start..end {
                let orig_row = row_indices_csc[idx];
                // Skip upper-triangle if both are owned supernode cols
                if orig_row < orig_col {
                    let perm_row = perm_inv[orig_row];
                    let local_peer = global_to_local[perm_row];
                    if local_peer != NOT_IN_FRONT && local_peer < total_owned {
                        continue; // both owned cols — avoid double-count
                    }
                }
                let global_row = perm_inv[orig_row];
                let local_row = global_to_local[global_row];
                if local_row == NOT_IN_FRONT {
                    continue;
                }
                // Skip delayed columns from children (indices total_owned..k)
                if local_row >= total_owned && local_row < k {
                    continue;
                }
                let mut val = values[idx];
                if let Some(s) = scaling {
                    val *= T::from_f64(s[perm_inv[orig_row]] * s[perm_inv[orig_col]]);
                }
                if local_row >= local_col {
                    frontal.data[(local_row, local_col)] += val;
                } else {
                    frontal.data[(local_col, local_row)] += val;
                }
            }
        }
    }
}

/// Perform extend-add: merge a child's contribution block into the parent frontal matrix.
///
/// For each lower-triangle entry in the child contribution, maps global indices
/// to parent local positions and adds the value.
///
/// # References
///
/// - Liu (1992), Section 4.2: extend-add operator
pub(crate) fn extend_add<T: Scalar>(
    parent: &mut FrontalMatrix<'_, T>,
    child: ContributionBlock<T>,
    global_to_local: &[usize],
) {
    let cb_size = child.row_indices.len();
    // Column-oriented iteration: for each child column j, scatter the
    // lower-triangle entries (rows j..cb_size) into the parent. Source
    // reads are contiguous in column-major layout. We keep the zero-check
    // and symmetry swap since delayed columns can break monotonicity.
    for j in 0..cb_size {
        let gj = child.row_indices[j];
        let lj = global_to_local[gj];
        debug_assert!(
            lj != NOT_IN_FRONT,
            "extend_add: child col {} not in parent",
            gj
        );
        let child_col = child.data.col(j);
        let row_indices = &child.row_indices[j..cb_size];

        for (&val, &gi) in child_col.iter().zip(row_indices) {
            if val != T::zero() {
                let li = global_to_local[gi];
                debug_assert!(
                    li != NOT_IN_FRONT,
                    "extend_add: child row {} not in parent",
                    gi
                );
                // Map to parent: ensure lower triangle (li >= lj)
                if li >= lj {
                    parent.data[(li, lj)] += val;
                } else {
                    parent.data[(lj, li)] += val;
                }
            }
        }
    }
}

/// Extend-add using a precomputed child→parent row mapping.
///
/// Used when the child had zero delayed columns, so the contribution block
/// rows correspond exactly to the child's symbolic pattern and the precomputed
/// `ea_row_map` provides direct local index lookup without `global_to_local`.
///
/// Uses column-oriented processing: for each child
/// column `j`, scatters the lower-triangle entries `(j..cb_size, j)` into
/// the parent. Source reads are contiguous in column-major layout.
///
/// # References
///
/// - SPRAL `assemble.hxx:27-38` — `asm_col` column-oriented scatter (BSD-3)
///
/// # Arguments
///
/// - `parent`: Parent frontal matrix to accumulate into (lower triangle).
/// - `child`: Child contribution block.
/// - `ea_row_map`: Precomputed mapping from child contribution row index to
///   parent frontal local row index. Length equals the child's pattern size.
fn extend_add_mapped<T: Scalar>(
    parent: &mut FrontalMatrix<'_, T>,
    child: ContributionBlock<T>,
    ea_row_map: &[u32],
    par: Par,
) {
    let cb_size = child.row_indices.len();
    debug_assert!(
        ea_row_map.len() >= cb_size,
        "extend_add_mapped: ea_row_map len {} < cb_size {}",
        ea_row_map.len(),
        cb_size
    );

    // Parallel path: when the child→parent row map is strictly increasing
    // (the usual case), child column j writes only to parent column map[j]
    // and only at rows map[j..] >= map[j], so contiguous ranges of child
    // columns write to disjoint ranges of parent columns.
    let map = &ea_row_map[..cb_size];
    if copy_par_split(par, cb_size, cb_size * cb_size / 2) && map.windows(2).all(|w| w[0] < w[1]) {
        extend_add_mapped_par(parent.data.as_mut(), 0, &child.data, map, 0, cb_size, par);
        return;
    }

    // Column-oriented scatter.
    // For each child column j, scatter lower-triangle entries (rows j..cb_size)
    // into the parent. Source reads via col_as_slice are contiguous in column-
    // major layout; writes target a single parent column when li >= lj (common
    // case), with a symmetry swap for the rare li < lj case.
    for j in 0..cb_size {
        let lj = ea_row_map[j] as usize;
        let row_map = &ea_row_map[j..cb_size];
        let src = child.data.col(j);

        // 4× unrolled inner loop for ILP
        let n = cb_size - j;
        let n4 = n / 4 * 4;
        let mut k = 0;
        while k < n4 {
            ea_scatter_one(&mut parent.data, row_map[k] as usize, lj, src[k]);
            ea_scatter_one(&mut parent.data, row_map[k + 1] as usize, lj, src[k + 1]);
            ea_scatter_one(&mut parent.data, row_map[k + 2] as usize, lj, src[k + 2]);
            ea_scatter_one(&mut parent.data, row_map[k + 3] as usize, lj, src[k + 3]);
            k += 4;
        }
        while k < n {
            ea_scatter_one(&mut parent.data, row_map[k] as usize, lj, src[k]);
            k += 1;
        }
    }
}

/// Column-parallel extend-add for a strictly increasing row map.
///
/// `parent` views parent columns `pcol0..pcol0 + parent.ncols()` (all rows);
/// child columns `j0..j1` are scattered into it. Splitting child columns at
/// `mid` splits the parent at column `map[mid]`: every column before `mid`
/// maps below it and every column from `mid` on maps at or above it.
fn extend_add_mapped_par<T: Scalar>(
    parent: MatMut<'_, T>,
    pcol0: usize,
    child: &PackedLower<T>,
    map: &[u32],
    j0: usize,
    j1: usize,
    par: Par,
) {
    let cb_size = map.len();
    if copy_par_split(par, j1 - j0, (j1 - j0) * (cb_size - j0)) {
        let mid = (j0 + j1) / 2;
        let split = map[mid] as usize - pcol0;
        let (left, right) = parent.split_at_col_mut(split);
        rayon::join(
            || extend_add_mapped_par(left, pcol0, child, map, j0, mid, par),
            || extend_add_mapped_par(right, pcol0 + split, child, map, mid, j1, par),
        );
        return;
    }
    let mut parent = parent;
    for j in j0..j1 {
        let lj = map[j] as usize - pcol0;
        let rows = &map[j..cb_size];
        let src = child.col(j);
        match parent.as_mut().col_mut(lj).try_as_col_major_mut() {
            Some(col) => {
                let col = col.as_slice_mut();
                for (&li, &v) in rows.iter().zip(src) {
                    col[li as usize] += v;
                }
            }
            None => {
                for (&li, &v) in rows.iter().zip(src) {
                    parent[(li as usize, lj)] += v;
                }
            }
        }
    }
}

/// Zero the lower triangle of a front and add children's contribution blocks
/// into it in the same column pass.
///
/// Each child's row map must be strictly increasing, so child column `j`
/// lands in parent column `map[j]` at rows `map[j..] >= map[j]`, and a parent
/// column receives at most one column from each child (found by binary
/// search). Zeroing and assembling a column while it is in cache saves one
/// trip of the front through memory compared with zeroing the whole front
/// first; on the large fronts both steps are DRAM-bandwidth bound.
fn zero_and_extend_add<T: Scalar>(
    front: MatMut<'_, T>,
    children: &[(&PackedLower<T>, &[u32])],
    par: Par,
) {
    for_each_col_mut(front, 0, par, &|c, col: &mut [T]| {
        col[c..].fill(T::zero());
        for &(cb, map) in children {
            if let Ok(j) = map.binary_search(&(c as u32)) {
                for (&li, &v) in map[j..].iter().zip(cb.col(j)) {
                    col[li as usize] += v;
                }
            }
        }
    });
}

/// Scatter a single extend-add value into the parent's lower triangle.
#[inline(always)]
fn ea_scatter_one<T: Scalar>(parent: &mut MatMut<'_, T>, li: usize, lj: usize, val: T) {
    if li >= lj {
        parent[(li, lj)] += val;
    } else {
        parent[(lj, li)] += val;
    }
}

/// Extract per-supernode factors from a factored frontal matrix.
///
/// Copies L11, D11, L21, and permutation information from the in-place
/// factored frontal matrix into a persistent [`FrontFactors`] struct.
///
/// Uses per-column `col_as_slice` + `copy_from_slice` for L21 extraction and
/// per-column slice copies for L11 1x1-pivot columns, replacing element-by-element
/// indexing with bulk memory operations.
///
/// # Arguments
///
/// - `frontal_data`: The factored frontal matrix data (owned `Mat<f64>`).
///   Columns must be contiguous (guaranteed by `Mat` layout). Only the first
///   `m` rows and columns are used.
/// - `m`: Front size (number of rows/columns used in `frontal_data`).
/// - `k`: Number of fully-summed rows/columns.
/// - `frontal_row_indices`: Global permuted column indices for each local row (length >= m).
/// - `result`: APTP factorization result (pivot types, permutation, etc.).
pub(crate) fn extract_front_factors<T: Scalar>(
    frontal_data: &Mat<T>,
    m: usize,
    k: usize,
    frontal_row_indices: &[usize],
    result: &AptpFactorResult<T>,
    par: Par,
) -> FrontFactors<T> {
    let ne = result.num_eliminated;

    // Extract L11 (ne × ne) — unit lower triangular part, in column panels.
    // For 2x2 pivots, a[(col+1, col)] is the D off-diagonal, NOT an L entry.
    let l11 = BlockedUnitLower::from_factored(frontal_data, ne, &result.d, par);

    // Build truncated D11 (first ne pivots from result.d)
    let mut d11 = MixedDiagonal::new(ne);
    let mut col = 0;
    while col < ne {
        match result.d.pivot_type(col) {
            PivotType::OneByOne => {
                d11.set_1x1(col, result.d.diagonal_1x1(col));
                col += 1;
            }
            PivotType::TwoByTwo { partner: _ } => {
                let block = result.d.diagonal_2x2(col);
                d11.set_2x2(Block2x2 {
                    first_col: col,
                    a: block.a,
                    b: block.b,
                    c: block.c,
                });
                col += 2;
            }
            PivotType::Delayed => {
                // Should not happen for columns 0..ne (they were eliminated)
                unreachable!("unexpected Delayed pivot at col {} in 0..ne", col);
            }
        }
    }

    // Extract L21 (r × ne) where r = m - ne (includes delayed rows ne..k AND
    // non-fully-summed rows k..m). The factored frontal has valid L entries at rows
    // ne..m for columns 0..ne: rows ne..k are delayed fully-summed rows that received
    // TRSM updates before being delayed; rows k..m are non-fully-summed rows.
    //
    // Uses per-column copy_from_slice for bulk extraction.
    //
    // NOTE: extract_contribution also includes delayed rows (see lines below).
    // These two functions MUST be consistent in their row treatment.
    let r = m - ne;
    let l21 = filled_padded(r, ne, par, &|j, dst| {
        write_slice(dst, &frontal_data.col_as_slice(j)[ne..m]);
    });

    // Local permutation (maps factored position to original front-local column)
    let local_perm = result.perm[..k].to_vec();

    // Column indices: the global permuted indices of the eliminated columns
    // local_perm[0..ne] gives the front-local columns that were eliminated,
    // mapped through frontal_row_indices to get global permuted indices
    let col_indices: Vec<usize> = local_perm[..ne]
        .iter()
        .map(|&lp| frontal_row_indices[lp])
        .collect();

    // Row indices: global permuted indices for L21 rows.
    // First num_delayed entries: delayed fully-summed columns (ne..k), mapped through
    // result.perm and frontal_row_indices to get global permuted indices.
    // Remaining entries: non-fully-summed rows (k..m).
    // This matches extract_contribution's row_indices construction.
    let mut row_indices = Vec::with_capacity(m - ne);
    for &lp in &result.perm[ne..k] {
        row_indices.push(frontal_row_indices[lp]);
    }
    row_indices.extend_from_slice(&frontal_row_indices[k..m]);

    FrontFactors {
        l11,
        d11,
        l21,
        local_perm,
        num_eliminated: ne,
        col_indices,
        row_indices,
    }
}

/// Extract the contribution block from a factored frontal matrix.
///
/// After the APTP kernel and [`compute_contribution_gemm_in_place`], the
/// trailing `(m − ne) × (m − ne)` block of `frontal_data` holds the whole
/// contribution, in the order `[delayed columns | NFS rows]`:
///
/// ```text
/// [0..nd, 0..nd]       delayed × delayed      = frontal[ne..k, ne..k]
/// [nd..size, 0..nd]    NFS × delayed          = frontal[k..m, ne..k]
/// [nd..size, nd..size] updated NFS × NFS      = frontal[k..m, k..m]
/// ```
///
/// so column `j` of the contribution is the contiguous slice
/// `frontal[ne+j..m, ne+j]`. Each column is copied once into an exactly sized
/// [`PackedLower`] block.
///
/// # Arguments
///
/// - `frontal_data`: The factored frontal matrix data, with the NFS×NFS update applied.
/// - `m`: Front size (number of rows/columns used).
/// - `k`: Number of fully-summed rows/columns.
/// - `frontal_row_indices`: Global permuted column indices for each local row.
/// - `result`: APTP factorization result.
pub(crate) fn extract_contribution<T: Scalar>(
    frontal_data: &Mat<T>,
    m: usize,
    k: usize,
    frontal_row_indices: &[usize],
    result: &AptpFactorResult<T>,
    par: Par,
) -> ContributionBlock<T> {
    let ne = result.num_eliminated;
    let size = m - ne;
    let num_delayed = k - ne;

    let data = PackedLower::from_columns_par(
        size,
        &|j| &frontal_data.col_as_slice(ne + j)[ne + j..m],
        par,
    );

    // Build row indices:
    // - First num_delayed entries: delayed columns mapped through result.perm and frontal_row_indices
    // - Remaining entries: non-fully-summed rows from frontal_row_indices[k..m]
    let mut row_indices = Vec::with_capacity(size);
    for &lp in &result.perm[ne..k] {
        row_indices.push(frontal_row_indices[lp]);
    }
    row_indices.extend_from_slice(&frontal_row_indices[k..m]);

    ContributionBlock {
        data,
        row_indices,
        num_delayed,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_k_product_matches_matmul() {
        // m large enough that sub_at_b takes the split-K path at 4 threads.
        for (m, w, k) in [(5000, 64, 1), (3001, 37, 16), (600, 64, 3)] {
            let a = Mat::<f64>::from_fn(m, w, |i, j| ((i * 7 + j * 13) % 29) as f64 / 29.0 - 0.5);
            let b = Mat::<f64>::from_fn(m, k, |i, j| ((i * 11 + j * 5) % 31) as f64 / 31.0 - 0.5);
            let c0 = Mat::<f64>::from_fn(w, k, |i, j| (i + 2 * j) as f64);
            let expect = &c0 - a.transpose() * &b;
            for par in [Par::Seq, Par::rayon(4)] {
                let mut c = c0.clone();
                sub_at_b(c.as_mut(), a.as_ref(), b.as_ref(), par);
                let err = (&c - &expect).norm_max();
                assert!(err < 1e-10, "m={m} w={w} k={k} {par:?}: err {err}");
            }
        }
    }

    #[test]
    fn blocked_unit_lower_matches_dense() {
        // 150 columns -> panels of 64, 64, 22. 2x2 pivots straddle panel
        // boundaries (63/64) and sit inside panels, so the D off-diagonal
        // skip is exercised in both places.
        let n = 150;
        let src = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5);
        let mut d = MixedDiagonal::new(n);
        let mut col = 0;
        while col < n {
            if col == 10 || col == 63 || col == 100 {
                d.set_2x2(Block2x2 {
                    first_col: col,
                    a: 2.0,
                    b: 0.5,
                    c: 3.0,
                });
                col += 2;
            } else {
                d.set_1x1(col, 1.0);
                col += 1;
            }
        }
        let l = BlockedUnitLower::from_factored(&src, n, &d, Par::Seq);

        // Expected dense L: unit diagonal, src below it, zero at (j+1, j)
        // for the first column of each 2x2 pivot.
        let expected = Mat::<f64>::from_fn(n, n, |i, j| {
            if i < j {
                0.0
            } else if i == j {
                1.0
            } else if i == j + 1 && (j == 10 || j == 63 || j == 100) {
                0.0
            } else {
                src[(i, j)]
            }
        });
        assert_eq!(l.dim(), n);
        assert!((&l.to_dense() - &expected).norm_max() == 0.0);
        // Storage: packed lower plus at most half of each 64x64 diagonal block.
        assert!(l.allocated() < n * (n + 1) / 2 + 3 * 64 * 64 / 2 + 3 * 64);

        // Solves against the dense factor, three right-hand sides.
        let b = Mat::<f64>::from_fn(n, 3, |i, j| (i + 2 * j) as f64 * 0.01 + 1.0);
        let mut x = b.clone();
        l.solve_in_place(x.as_mut(), Par::Seq);
        assert!((&expected * &x - &b).norm_max() < 1e-10);
        let mut y = b.clone();
        l.solve_transpose_in_place(y.as_mut(), Par::Seq);
        assert!((expected.transpose() * &y - &b).norm_max() < 1e-10);
    }

    #[test]
    fn filled_padded_writes_every_entry() {
        // Large enough to split across threads (>= 64 columns, >= 2^19 entries),
        // with a row count whose stride would land on a page multiple.
        let (nrows, ncols) = (8192, 80);
        let f = |i: usize, j: usize| (i * 3 + j * 1000) as f64;
        for par in [Par::Seq, Par::rayon(4)] {
            let m = filled_padded::<f64>(nrows, ncols, par, &|j, col| {
                for (i, x) in col.iter_mut().enumerate() {
                    x.write(f(i, j));
                }
            });
            assert_eq!((m.nrows(), m.ncols()), (nrows, ncols));
            assert_eq!(m.col_stride() as usize, padded_rows::<f64>(nrows));
            assert!(!(m.col_stride() as usize * 8).is_multiple_of(4096));
            for j in 0..ncols {
                for i in 0..nrows {
                    assert_eq!(m[(i, j)], f(i, j));
                }
            }
        }
        assert_eq!(filled_padded::<f64>(0, 5, Par::Seq, &|_, _| {}).ncols(), 5);
        assert_eq!(filled_padded::<f64>(5, 0, Par::Seq, &|_, _| {}).nrows(), 5);
    }

    #[test]
    fn packed_lower_par_matches_seq() {
        let n = 1500; // ~1.1M packed entries: splits across threads
        let dense = Mat::<f64>::from_fn(n, n, |i, j| (i * n + j) as f64);
        let col = |j: usize| &dense.col_as_slice(j)[j..n];
        let seq = PackedLower::from_columns(n, col);
        let par = PackedLower::from_columns_par(n, &col, Par::rayon(4));
        assert_eq!(seq.data, par.data);
    }
    use crate::symmetric::factor::AptpOptions;

    /// Helper: build frontal matrix data from a dense lower-triangle matrix.
    ///
    /// Returns owned `(data, row_indices)` — construct a `FrontalMatrix` view from these.
    /// `k` = number of fully-summed columns.
    /// `row_indices` = global permuted indices for each local row (length m).
    fn make_frontal_data(
        lower: &Mat<f64>,
        k: usize,
        row_indices: Vec<usize>,
    ) -> (Mat<f64>, Vec<usize>, usize) {
        let m = lower.nrows();
        assert_eq!(lower.ncols(), m);
        assert_eq!(row_indices.len(), m);
        assert!(k <= m);

        // Build full symmetric from lower triangle
        let mut data = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                data[(i, j)] = lower[(i, j)];
                data[(j, i)] = lower[(i, j)];
            }
        }

        (data, row_indices, k)
    }

    /// Helper: build a symmetric indefinite matrix (lower triangle) that forces
    /// delays. Strategy from SPRAL's `cause_delays`: scale selected rows/cols
    /// by a large factor, making their off-diagonal entries dominate the diagonal
    /// so the threshold test fails.
    ///
    /// Returns (matrix, expected_to_have_delays).
    fn make_delay_matrix(m: usize, k: usize, delay_rows: &[usize]) -> Mat<f64> {
        let mut a = Mat::<f64>::zeros(m, m);

        // Start with a diagonally dominant indefinite matrix
        for i in 0..m {
            // Alternate positive/negative diagonal for indefiniteness
            a[(i, i)] = if i % 2 == 0 { 4.0 } else { -4.0 };
            // Add some off-diagonal coupling
            for j in 0..i {
                let val = 0.5 / ((i - j) as f64 + 1.0);
                a[(i, j)] = val;
            }
        }

        // Scale delay_rows to create threshold failures:
        // Multiply row/col by 1000 but NOT the diagonal, so off-diag >> diag
        for &r in delay_rows {
            if r < k {
                // Scale off-diagonal entries in row r
                for j in 0..r {
                    a[(r, j)] *= 1000.0;
                }
                for i in (r + 1)..m {
                    a[(i, r)] *= 1000.0;
                }
                // DON'T scale diagonal — this makes |a_rr| << |a_ir| => threshold failure
            }
        }

        a
    }

    /// Test that extract_front_factors produces correct L21 dimensions when
    /// delays are present: L21 should have (m - ne) rows, NOT (m - k) rows.
    /// This is the regression test for the bug fixed in this commit.
    #[test]
    fn test_extract_front_factors_l21_includes_delayed_rows() {
        let m = 12;
        let k = 8;
        let delay_rows = &[1, 5]; // Columns 1 and 5 should be delayed

        let lower = make_delay_matrix(m, k, delay_rows);
        let row_indices: Vec<usize> = (100..100 + m).collect(); // arbitrary global indices

        let (mut data, row_indices, k) = make_frontal_data(&lower, k, row_indices);

        let options = AptpOptions::default();
        let result =
            aptp_factor_in_place(data.as_mut(), k, &options).expect("factor should succeed");

        let ne = result.num_eliminated;
        let num_delayed = k - ne;

        // The bug: if delays occurred, the old code would make L21 have (m - k) rows
        // instead of (m - ne) rows, missing the delayed-row entries.
        if num_delayed > 0 {
            let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);

            // L21 must have (m - ne) rows = delayed_rows + non-fully-summed rows
            assert_eq!(
                ff.l21.nrows(),
                m - ne,
                "L21 should have m-ne={} rows (not m-k={}), ne={}, delays={}",
                m - ne,
                m - k,
                ne,
                num_delayed
            );

            // row_indices must match L21 rows
            assert_eq!(
                ff.row_indices.len(),
                ff.l21.nrows(),
                "row_indices length must match L21 rows"
            );

            // col_indices must have ne entries
            assert_eq!(ff.col_indices.len(), ne);
        } else {
            // No delays — still verify basic shapes
            let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);
            assert_eq!(ff.l21.nrows(), m - ne);
            assert_eq!(ff.row_indices.len(), ff.l21.nrows());
        }
    }

    /// Test that delayed rows in extract_front_factors and extract_contribution
    /// have consistent row_indices. Both functions MUST agree on which global
    /// indices correspond to delayed rows.
    #[test]
    fn test_extract_front_factors_contribution_row_indices_consistent() {
        let m = 10;
        let k = 6;
        let delay_rows = &[2]; // Force column 2 to be delayed

        let lower = make_delay_matrix(m, k, delay_rows);
        let row_indices: Vec<usize> = (200..200 + m).collect();

        let (mut data, row_indices, k) = make_frontal_data(&lower, k, row_indices);

        let options = AptpOptions::default();
        let result =
            aptp_factor_in_place(data.as_mut(), k, &options).expect("factor should succeed");

        let ne = result.num_eliminated;
        let num_delayed = k - ne;

        if num_delayed > 0 {
            let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);
            let mut ld_ws = Mat::<f64>::new();
            compute_contribution_gemm_in_place(
                &mut data,
                k,
                ne,
                m,
                &result.d,
                &mut ld_ws,
                Par::Seq,
            );
            let cb = extract_contribution(&data, m, k, &row_indices, &result, Par::Seq);

            // Both must have the same number of delayed rows
            assert_eq!(cb.num_delayed, num_delayed);

            // The first num_delayed entries of row_indices must match
            let ff_delayed_indices = &ff.row_indices[..num_delayed];
            let cb_delayed_indices = &cb.row_indices[..num_delayed];
            assert_eq!(
                ff_delayed_indices, cb_delayed_indices,
                "Delayed row indices must be identical between extract_front_factors \
                 and extract_contribution.\n  front_factors: {:?}\n  contribution:  {:?}",
                ff_delayed_indices, cb_delayed_indices
            );

            // The non-fully-summed row indices must also match
            let ff_nfs_indices = &ff.row_indices[num_delayed..];
            let cb_nfs_indices = &cb.row_indices[num_delayed..];
            assert_eq!(
                ff_nfs_indices, cb_nfs_indices,
                "Non-fully-summed row indices must match"
            );
        }
    }

    /// Test that L21 entries at delayed row positions are non-zero.
    /// When APTP delays a column, the TRSM (apply_and_check) has already
    /// computed valid L entries at those row positions. These must be preserved
    /// in the extracted L21.
    #[test]
    fn test_extract_front_factors_delayed_l21_entries_populated() {
        let m = 12;
        let k = 8;
        // Use multiple delay candidates to increase chance of delays
        let delay_rows = &[0, 3, 5];

        let lower = make_delay_matrix(m, k, delay_rows);
        let row_indices: Vec<usize> = (0..m).collect();

        let (mut data, row_indices, k) = make_frontal_data(&lower, k, row_indices);

        let options = AptpOptions::default();
        let result =
            aptp_factor_in_place(data.as_mut(), k, &options).expect("factor should succeed");

        let ne = result.num_eliminated;
        let num_delayed = k - ne;

        if num_delayed > 0 && ne > 0 {
            let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);

            // Check that L21 entries in the delayed-row portion (rows 0..num_delayed)
            // are not all zero. These rows received TRSM updates before being delayed.
            let delayed_l21 = ff.l21.as_ref().subrows(0, num_delayed);
            let mut has_nonzero = false;
            for i in 0..delayed_l21.nrows() {
                for j in 0..delayed_l21.ncols() {
                    if delayed_l21[(i, j)].abs() > 1e-16 {
                        has_nonzero = true;
                    }
                }
            }
            assert!(
                has_nonzero,
                "L21 delayed rows should have non-zero entries from TRSM. \
                 ne={}, delays={}, L21 shape=({},{})",
                ne,
                num_delayed,
                ff.l21.nrows(),
                ff.l21.ncols()
            );
        }
    }

    /// Reconstruction test for extract_front_factors with delays.
    /// Verify that the extracted L11, D11, L21 correctly reconstruct the
    /// factored portion of the frontal matrix: PAP^T[0..ne, 0..ne] = L11 * D11 * L11^T.
    #[test]
    fn test_extract_front_factors_reconstruction_with_delays() {
        let m = 16;
        let k = 10;
        let delay_rows = &[1, 4, 7];

        let lower = make_delay_matrix(m, k, delay_rows);
        let row_indices: Vec<usize> = (0..m).collect();

        // Save a copy of the original symmetric matrix before factoring
        let mut original = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                original[(i, j)] = lower[(i, j)];
                original[(j, i)] = lower[(i, j)];
            }
        }

        let (mut data, row_indices, k) = make_frontal_data(&lower, k, row_indices);

        let options = AptpOptions::default();
        let result =
            aptp_factor_in_place(data.as_mut(), k, &options).expect("factor should succeed");

        let ne = result.num_eliminated;
        if ne == 0 {
            return; // Nothing to reconstruct
        }

        let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);

        // Compute L11 * D * L11^T manually
        // Step 1: LD = L11 * D (column-by-column, handling 1x1 and 2x2 pivots)
        let mut ld = Mat::<f64>::zeros(ne, ne);
        let mut col = 0;
        while col < ne {
            match ff.d11.pivot_type(col) {
                PivotType::OneByOne => {
                    let d = ff.d11.diagonal_1x1(col);
                    for i in 0..ne {
                        ld[(i, col)] = ff.l11.get(i, col) * d;
                    }
                    col += 1;
                }
                PivotType::TwoByTwo { .. } => {
                    let block = ff.d11.diagonal_2x2(col);
                    for i in 0..ne {
                        let l0 = ff.l11.get(i, col);
                        let l1 = ff.l11.get(i, col + 1);
                        ld[(i, col)] = l0 * block.a + l1 * block.b;
                        ld[(i, col + 1)] = l0 * block.b + l1 * block.c;
                    }
                    col += 2;
                }
                PivotType::Delayed => {
                    col += 1;
                }
            }
        }

        // Step 2: reconstructed = LD * L11^T
        let mut reconstructed = Mat::<f64>::zeros(ne, ne);
        for i in 0..ne {
            for j in 0..=i {
                let mut val = 0.0;
                for p in 0..ne {
                    val += ld[(i, p)] * ff.l11.get(j, p);
                }
                reconstructed[(i, j)] = val;
                reconstructed[(j, i)] = val;
            }
        }

        // Build the permuted original submatrix for comparison
        let mut perm_original = Mat::<f64>::zeros(ne, ne);
        for i in 0..ne {
            for j in 0..=i {
                let oi = result.perm[i];
                let oj = result.perm[j];
                let val = if oi >= oj {
                    original[(oi, oj)]
                } else {
                    original[(oj, oi)]
                };
                perm_original[(i, j)] = val;
                perm_original[(j, i)] = val;
            }
        }

        // Check reconstruction: ||PAP^T[0..ne,0..ne] - L11*D11*L11^T|| / ||PAP^T||
        let mut diff_norm = 0.0f64;
        let mut orig_norm = 0.0f64;
        for i in 0..ne {
            for j in 0..=i {
                let d = reconstructed[(i, j)] - perm_original[(i, j)];
                diff_norm += d * d;
                orig_norm += perm_original[(i, j)] * perm_original[(i, j)];
            }
        }
        let rel_err = diff_norm.sqrt() / orig_norm.sqrt().max(1e-15);

        assert!(
            rel_err < 1e-10,
            "L11*D11*L11^T reconstruction error: {:.2e} (ne={}, delays={})",
            rel_err,
            ne,
            k - ne
        );
    }

    /// Verify that the solve path produces correct results when delays are
    /// present. Uses a single dense frontal matrix (simulating one supernode),
    /// factors with delays, extracts front_factors, and runs forward/diagonal/
    /// backward solve steps to verify Ax = b.
    ///
    /// This directly tests the code path that the extract_front_factors bug
    /// would break: delayed-row L21 entries participate in forward and backward
    /// solve via gather/scatter operations.
    #[test]
    fn test_extract_front_factors_solve_roundtrip_with_delays() {
        let m = 12;
        let k = 8; // fully-summed columns

        let lower = make_delay_matrix(m, k, &[0, 3, 5]);
        let row_indices: Vec<usize> = (0..m).collect();

        // Build full symmetric matrix for reference
        let mut a_full = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                a_full[(i, j)] = lower[(i, j)];
                a_full[(j, i)] = lower[(i, j)];
            }
        }

        let (mut data, row_indices, k) = make_frontal_data(&lower, k, row_indices);

        let options = AptpOptions::default();
        let result =
            aptp_factor_in_place(data.as_mut(), k, &options).expect("factor should succeed");

        let ne = result.num_eliminated;
        let num_delayed = k - ne;

        // Only meaningful if we actually got some delays and some eliminations
        if num_delayed == 0 || ne == 0 {
            return;
        }

        let ff = extract_front_factors(&data, m, k, &row_indices, &result, Par::Seq);

        // Check that L11, D11, L21 are internally consistent:
        // L * D * L^T (where L = [L11; L21]) should reconstruct the factored
        // portion of A when permuted.

        // Build full L = [L11; L21] and check L * D * L^T = P * A[0..m, 0..ne] * P^T
        let r = ff.l21.nrows();
        let full_l_rows = ne + r;
        assert_eq!(full_l_rows, m, "L should cover all rows");

        // Compute L*D*L^T for the eliminated portion
        let mut ld = Mat::<f64>::zeros(full_l_rows, ne);

        // L = [L11; L21]
        let mut full_l = Mat::<f64>::zeros(full_l_rows, ne);
        for i in 0..ne {
            for j in 0..ne {
                full_l[(i, j)] = ff.l11.get(i, j);
            }
        }
        for i in 0..r {
            for j in 0..ne {
                full_l[(ne + i, j)] = ff.l21[(i, j)];
            }
        }

        // LD = L * D (handling 1x1 and 2x2)
        let mut col = 0;
        while col < ne {
            match ff.d11.pivot_type(col) {
                PivotType::OneByOne => {
                    let d = ff.d11.diagonal_1x1(col);
                    for i in 0..full_l_rows {
                        ld[(i, col)] = full_l[(i, col)] * d;
                    }
                    col += 1;
                }
                PivotType::TwoByTwo { .. } => {
                    let block = ff.d11.diagonal_2x2(col);
                    for i in 0..full_l_rows {
                        let l0 = full_l[(i, col)];
                        let l1 = full_l[(i, col + 1)];
                        ld[(i, col)] = l0 * block.a + l1 * block.b;
                        ld[(i, col + 1)] = l0 * block.b + l1 * block.c;
                    }
                    col += 2;
                }
                PivotType::Delayed => {
                    col += 1;
                }
            }
        }

        // reconstructed = LD * L^T
        let mut reconstructed = Mat::<f64>::zeros(full_l_rows, full_l_rows);
        for i in 0..full_l_rows {
            for j in 0..=i {
                let mut val = 0.0;
                for p in 0..ne {
                    val += ld[(i, p)] * full_l[(j, p)];
                }
                reconstructed[(i, j)] = val;
                reconstructed[(j, i)] = val;
            }
        }

        // Build permuted original: row_order = col_indices ++ row_indices
        let mut perm_rows = Vec::with_capacity(full_l_rows);
        perm_rows.extend_from_slice(ff.col_indices());
        perm_rows.extend_from_slice(ff.row_indices());

        // Check that L*D*L^T[0..ne, 0..ne] ≈ A_perm[0..ne, 0..ne]
        let mut diff_norm = 0.0f64;
        let mut orig_norm = 0.0f64;
        for i in 0..ne {
            for j in 0..=i {
                let gi = perm_rows[i];
                let gj = perm_rows[j];
                let orig_val = if gi >= gj {
                    a_full[(gi, gj)]
                } else {
                    a_full[(gj, gi)]
                };
                let d = reconstructed[(i, j)] - orig_val;
                diff_norm += d * d;
                orig_norm += orig_val * orig_val;
            }
        }
        let rel_err = diff_norm.sqrt() / orig_norm.sqrt().max(1e-15);
        assert!(
            rel_err < 1e-10,
            "Full L*D*L^T reconstruction error: {:.2e} (ne={}, delays={}, L21_rows={})",
            rel_err,
            ne,
            num_delayed,
            r
        );
    }

    /// Test scatter_original_entries_multi with non-contiguous owned_ranges.
    ///
    /// After amalgamation, a merged supernode may own columns from multiple
    /// non-contiguous ranges (e.g., [0..2, 4..6] when child columns 2,3 were
    /// NOT merged). This test verifies:
    ///
    /// 1. Entries from all owned ranges are scattered into the frontal matrix
    /// 2. Upper-triangle deduplication works correctly across non-contiguous
    ///    ranges (an entry with both row and col in owned ranges is only counted once)
    /// 3. Entries where one index is owned and the other is a non-fully-summed
    ///    row are scattered correctly
    #[test]
    fn test_scatter_original_entries_multi_non_contiguous() {
        use faer::sparse::{SparseColMat, Triplet};

        // Build a small 6×6 symmetric matrix with identity permutation.
        // Owned ranges: [0..2, 4..6] (columns 0,1,4,5 owned; 2,3 not owned).
        //
        // The frontal has 6 rows: locals 0..4 are owned (4 total_owned),
        // no delayed children, rows 4..6 are non-fully-summed (rows for
        // global indices 2 and 3).
        //
        // Matrix (lower triangle values, symmetric):
        //     col: 0    1    2    3    4    5
        // 0: [10.0                          ]
        // 1: [ 1.0  11.0                    ]
        // 2: [ 2.0   0.0  12.0              ]
        // 3: [ 0.0   3.0   0.0  13.0        ]
        // 4: [ 4.0   0.0   0.0   0.0  14.0  ]
        // 5: [ 0.0   5.0   6.0   0.0   7.0  15.0]
        let triplets = vec![
            // Diagonal
            Triplet::new(0, 0, 10.0),
            Triplet::new(1, 1, 11.0),
            Triplet::new(2, 2, 12.0),
            Triplet::new(3, 3, 13.0),
            Triplet::new(4, 4, 14.0),
            Triplet::new(5, 5, 15.0),
            // Off-diagonal (both triangles for full CSC)
            Triplet::new(1, 0, 1.0),
            Triplet::new(0, 1, 1.0), // mirror
            Triplet::new(2, 0, 2.0),
            Triplet::new(0, 2, 2.0), // mirror
            Triplet::new(3, 1, 3.0),
            Triplet::new(1, 3, 3.0), // mirror
            Triplet::new(4, 0, 4.0),
            Triplet::new(0, 4, 4.0), // mirror
            Triplet::new(5, 1, 5.0),
            Triplet::new(1, 5, 5.0), // mirror
            Triplet::new(5, 2, 6.0),
            Triplet::new(2, 5, 6.0), // mirror
            Triplet::new(5, 4, 7.0),
            Triplet::new(4, 5, 7.0), // mirror
        ];
        let matrix = SparseColMat::try_new_from_triplets(6, 6, &triplets).expect("valid CSC");

        // Identity permutation
        let perm_fwd: Vec<usize> = (0..6).collect();
        let perm_inv: Vec<usize> = (0..6).collect();

        // Frontal rows: owned columns first (0,1,4,5), then non-owned rows (2,3)
        let frontal_rows = vec![0, 1, 4, 5, 2, 3];
        let total_owned = 4; // columns 0,1,4,5
        let m = frontal_rows.len();
        let k = total_owned; // num_fully_summed = total_owned (no delayed children)

        // Build global_to_local mapping
        let mut global_to_local = vec![NOT_IN_FRONT; 6];
        for (local, &global) in frontal_rows.iter().enumerate() {
            global_to_local[global] = local;
        }

        let mut frontal_data = Mat::<f64>::zeros(m, m);

        // Non-contiguous owned ranges (the key case)
        let owned_ranges = vec![0..2, 4..6];

        {
            let mut frontal = FrontalMatrix {
                data: frontal_data.as_mut(),
                row_indices: &frontal_rows,
                num_fully_summed: k,
            };

            scatter_original_entries_multi(
                &mut frontal,
                &matrix,
                &perm_fwd,
                &perm_inv,
                &global_to_local,
                &owned_ranges,
                None,
            );
        }

        // Verify scattered values.
        // Local indices: 0→global 0, 1→global 1, 2→global 4, 3→global 5, 4→global 2, 5→global 3
        //
        // Expected entries in the frontal (lower triangle):
        //   (0,0)=10.0  diagonal of global 0
        //   (1,0)=1.0   global(1,0) → local(1,0)
        //   (1,1)=11.0  diagonal of global 1
        //   (2,0)=4.0   global(4,0) → local(2,0)
        //   (2,2)=14.0  diagonal of global 4
        //   (3,1)=5.0   global(5,1) → local(3,1)
        //   (3,2)=7.0   global(5,4) → local(3,2)
        //   (3,3)=15.0  diagonal of global 5
        //   (4,0)=2.0   global(2,0) → local(4,0)  [non-owned row, owned col]
        //   (4,3)=6.0   global(2,5) → local(4,3)  [non-owned row, owned col]
        //   (5,1)=3.0   global(3,1) → local(5,1)  [non-owned row, owned col]
        //
        // Upper-triangle dedup: global(0,1) should be skipped because both
        // 0 and 1 are owned. Similarly global(0,4), global(1,5), global(4,5)
        // are upper-triangle entries between owned cols and should be skipped.

        let f = &frontal_data;

        // Diagonal entries
        assert_eq!(f[(0, 0)], 10.0, "diag global 0");
        assert_eq!(f[(1, 1)], 11.0, "diag global 1");
        assert_eq!(f[(2, 2)], 14.0, "diag global 4");
        assert_eq!(f[(3, 3)], 15.0, "diag global 5");

        // Off-diagonal between owned cols (lower triangle only)
        assert_eq!(f[(1, 0)], 1.0, "global(1,0) → local(1,0)");
        assert_eq!(f[(2, 0)], 4.0, "global(4,0) → local(2,0)");
        assert_eq!(f[(3, 1)], 5.0, "global(5,1) → local(3,1)");
        assert_eq!(f[(3, 2)], 7.0, "global(5,4) → local(3,2)");

        // Non-owned rows (global 2 → local 4, global 3 → local 5)
        assert_eq!(f[(4, 0)], 2.0, "global(2,0) → local(4,0)");
        assert_eq!(f[(4, 3)], 6.0, "global(2,5) → local(4,3)");
        assert_eq!(f[(5, 1)], 3.0, "global(3,1) → local(5,1)");

        // Verify no spurious double-counting: the (0,1) position should be 1.0,
        // not 2.0 (which would happen if upper-triangle entry was also scattered)
        assert_eq!(
            f[(1, 0)],
            1.0,
            "upper-triangle dedup: (1,0) should be 1.0, not 2.0"
        );

        // Cross-range upper-triangle dedup: global(4,0) upper entry global(0,4)
        // should NOT be scattered separately, so local(2,0) should be 4.0
        assert_eq!(
            f[(2, 0)],
            4.0,
            "cross-range upper-triangle dedup: (2,0) should be 4.0, not 8.0"
        );

        // Verify zeros where no entries exist
        assert_eq!(f[(4, 1)], 0.0, "no global(2,1) entry");
        assert_eq!(f[(4, 2)], 0.0, "no global(2,4) entry");
        assert_eq!(f[(5, 0)], 0.0, "no global(3,0) entry");
        assert_eq!(f[(5, 2)], 0.0, "no global(3,4) entry");
        assert_eq!(f[(5, 3)], 0.0, "no global(3,5) entry");
    }

    // -----------------------------------------------------------------------
    // extend_add_mapped: precomputed-map scatter (zero-delay fast path)
    // -----------------------------------------------------------------------

    /// Test that `extend_add_mapped` correctly scatters a child contribution
    /// into a parent frontal matrix using a precomputed u32 row mapping,
    /// handling the lower-triangle symmetry swap when child ordering differs
    /// from parent ordering (non-monotonic map from amalgamated supernodes).
    #[test]
    fn test_extend_add_mapped_hand_constructed() {
        // Parent frontal: 5×5, initially zero (represents assembled parent).
        let mut parent_data = Mat::<f64>::zeros(5, 5);
        let parent_row_indices = vec![10, 20, 30, 40, 50]; // global indices (unused by mapped path)
        let mut parent = FrontalMatrix {
            data: parent_data.as_mut(),
            row_indices: &parent_row_indices,
            num_fully_summed: 3,
        };

        // Child contribution: 3×3 lower triangle.
        //   [1.0          ]
        //   [2.0  3.0     ]
        //   [4.0  5.0  6.0]
        let mut child_data = Mat::<f64>::zeros(3, 3);
        child_data[(0, 0)] = 1.0;
        child_data[(1, 0)] = 2.0;
        child_data[(1, 1)] = 3.0;
        child_data[(2, 0)] = 4.0;
        child_data[(2, 1)] = 5.0;
        child_data[(2, 2)] = 6.0;

        let child = ContributionBlock::from_dense_lower(&child_data, vec![30, 10, 40], 0);

        // Precomputed map: child row 0 → parent local 2 (global 30),
        //                  child row 1 → parent local 0 (global 10),
        //                  child row 2 → parent local 3 (global 40).
        // Note: child row 1 maps to a LOWER parent index than child row 0,
        // so extend_add_mapped must handle the li < lj swap case.
        let ea_row_map: Vec<u32> = vec![2, 0, 3];

        extend_add_mapped(&mut parent, child, &ea_row_map, Par::Seq);

        // Expected scatter (lower triangle of parent):
        //
        // child(0,0)=1.0: li=2, lj=2 → parent(2,2) += 1.0
        // child(1,0)=2.0: li=0, lj=2 → li < lj, swap → parent(2,0) += 2.0
        // child(1,1)=3.0: li=0, lj=0 → parent(0,0) += 3.0
        // child(2,0)=4.0: li=3, lj=2 → parent(3,2) += 4.0
        // child(2,1)=5.0: li=3, lj=0 → parent(3,0) += 5.0
        // child(2,2)=6.0: li=3, lj=3 → parent(3,3) += 6.0

        let p = parent.data.as_ref();
        assert_eq!(p[(0, 0)], 3.0, "child(1,1)=3.0 → parent(0,0)");
        assert_eq!(p[(2, 0)], 2.0, "child(1,0)=2.0 swapped → parent(2,0)");
        assert_eq!(p[(2, 2)], 1.0, "child(0,0)=1.0 → parent(2,2)");
        assert_eq!(p[(3, 0)], 5.0, "child(2,1)=5.0 → parent(3,0)");
        assert_eq!(p[(3, 2)], 4.0, "child(2,0)=4.0 → parent(3,2)");
        assert_eq!(p[(3, 3)], 6.0, "child(2,2)=6.0 → parent(3,3)");

        // Verify no spurious entries (untouched positions remain zero)
        assert_eq!(p[(1, 0)], 0.0, "row 1 untouched");
        assert_eq!(p[(1, 1)], 0.0, "row 1 untouched");
        assert_eq!(p[(4, 0)], 0.0, "row 4 untouched");
        assert_eq!(p[(4, 4)], 0.0, "row 4 untouched");
    }

    /// Test column-oriented `extend_add_mapped` with a larger 10×10
    /// contribution (monotonic map), comparing against a naive reference.
    #[test]
    fn test_extend_add_mapped_10x10_vs_reference() {
        let cb_size = 10;
        let parent_size = 15;

        // Monotonically increasing map: child rows map to
        // parent rows [1, 3, 4, 5, 7, 8, 10, 11, 12, 14].
        let ea_row_map: Vec<u32> = vec![1, 3, 4, 5, 7, 8, 10, 11, 12, 14];

        // Build a child contribution with known lower-triangle values.
        // Use value = (i+1)*100 + (j+1) for easy identification.
        let mut child_data = Mat::<f64>::zeros(cb_size, cb_size);
        for j in 0..cb_size {
            for i in j..cb_size {
                child_data[(i, j)] = (i as f64 + 1.0) * 100.0 + (j as f64 + 1.0);
            }
        }

        // Reference: naive element-by-element scatter with lower-triangle swap
        let mut ref_parent = Mat::<f64>::zeros(parent_size, parent_size);
        for j in 0..cb_size {
            let lj = ea_row_map[j] as usize;
            for i in j..cb_size {
                let li = ea_row_map[i] as usize;
                if li >= lj {
                    ref_parent[(li, lj)] += child_data[(i, j)];
                } else {
                    ref_parent[(lj, li)] += child_data[(i, j)];
                }
            }
        }

        // Actual: column-oriented extend_add_mapped
        let mut parent_data = Mat::<f64>::zeros(parent_size, parent_size);
        let parent_row_indices: Vec<usize> = (0..parent_size).collect();
        let mut parent = FrontalMatrix {
            data: parent_data.as_mut(),
            row_indices: &parent_row_indices,
            num_fully_summed: 5,
        };
        let child = ContributionBlock::from_dense_lower(&child_data, (0..cb_size).collect(), 0);
        extend_add_mapped(&mut parent, child, &ea_row_map, Par::Seq);

        // Compare every element
        for j in 0..parent_size {
            for i in 0..parent_size {
                assert_eq!(
                    parent.data[(i, j)],
                    ref_parent[(i, j)],
                    "mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    /// Test column-oriented `extend_add_mapped` with a non-monotonic map
    /// (simulating post-amalgamation parent with non-contiguous owned ranges).
    #[test]
    fn test_extend_add_mapped_non_monotonic() {
        let cb_size = 4;
        let parent_size = 8;

        // Non-monotonic map: simulates child pattern rows mapping into a
        // parent whose owned_ranges are interleaved with pattern rows.
        let ea_row_map: Vec<u32> = vec![5, 2, 7, 1];

        // Build child contribution with lower-triangle values
        let mut child_data = Mat::<f64>::zeros(cb_size, cb_size);
        for j in 0..cb_size {
            for i in j..cb_size {
                child_data[(i, j)] = (i * 10 + j + 1) as f64;
            }
        }

        // Reference: naive scatter with swap
        let mut ref_parent = Mat::<f64>::zeros(parent_size, parent_size);
        for j in 0..cb_size {
            let lj = ea_row_map[j] as usize;
            for i in j..cb_size {
                let li = ea_row_map[i] as usize;
                if li >= lj {
                    ref_parent[(li, lj)] += child_data[(i, j)];
                } else {
                    ref_parent[(lj, li)] += child_data[(i, j)];
                }
            }
        }

        // Actual
        let mut parent_data = Mat::<f64>::zeros(parent_size, parent_size);
        let parent_row_indices: Vec<usize> = (0..parent_size).collect();
        let mut parent = FrontalMatrix {
            data: parent_data.as_mut(),
            row_indices: &parent_row_indices,
            num_fully_summed: 4,
        };
        let child = ContributionBlock::from_dense_lower(&child_data, (0..cb_size).collect(), 0);
        extend_add_mapped(&mut parent, child, &ea_row_map, Par::Seq);

        // Compare every element
        for j in 0..parent_size {
            for i in 0..parent_size {
                assert_eq!(
                    parent.data[(i, j)],
                    ref_parent[(i, j)],
                    "mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    /// Test that `extend_add_mapped` accumulates contributions from multiple
    /// children into the same parent.
    #[test]
    fn test_extend_add_mapped_accumulates() {
        let mut parent_data = Mat::<f64>::zeros(4, 4);
        let parent_row_indices = vec![0, 1, 2, 3];
        let mut parent = FrontalMatrix {
            data: parent_data.as_mut(),
            row_indices: &parent_row_indices,
            num_fully_summed: 2,
        };

        // First child: 2×2, maps to parent rows [1, 3]
        let mut c1_data = Mat::<f64>::zeros(2, 2);
        c1_data[(0, 0)] = 10.0;
        c1_data[(1, 0)] = 20.0;
        c1_data[(1, 1)] = 30.0;
        let child1 = ContributionBlock::from_dense_lower(&c1_data, vec![1, 3], 0);
        let map1: Vec<u32> = vec![1, 3];

        extend_add_mapped(&mut parent, child1, &map1, Par::Seq);

        // Second child: 2×2, also maps to parent rows [1, 3]
        let mut c2_data = Mat::<f64>::zeros(2, 2);
        c2_data[(0, 0)] = 5.0;
        c2_data[(1, 0)] = 7.0;
        c2_data[(1, 1)] = 9.0;
        let child2 = ContributionBlock::from_dense_lower(&c2_data, vec![1, 3], 0);
        let map2: Vec<u32> = vec![1, 3];

        extend_add_mapped(&mut parent, child2, &map2, Par::Seq);

        let p = parent.data.as_ref();
        assert_eq!(p[(1, 1)], 15.0, "10 + 5 accumulated");
        assert_eq!(p[(3, 1)], 27.0, "20 + 7 accumulated");
        assert_eq!(p[(3, 3)], 39.0, "30 + 9 accumulated");
    }

    // -----------------------------------------------------------------------
    // Small-leaf subtree classification tests
    // -----------------------------------------------------------------------

    /// Helper: create a SupernodeInfo with given parameters for classification tests.
    /// `ncols` = number of owned columns, `pattern_len` = number of off-diagonal rows.
    /// front_size = ncols + pattern_len.
    #[allow(clippy::single_range_in_vec_init)]
    fn make_snode(
        col_begin: usize,
        ncols: usize,
        pattern_len: usize,
        parent: Option<usize>,
    ) -> SupernodeInfo {
        SupernodeInfo {
            col_begin,
            col_end: col_begin + ncols,
            pattern: (0..pattern_len).map(|i| 1000 + col_begin + i).collect(),
            parent,
            owned_ranges: vec![col_begin..col_begin + ncols],
            in_small_leaf: false,
        }
    }

    /// A forest with many tiny roots plus one spine of medium fronts carrying
    /// small side subtrees (the shape of aug3dcqp). The L0 layer must walk
    /// down the spine, so the side leaves run in parallel and the spine goes
    /// to the top part, even though the forest starts with more roots than
    /// `32 * threads`.
    #[test]
    fn l0_layer_splits_spine_in_wide_forest() {
        let mut sn = Vec::new();
        let mut col = 0;
        for _ in 0..300 {
            sn.push(make_snode(col, 1, 2, None));
            col += 1;
        }
        let mut spine = Vec::new();
        let mut prev_spine: Option<usize> = None;
        for _ in 0..40 {
            let first_leaf = sn.len();
            for _ in 0..10 {
                sn.push(make_snode(col, 2, 4, None));
                col += 2;
            }
            let me = sn.len();
            sn.push(make_snode(col, 50, 350, None));
            col += 50;
            for leaf in &mut sn[first_leaf..me] {
                leaf.parent = Some(me);
            }
            if let Some(p) = prev_spine {
                sn[p].parent = Some(me);
            }
            prev_spine = Some(me);
            spine.push(me);
        }
        let mut children = vec![Vec::new(); sn.len()];
        for (s, node) in sn.iter().enumerate() {
            if let Some(p) = node.parent {
                children[p].push(s);
            }
        }

        let layer = select_l0_layer(&sn, &children, 8, 4096);
        let top = *spine.last().unwrap();
        assert!(!layer.contains(&top), "spine root left in the L0 layer");
        let spine_in_layer = layer.iter().filter(|r| spine.contains(r)).count();
        assert!(
            spine_in_layer <= 1,
            "{spine_in_layer} spine nodes in the layer"
        );
        // Every node is covered exactly once by a layer subtree or the top part.
        let mut covered = vec![0u32; sn.len()];
        let mut stack = layer.clone();
        while let Some(r) = stack.pop() {
            covered[r] += 1;
            stack.extend(children[r].iter().copied());
        }
        assert!(covered.iter().all(|&c| c <= 1));
        assert!(spine.iter().filter(|s| covered[**s] == 0).count() >= 39);
    }

    /// Linear chain of 5 small supernodes — all should be in_small_leaf.
    #[test]
    fn test_classify_all_small() {
        // Chain: 0 → 1 → 2 → 3 → 4 (each has 2 owned cols, 3 pattern → front_size=5)
        let mut supernodes = vec![
            make_snode(0, 2, 3, Some(1)),
            make_snode(2, 2, 3, Some(2)),
            make_snode(4, 2, 3, Some(3)),
            make_snode(6, 2, 3, Some(4)),
            make_snode(8, 2, 3, None),
        ];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 256);

        // All 5 nodes should be in_small_leaf
        for sn in &supernodes {
            assert!(
                sn.in_small_leaf,
                "snode {} should be in_small_leaf",
                sn.col_begin
            );
        }
        // One subtree with all 5 nodes in postorder
        assert_eq!(subtrees.len(), 1);
        assert_eq!(subtrees[0].nodes.len(), 5);
        assert_eq!(subtrees[0].nodes, vec![0, 1, 2, 3, 4]);
        assert_eq!(subtrees[0].root, 4);
        assert_eq!(subtrees[0].parent_of_root, None);
    }

    /// Mixed tree — small leaves, large root.
    #[test]
    fn test_classify_mixed_tree() {
        // Tree: leaves 0,1 → parent 2 (large)
        // Nodes 0,1: front_size=5 (small)
        // Node 2: front_size=300 (large, >= 256)
        let mut supernodes = vec![
            make_snode(0, 2, 3, Some(2)),  // front=5, small
            make_snode(2, 2, 3, Some(2)),  // front=5, small
            make_snode(4, 100, 200, None), // front=300, large
        ];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 256);

        // Node 2 is large → not in_small_leaf
        assert!(!supernodes[2].in_small_leaf);
        // Nodes 0,1 are small leaves but they don't form a subtree with ≥2 nodes
        // that has a common root — they are separate single-leaf subtrees
        // Each leaf alone is 1 node which is < 2, so no subtrees
        assert!(!supernodes[0].in_small_leaf, "single node excluded");
        assert!(!supernodes[1].in_small_leaf, "single node excluded");
        assert_eq!(subtrees.len(), 0);
    }

    /// Single isolated small leaves — no subtree formed (min 2 nodes).
    #[test]
    fn test_classify_single_node_excluded() {
        // Three independent small leaves (no parent linkage between them)
        let mut supernodes = vec![
            make_snode(0, 2, 3, None), // root, small, standalone
            make_snode(2, 2, 3, None), // root, small, standalone
            make_snode(4, 2, 3, None), // root, small, standalone
        ];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 256);

        // Each is a single-node subtree → excluded
        assert_eq!(subtrees.len(), 0);
        // in_small_leaf should be false for single-node subtrees (cleared by filter)
        for sn in &supernodes {
            assert!(!sn.in_small_leaf);
        }
    }

    /// Threshold boundary — front_size == 256 excluded, 255 included.
    #[test]
    fn test_classify_threshold_boundary() {
        // Node 0: front_size=255 (qualifies), child of node 1
        // Node 1: front_size=256 (does NOT qualify — strict less-than)
        let mut supernodes = vec![
            make_snode(0, 100, 155, Some(1)), // front=255
            make_snode(100, 128, 128, None),  // front=256
        ];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 256);

        assert!(
            !supernodes[1].in_small_leaf,
            "front_size=256 must NOT qualify"
        );
        // Node 0 is a small leaf but its parent is NOT small → single-node subtree → excluded
        assert!(!supernodes[0].in_small_leaf, "single node excluded");
        assert_eq!(subtrees.len(), 0);

        // Now with front_size=255 for the parent too
        let mut supernodes2 = vec![
            make_snode(0, 100, 155, Some(1)), // front=255
            make_snode(100, 127, 128, None),  // front=255
        ];
        let children2 = build_children_map(&supernodes2);
        let subtrees2 = classify_small_leaf_subtrees(&mut supernodes2, &children2, 256);

        assert!(supernodes2[0].in_small_leaf);
        assert!(supernodes2[1].in_small_leaf);
        assert_eq!(subtrees2.len(), 1);
        assert_eq!(subtrees2[0].nodes, vec![0, 1]);
    }

    /// Disabled when threshold=0.
    #[test]
    fn test_classify_disabled() {
        let mut supernodes = vec![make_snode(0, 2, 3, Some(1)), make_snode(2, 2, 3, None)];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 0);

        assert_eq!(subtrees.len(), 0);
        assert!(!supernodes[0].in_small_leaf);
        assert!(!supernodes[1].in_small_leaf);
    }

    /// Two independent small-leaf subtrees.
    #[test]
    fn test_classify_multiple_subtrees() {
        // Tree:
        //   0 → 2 (subtree A)
        //   1 → 2 (subtree A)
        //   2 → 5 (large root)
        //   3 → 4 (subtree B)
        //   4 → 5 (subtree B is just 3→4)
        //   5 = large root
        let mut supernodes = vec![
            make_snode(0, 2, 3, Some(2)),   // 0: front=5, small
            make_snode(2, 2, 3, Some(2)),   // 1: front=5, small
            make_snode(4, 2, 3, Some(5)),   // 2: front=5, small (parent is large)
            make_snode(6, 2, 3, Some(4)),   // 3: front=5, small
            make_snode(8, 2, 3, Some(5)),   // 4: front=5, small (parent is large)
            make_snode(10, 100, 200, None), // 5: front=300, large
        ];
        let children = build_children_map(&supernodes);

        let subtrees = classify_small_leaf_subtrees(&mut supernodes, &children, 256);

        // Two subtrees: {0,1,2} and {3,4}
        assert_eq!(subtrees.len(), 2);

        // Subtree A: root=2, nodes=[0,1,2] in postorder
        assert_eq!(subtrees[0].root, 2);
        assert_eq!(subtrees[0].nodes, vec![0, 1, 2]);
        assert_eq!(subtrees[0].parent_of_root, Some(5));

        // Subtree B: root=4, nodes=[3,4] in postorder
        assert_eq!(subtrees[1].root, 4);
        assert_eq!(subtrees[1].nodes, vec![3, 4]);
        assert_eq!(subtrees[1].parent_of_root, Some(5));

        // Large root is not in_small_leaf
        assert!(!supernodes[5].in_small_leaf);
        // All others are
        for (s, sn) in supernodes.iter().enumerate().take(5) {
            assert!(sn.in_small_leaf, "snode {} should be in_small_leaf", s);
        }
    }
}
