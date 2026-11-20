//! Per-supernode triangular solve for multifrontal LDL^T factorization.
//!
//! Provides the core solve function [`aptp_solve`] and per-supernode helper
//! functions that traverse the assembly tree performing gather/solve/scatter
//! operations using faer's dense triangular solve and matmul kernels.
//!
//! The solve operates in **permuted coordinates** — the caller (`SparseLDLT<T>`)
//! handles permutation (P) and optional scaling (T) around this core.
//!
//! # Algorithm
//!
//! Given P^T A P = L D L^T from multifrontal factorization:
//! 1. Forward solve (postorder): L y = b
//! 2. Diagonal solve: D z = y
//! 3. Backward solve (reverse postorder): L^T x = z
//!
//! # References
//!
//! - Duff, Hogg & Lopez (2020), Section 3: APTP solve algorithm
//! - Liu (1992), Sections 4-5: multifrontal solve traversal
//! - faer `SupernodalIntranodeLbltRef::solve_in_place_no_numeric_permute_with_conj`

use faer::dyn_stack::StackReq;
use faer::linalg::matmul::matmul_with_conj;
use faer::linalg::temp_mat_scratch;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Conj, Par};

use super::numeric::{AptpNumeric, FrontFactors, INTRA_NODE_THRESHOLD};
use super::symbolic::AptpSymbolic;
use crate::error::SparseError;
use crate::scalar::Scalar;

/// Wall-clock totals for the steps of the solve (feature `diagnostic`).
/// Read and reset with [`take_solve_profile`]. Slots with "big" are fronts
/// with `ne + r >= INTRA_NODE_THRESHOLD` (the ones that get `par`).
#[cfg(feature = "diagnostic")]
pub(crate) mod solve_profile {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub(crate) const NAMES: [&str; 17] = [
        "perm + scale (in)",
        "fwd small fronts (seq)",
        "fwd big gather/writeback",
        "fwd big L11 solve",
        "fwd big L21 GEMV",
        "fwd big scatter",
        "solve plan build",
        "bwd small fronts (seq)",
        "bwd big gather",
        "bwd big L21^T GEMV",
        "bwd big L11^T solve",
        "bwd big writeback",
        "unscale + unperm (out)",
        "aptp_solve total",
        "fwd subtree tasks (wall)",
        "fwd subtree reduce",
        "bwd subtree tasks (wall)",
    ];
    pub(crate) static NANOS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
    /// Factor bytes read by the slot (L11 lower triangle or L21).
    pub(crate) static BYTES: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
    pub(crate) static BIG_FRONTS: AtomicU64 = AtomicU64::new(0);
    /// Per-front slots (1..=5, 7..=11) only accumulate while this is set;
    /// it is cleared while subtree tasks run so those slots stay wall clock.
    pub(crate) static PER_NODE: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(true);

    #[inline]
    pub(crate) fn add(i: usize, t: std::time::Instant, bytes: usize) {
        NANOS[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        BYTES[i].fetch_add(bytes as u64, Relaxed);
    }
}

/// Take (and reset) the solve step timings: `(name, seconds, factor bytes)`,
/// plus the number of big-front visits (feature `diagnostic`).
#[cfg(feature = "diagnostic")]
pub fn take_solve_profile() -> (Vec<(&'static str, f64, f64)>, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let steps = solve_profile::NAMES
        .iter()
        .zip(solve_profile::NANOS.iter().zip(solve_profile::BYTES.iter()))
        .map(|(n, (a, b))| {
            (
                *n,
                a.swap(0, Relaxed) as f64 * 1e-9,
                b.swap(0, Relaxed) as f64,
            )
        })
        .collect();
    (steps, solve_profile::BIG_FRONTS.swap(0, Relaxed))
}

/// Start a timer only when profiling is compiled in.
macro_rules! tic {
    () => {{
        #[cfg(feature = "diagnostic")]
        let t = Some(std::time::Instant::now());
        #[cfg(not(feature = "diagnostic"))]
        let t: Option<()> = None;
        t
    }};
}

/// Start a per-front timer (skipped while subtree tasks run).
macro_rules! tic_node {
    () => {{
        #[cfg(feature = "diagnostic")]
        let t = solve_profile::PER_NODE
            .load(core::sync::atomic::Ordering::Relaxed)
            .then(std::time::Instant::now);
        #[cfg(not(feature = "diagnostic"))]
        let t: Option<()> = None;
        t
    }};
}

/// Record a timer started with `tic!` into slot `$i`.
macro_rules! toc {
    ($t:expr, $i:expr, $bytes:expr) => {{
        #[cfg(feature = "diagnostic")]
        if let Some(t) = $t {
            $crate::symmetric::solve::solve_profile::add($i, t, $bytes);
        }
        #[cfg(not(feature = "diagnostic"))]
        let _ = (&$t, $bytes);
    }};
}
pub(crate) use {tic, toc};

/// Per-supernode triangular solve through the multifrontal factor structure.
///
/// Solves L D L^T x = b in-place, where b is provided in `rhs` (permuted
/// coordinate system — the caller handles P and T). Traverses the assembly
/// tree performing per-supernode gather/solve/scatter operations.
///
/// # Arguments
///
/// - `symbolic`: Symbolic analysis (supernode structure, traversal order)
/// - `numeric`: Numeric factors (per-supernode L11, D11, L21)
/// - `rhs`: Right-hand side, modified in-place to contain solution
/// - `stack`: Pre-allocated workspace (from `aptp_solve_scratch()`)
///
/// # Errors
///
/// Returns `SparseError::DimensionMismatch` if `rhs.len() != symbolic.nrows()`.
///
/// # References
///
/// - Duff, Hogg & Lopez (2020), Section 3: APTP solve algorithm
/// - Liu (1992), Sections 4-5: multifrontal solve traversal
pub fn aptp_solve<T: Scalar>(
    symbolic: &AptpSymbolic,
    numeric: &AptpNumeric<T>,
    rhs: &mut [T],
    _stack: &mut faer::dyn_stack::MemStack,
    par: Par,
) -> Result<(), SparseError> {
    let n = symbolic.nrows();
    if rhs.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: (n, 1),
            got: (rhs.len(), 1),
            context: "RHS length must match matrix dimension".to_string(),
        });
    }

    if n == 0 {
        return Ok(());
    }

    let factors = numeric.front_factors();
    let max_size = numeric.stats().max_front_size;
    let t_total = tic!();

    // With threads, split the assembly tree into independent subtrees that
    // run concurrently (sequential kernels inside) below a top part that
    // runs front by front with parallel kernels.
    let t_plan = tic!();
    let plan = if par.degree() > 1 {
        SolvePlan::build(numeric, par.degree())
    } else {
        None
    };
    toc!(t_plan, 6, 0);

    match plan {
        Some(plan) => plan.solve(factors, rhs, max_size, par),
        None => {
            let mut work = vec![T::zero(); max_size];
            let mut work2 = vec![T::zero(); max_size];
            // Forward solve (postorder = index order), then the diagonal and
            // backward solves fused per front in reverse postorder. A front's
            // own entries are final after its forward solve and untouched
            // until its backward solve, so fusing D gives identical results.
            for ff in factors {
                forward_solve_supernode(ff, rhs, &mut work, &mut work2, par);
            }
            for ff in factors.iter().rev() {
                diag_backward_solve_supernode(ff, rhs, &mut work, &mut work2, par);
            }
        }
    }
    toc!(t_total, 13, 0);

    Ok(())
}

/// Access to the right-hand side for the per-front solve steps.
///
/// `[T]` is the plain sequential case. [`TaskRhs`] is the view a subtree
/// task gets: it writes its own entries in place and buffers updates to
/// entries owned by the top part of the tree.
pub(crate) trait SolveRhs<T> {
    /// Entry `i`.
    fn get(&self, i: usize) -> T;
    /// Overwrite entry `i` (only ever an entry the current front owns).
    fn set(&mut self, i: usize, v: T);
    /// Entry `i` -= `v` (forward scatter into an ancestor's entry).
    fn sub(&mut self, i: usize, v: T);
}

impl<T: Scalar> SolveRhs<T> for [T] {
    #[inline(always)]
    fn get(&self, i: usize) -> T {
        self[i]
    }
    #[inline(always)]
    fn set(&mut self, i: usize, v: T) {
        self[i] = v;
    }
    #[inline(always)]
    fn sub(&mut self, i: usize, v: T) {
        self[i] -= v;
    }
}

impl<T: Scalar> SolveRhs<T> for Vec<T> {
    #[inline(always)]
    fn get(&self, i: usize) -> T {
        self[i]
    }
    #[inline(always)]
    fn set(&mut self, i: usize, v: T) {
        self[i] = v;
    }
    #[inline(always)]
    fn sub(&mut self, i: usize, v: T) {
        self[i] -= v;
    }
}

/// Raw pointer to the right-hand side shared by concurrent subtree tasks.
#[derive(Clone, Copy)]
struct SharedRhs<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: tasks only access entries as described on `SolvePlan::solve`:
// disjoint writes, and reads of entries no task writes in that phase.
unsafe impl<T: Send> Send for SharedRhs<T> {}
unsafe impl<T: Send> Sync for SharedRhs<T> {}

/// A subtree task's view of the right-hand side (see [`SolvePlan`]).
struct TaskRhs<'a, T> {
    rhs: SharedRhs<T>,
    /// Pending forward updates to top-part entries, indexed by `top_idx`.
    buf: &'a mut [T],
    top_idx: &'a [u32],
}

impl<T: Scalar> SolveRhs<T> for TaskRhs<'_, T> {
    #[inline(always)]
    fn get(&self, i: usize) -> T {
        assert!(i < self.rhs.len);
        // SAFETY: in bounds; no other task writes entry `i` in this phase
        // (it is owned by this subtree or by the top part, see `SolvePlan`).
        unsafe { self.rhs.ptr.add(i).read() }
    }
    #[inline(always)]
    fn set(&mut self, i: usize, v: T) {
        assert!(i < self.rhs.len);
        debug_assert_eq!(self.top_idx[i], NOT_TOP);
        // SAFETY: in bounds; entry `i` belongs to a front in this subtree,
        // so no other task reads or writes it.
        unsafe { self.rhs.ptr.add(i).write(v) }
    }
    #[inline(always)]
    fn sub(&mut self, i: usize, v: T) {
        match self.top_idx[i] {
            NOT_TOP => {
                let x = self.get(i);
                self.set(i, x - v);
            }
            k => self.buf[k as usize] += v,
        }
    }
}

/// `top_idx` value for entries owned by a front inside some subtree task.
const NOT_TOP: u32 = u32::MAX;

/// Subtree tasks per thread: smaller tasks balance better, but move more
/// of the tree into the top part.
const SOLVE_TASKS_PER_THREAD: usize = 4;

/// Partition of the assembly tree for the threaded solve.
///
/// The tree is split, from the roots down, until every remaining subtree
/// reads at most `1 / (SOLVE_TASKS_PER_THREAD * threads)` of the factor.
/// Those subtrees are the tasks; the nodes above them are the top part.
///
/// # References
///
/// - Geist & Ng (1989), "Task scheduling for parallel sparse Cholesky
///   factorization": subtree-to-thread mapping by subtree cost
/// - Amestoy, Duff, L'Excellent & Rouet (2010) and the MUMPS solve-phase
///   documentation: L0 subtrees solved independently below a top part
struct SolvePlan {
    /// Nodes of each independent subtree in postorder, largest task first.
    tasks: Vec<Vec<usize>>,
    /// Nodes above the subtrees, in postorder.
    top: Vec<usize>,
    /// For each entry: its slot in a task's pending-update buffer, or
    /// `NOT_TOP` when a task node eliminates it. Entries no task node
    /// eliminates (top-part columns, and any never eliminated) get a slot.
    top_idx: Vec<u32>,
    n_top: usize,
}

impl SolvePlan {
    fn build<T: Scalar>(numeric: &AptpNumeric<T>, threads: usize) -> Option<Self> {
        let factors = numeric.front_factors();
        let parent = numeric.supernode_parents();
        let ns = factors.len();
        if threads < 2 || ns < 2 || parent.len() != ns {
            return None;
        }

        // Subtree cost: factor entries read, plus one per front for overhead.
        let mut cost = vec![0.0f64; ns];
        let mut children = vec![Vec::new(); ns];
        let mut roots = Vec::new();
        for s in 0..ns {
            let ne = factors[s].num_eliminated() as f64;
            let r = factors[s].row_indices().len() as f64;
            cost[s] += ne * (ne + 1.0) / 2.0 + ne * r + 1.0;
            match parent[s] {
                Some(p) => {
                    debug_assert!(p > s, "supernodes must be in postorder");
                    cost[p] += cost[s];
                    children[p].push(s);
                }
                None => roots.push(s),
            }
        }
        let total: f64 = roots.iter().map(|&r| cost[r]).sum();
        let target = total / (SOLVE_TASKS_PER_THREAD * threads) as f64;

        let mut layer = Vec::new();
        let mut top = Vec::new();
        let mut stack = roots;
        while let Some(r) = stack.pop() {
            if cost[r] > target && !children[r].is_empty() {
                top.push(r);
                stack.extend_from_slice(&children[r]);
            } else {
                layer.push(r);
            }
        }
        if layer.len() < 2 {
            return None;
        }
        top.sort_unstable();
        layer.sort_by(|a, b| cost[*b].total_cmp(&cost[*a]));

        let n = numeric.n();
        let mut top_idx = vec![0u32; n];
        let tasks: Vec<Vec<usize>> = layer
            .iter()
            .map(|&r| {
                let mut nodes = vec![r];
                let mut i = 0;
                while i < nodes.len() {
                    let s = nodes[i];
                    nodes.extend_from_slice(&children[s]);
                    i += 1;
                }
                nodes.sort_unstable();
                for &s in &nodes {
                    for &c in factors[s].col_indices() {
                        top_idx[c] = NOT_TOP;
                    }
                }
                nodes
            })
            .collect();
        let mut n_top = 0usize;
        for t in top_idx.iter_mut().filter(|t| **t != NOT_TOP) {
            *t = u32::try_from(n_top).ok().filter(|&k| k != NOT_TOP)?;
            n_top += 1;
        }
        Some(Self {
            tasks,
            top,
            top_idx,
            n_top,
        })
    }

    /// Forward, diagonal and backward solve using the partition.
    ///
    /// 1. Subtree tasks run the forward solve concurrently. A task writes
    ///    rhs entries its own fronts eliminate (no other task touches those)
    ///    and buffers its updates to every other entry. Those entries are
    ///    rows of the task root's L21 (each front's L21 rows are eliminated
    ///    by its ancestors or not at all, and anything not eliminated inside
    ///    the subtree reaches the root's front), so the buffer is drained
    ///    from the root's rows.
    /// 2. The buffered updates are subtracted, task by task.
    /// 3. The top part runs forward, then diagonal + backward, front by
    ///    front with parallel kernels.
    /// 4. Subtree tasks run diagonal + backward concurrently. They read top
    ///    entries (final, and nobody writes them now) and their own entries,
    ///    and write only their own.
    fn solve<T: Scalar>(
        &self,
        factors: &[FrontFactors<T>],
        rhs: &mut [T],
        max_size: usize,
        par: Par,
    ) {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let shared = SharedRhs {
            ptr: rhs.as_mut_ptr(),
            len: rhs.len(),
        };
        let top_idx = self.top_idx.as_slice();
        let workspace = || {
            (
                vec![T::zero(); self.n_top],
                vec![T::zero(); max_size],
                vec![T::zero(); max_size],
            )
        };
        #[cfg(feature = "diagnostic")]
        solve_profile::PER_NODE.store(false, core::sync::atomic::Ordering::Relaxed);

        // 1. Forward solve in the subtrees.
        let t = tic!();
        let pending: Vec<Vec<T>> = self
            .tasks
            .par_iter()
            .map_init(workspace, |(buf, work, work2), nodes| {
                let mut view = TaskRhs {
                    rhs: shared,
                    buf,
                    top_idx,
                };
                for &s in nodes {
                    forward_solve_supernode(&factors[s], &mut view, work, work2, Par::Seq);
                }
                let root = nodes[nodes.len() - 1];
                let out = factors[root]
                    .row_indices()
                    .iter()
                    .map(|&i| {
                        let k = top_idx[i];
                        debug_assert_ne!(k, NOT_TOP, "task root row inside a subtree");
                        core::mem::replace(&mut buf[k as usize], T::zero())
                    })
                    .collect();
                debug_assert!(
                    buf.iter().all(|&x| x == T::zero()),
                    "forward update outside the task root's rows"
                );
                out
            })
            .collect();
        toc!(t, 14, 0);

        // 2. Apply the buffered updates.
        let t = tic!();
        for (nodes, vals) in self.tasks.iter().zip(&pending) {
            let root = nodes[nodes.len() - 1];
            for (&i, &v) in factors[root].row_indices().iter().zip(vals) {
                rhs[i] -= v;
            }
        }
        drop(pending);
        toc!(t, 15, 0);
        #[cfg(feature = "diagnostic")]
        solve_profile::PER_NODE.store(true, core::sync::atomic::Ordering::Relaxed);

        // 3. Top part, front by front.
        let mut work = vec![T::zero(); max_size];
        let mut work2 = vec![T::zero(); max_size];
        for &s in &self.top {
            forward_solve_supernode(&factors[s], rhs, &mut work, &mut work2, par);
        }
        for &s in self.top.iter().rev() {
            diag_backward_solve_supernode(&factors[s], rhs, &mut work, &mut work2, par);
        }

        // 4. Diagonal + backward solve in the subtrees.
        #[cfg(feature = "diagnostic")]
        solve_profile::PER_NODE.store(false, core::sync::atomic::Ordering::Relaxed);
        let shared = SharedRhs {
            ptr: rhs.as_mut_ptr(),
            len: rhs.len(),
        };
        let t = tic!();
        self.tasks.par_iter().for_each_init(
            || (vec![T::zero(); max_size], vec![T::zero(); max_size]),
            |(work, work2), nodes| {
                let mut view = TaskRhs {
                    rhs: shared,
                    buf: &mut [],
                    top_idx,
                };
                for &s in nodes.iter().rev() {
                    diag_backward_solve_supernode(&factors[s], &mut view, work, work2, Par::Seq);
                }
            },
        );
        toc!(t, 16, 0);
        #[cfg(feature = "diagnostic")]
        solve_profile::PER_NODE.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Workspace requirement for `aptp_solve`.
///
/// Returns `StackReq` for the temporary per-supernode buffer.
/// Size is determined by `max_front_size` from factorization stats.
pub fn aptp_solve_scratch<T: Scalar>(numeric: &AptpNumeric<T>, rhs_ncols: usize) -> StackReq {
    let max_size = numeric.stats().max_front_size;
    temp_mat_scratch::<T>(max_size, rhs_ncols)
}

/// Forward solve for a single supernode.
///
/// 1. Gather rhs[col_indices[i]] → work buffer
/// 2. Solve L11 * y = work (unit lower triangular, in-place)
/// 3. Write work → rhs[col_indices[i]]
/// 4. Scatter: rhs[row_indices[j]] -= (L21 * work)[j]
pub(crate) fn forward_solve_supernode<T: Scalar, R: SolveRhs<T> + ?Sized>(
    ff: &FrontFactors<T>,
    rhs: &mut R,
    work: &mut [T],
    work2: &mut [T],
    par: Par,
) {
    let ne = ff.num_eliminated();
    if ne == 0 {
        return;
    }

    let col_indices = ff.col_indices();
    let row_indices = ff.row_indices();
    let r = row_indices.len();

    // Apply intra-node threshold: small fronts use Par::Seq
    let big = ne + r >= INTRA_NODE_THRESHOLD;
    let effective_par = if big { par } else { Par::Seq };
    let sz = core::mem::size_of::<T>();
    let (l11_bytes, l21_bytes) = (ne * (ne + 1) / 2 * sz, ne * r * sz);
    // Small fronts are timed as a whole; big ones step by step.
    let t_small = if big { None } else { tic_node!() };
    #[cfg(feature = "diagnostic")]
    if big && solve_profile::PER_NODE.load(core::sync::atomic::Ordering::Relaxed) {
        solve_profile::BIG_FRONTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    // Gather: work[i] = rhs[col_indices[i]]
    let t = if big { tic_node!() } else { None };
    for i in 0..ne {
        work[i] = rhs.get(col_indices[i]);
    }
    toc!(t, 2, 0);

    // Solve L11 * y = work (unit lower triangular, in-place)
    let t = if big { tic_node!() } else { None };
    if ne > 1 {
        let local = MatMut::from_column_major_slice_mut(&mut work[..ne], ne, 1);
        ff.l11().solve_in_place(local, effective_par);
    }
    toc!(t, 3, l11_bytes);

    // Write back: rhs[col_indices[i]] = work[i]
    let t = if big { tic_node!() } else { None };
    for i in 0..ne {
        rhs.set(col_indices[i], work[i]);
    }
    toc!(t, 2, 0);

    // Scatter: rhs[row_indices[j]] -= (L21 * work[..ne])[j]
    if r > 0 {
        let t = if big { tic_node!() } else { None };
        let local = MatRef::from_column_major_slice(&work[..ne], ne, 1);
        let tmp = &mut work2[..r];
        tmp.fill(T::zero());
        let mut tmp_mat = MatMut::from_column_major_slice_mut(tmp, r, 1);
        matmul_with_conj(
            tmp_mat.as_mut(),
            Accum::Replace,
            ff.l21().as_ref(),
            Conj::No,
            local,
            Conj::No,
            T::from_f64(1.0),
            effective_par,
        );
        toc!(t, 4, l21_bytes);
        let t = if big { tic_node!() } else { None };
        for j in 0..r {
            rhs.sub(row_indices[j], tmp[j]);
        }
        toc!(t, 5, 0);
    }
    toc!(t_small, 1, l11_bytes + l21_bytes);
}

/// Diagonal solve for a single supernode.
///
/// 1. Gather rhs[col_indices[i]] → work buffer
/// 2. D11.solve_in_place(work) (handles 1x1, 2x2, and zero pivots)
/// 3. Write work → rhs[col_indices[i]]
#[cfg(test)]
pub(crate) fn diagonal_solve_supernode<T: Scalar>(
    ff: &FrontFactors<T>,
    rhs: &mut [T],
    work: &mut [T],
) {
    let ne = ff.num_eliminated();
    if ne == 0 {
        return;
    }

    let col_indices = ff.col_indices();

    // Gather
    for i in 0..ne {
        work[i] = rhs[col_indices[i]];
    }

    // D-solve
    ff.d11().solve_in_place(&mut work[..ne]);

    // Write back
    for i in 0..ne {
        rhs[col_indices[i]] = work[i];
    }
}

/// Backward solve for a single supernode.
///
/// 1. Gather rhs[col_indices[i]] → work buffer
/// 2. Gather rhs[row_indices[j]] → work2 buffer (if r > 0)
/// 3. work -= L21^T * work2 (if r > 0)
/// 4. Solve L11^T * z = work (unit upper triangular via transpose)
/// 5. Write work → rhs[col_indices[i]]
#[cfg(test)]
pub(crate) fn backward_solve_supernode<T: Scalar>(
    ff: &FrontFactors<T>,
    rhs: &mut [T],
    work: &mut [T],
    work2: &mut [T],
    par: Par,
) {
    backward_impl(ff, rhs, work, work2, par, false);
}

/// Diagonal then backward solve for a single supernode: as
/// [`backward_solve_supernode`], with `D11⁻¹` applied to the gathered
/// entries first (the same as [`diagonal_solve_supernode`] beforehand).
pub(crate) fn diag_backward_solve_supernode<T: Scalar, R: SolveRhs<T> + ?Sized>(
    ff: &FrontFactors<T>,
    rhs: &mut R,
    work: &mut [T],
    work2: &mut [T],
    par: Par,
) {
    backward_impl(ff, rhs, work, work2, par, true);
}

fn backward_impl<T: Scalar, R: SolveRhs<T> + ?Sized>(
    ff: &FrontFactors<T>,
    rhs: &mut R,
    work: &mut [T],
    work2: &mut [T],
    par: Par,
    with_diag: bool,
) {
    let ne = ff.num_eliminated();
    if ne == 0 {
        return;
    }

    let col_indices = ff.col_indices();
    let row_indices = ff.row_indices();
    let r = row_indices.len();

    // Apply intra-node threshold: small fronts use Par::Seq
    let big = ne + r >= INTRA_NODE_THRESHOLD;
    let effective_par = if big { par } else { Par::Seq };
    let sz = core::mem::size_of::<T>();
    let (l11_bytes, l21_bytes) = (ne * (ne + 1) / 2 * sz, ne * r * sz);
    let t_small = if big { None } else { tic_node!() };

    // Gather local: work[i] = rhs[col_indices[i]]
    let t = if big { tic_node!() } else { None };
    for i in 0..ne {
        work[i] = rhs.get(col_indices[i]);
    }
    if with_diag {
        ff.d11().solve_in_place(&mut work[..ne]);
    }

    // Gather-update from L21^T
    if r > 0 {
        // Gather row values into work2
        for j in 0..r {
            work2[j] = rhs.get(row_indices[j]);
        }
        toc!(t, 8, 0);
        let t = if big { tic_node!() } else { None };
        let tmp = MatRef::from_column_major_slice(&work2[..r], r, 1);
        let mut local = MatMut::from_column_major_slice_mut(&mut work[..ne], ne, 1);

        // work -= L21^T * work2
        matmul_with_conj(
            local.as_mut(),
            Accum::Add,
            ff.l21().as_ref().transpose(),
            Conj::No,
            tmp,
            Conj::No,
            T::from_f64(-1.0),
            effective_par,
        );
        toc!(t, 9, l21_bytes);
    } else {
        toc!(t, 8, 0);
    }

    // Solve L11^T * z = work (unit upper triangular via transpose)
    let t = if big { tic_node!() } else { None };
    if ne > 1 {
        let local = MatMut::from_column_major_slice_mut(&mut work[..ne], ne, 1);
        ff.l11().solve_transpose_in_place(local, effective_par);
    }
    toc!(t, 10, l11_bytes);

    // Write back
    let t = if big { tic_node!() } else { None };
    for i in 0..ne {
        rhs.set(col_indices[i], work[i]);
    }
    toc!(t, 11, 0);
    toc!(t_small, 7, l11_bytes + l21_bytes);
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::linalg::cholesky::SymmetricOrdering;
    use faer::sparse::{SparseColMat, Triplet};

    use crate::symmetric::factor::AptpOptions;
    use crate::symmetric::numeric::AptpNumeric;
    use crate::symmetric::symbolic::AptpSymbolic;

    /// Build a full symmetric CSC matrix from lower-triangular entries.
    fn sparse_from_lower(n: usize, lower: &[(usize, usize, f64)]) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for &(i, j, v) in lower {
            triplets.push(Triplet::new(i, j, v));
            if i != j {
                triplets.push(Triplet::new(j, i, v));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).expect("triplets")
    }

    /// Factor a matrix with identity ordering, returning (symbolic, numeric).
    fn factor_identity(matrix: &SparseColMat<usize, f64>) -> (AptpSymbolic, AptpNumeric) {
        let symbolic =
            AptpSymbolic::analyze(matrix.symbolic(), SymmetricOrdering::Identity).expect("analyze");
        let numeric =
            AptpNumeric::factor(&symbolic, matrix, &AptpOptions::default(), None).expect("factor");
        (symbolic, numeric)
    }

    /// Sparse matvec: y = A * x (full symmetric CSC).
    fn matvec(a: &SparseColMat<usize, f64>, x: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let col_ptr = a.symbolic().col_ptr();
        let row_idx = a.symbolic().row_idx();
        let values = a.val();
        let mut y = vec![0.0; n];
        for j in 0..n {
            for idx in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[idx];
                y[i] += values[idx] * x[j];
            }
        }
        y
    }

    #[test]
    fn forward_solve_gather_scatter() {
        // 3x3 PD: [[4,1,0.5],[1,3,0],[0.5,0,2]]
        let matrix = sparse_from_lower(
            3,
            &[
                (0, 0, 4.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (2, 0, 0.5),
                (2, 2, 2.0),
            ],
        );
        let (symbolic, numeric) = factor_identity(&matrix);
        let factors = numeric.front_factors();

        // Known RHS
        let x_exact = [1.0, 2.0, 3.0];
        let b = matvec(&matrix, &x_exact);

        let (perm_fwd, _) = symbolic.perm_vecs();
        let n = 3;
        let mut rhs_perm: Vec<f64> = (0..n).map(|i| b[perm_fwd[i]]).collect();

        // Forward solve only
        let max_size = numeric.stats().max_front_size;
        let mut work = vec![0.0f64; max_size];
        let mut work2 = vec![0.0f64; max_size];
        for ff in factors {
            forward_solve_supernode(ff, &mut rhs_perm, &mut work, &mut work2, Par::Seq);
        }

        // After forward solve, rhs_perm should be L^{-1} b_perm.
        // We can't easily check exact values without reconstructing L,
        // but completing the full solve should recover x.
        for ff in factors {
            diagonal_solve_supernode(ff, &mut rhs_perm, &mut work);
        }
        let n_sn = factors.len();
        for s in (0..n_sn).rev() {
            backward_solve_supernode(&factors[s], &mut rhs_perm, &mut work, &mut work2, Par::Seq);
        }

        let mut x_computed = vec![0.0f64; n];
        for new in 0..n {
            x_computed[perm_fwd[new]] = rhs_perm[new];
        }

        let err: f64 = x_computed
            .iter()
            .zip(x_exact.iter())
            .map(|(c, e)| (c - e).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12, "full pipeline error: {:.2e}", err);
    }

    #[test]
    fn diagonal_solve_known_values() {
        // 4x4 diagonal matrix: diag = [2, 5, 3, 7]
        let matrix = sparse_from_lower(4, &[(0, 0, 2.0), (1, 1, 5.0), (2, 2, 3.0), (3, 3, 7.0)]);
        let (_, numeric) = factor_identity(&matrix);
        let factors = numeric.front_factors();

        // rhs = [4, 10, 9, 21] → x = [2, 2, 3, 3]
        let mut rhs = vec![4.0, 10.0, 9.0, 21.0];
        let max_size = numeric.stats().max_front_size;
        let mut work = vec![0.0f64; max_size];

        for ff in factors {
            diagonal_solve_supernode(ff, &mut rhs, &mut work);
        }

        let expected = [2.0, 2.0, 3.0, 3.0];
        for i in 0..4 {
            assert!(
                (rhs[i] - expected[i]).abs() < 1e-14,
                "rhs[{}] = {}, expected {}",
                i,
                rhs[i],
                expected[i]
            );
        }
    }

    #[test]
    fn backward_solve_reverses_forward() {
        // 4x4 tridiagonal: diag=[4,4,4,4], off-diag=1
        let matrix = sparse_from_lower(
            4,
            &[
                (0, 0, 4.0),
                (1, 0, 1.0),
                (1, 1, 4.0),
                (2, 1, 1.0),
                (2, 2, 4.0),
                (3, 2, 1.0),
                (3, 3, 4.0),
            ],
        );
        let (symbolic, numeric) = factor_identity(&matrix);
        let factors = numeric.front_factors();

        let x_exact = [1.0, -1.0, 2.0, -2.0];
        let b = matvec(&matrix, &x_exact);

        let (perm_fwd, _) = symbolic.perm_vecs();
        let n = 4;
        let mut rhs: Vec<f64> = (0..n).map(|i| b[perm_fwd[i]]).collect();

        let max_size = numeric.stats().max_front_size;
        let mut work = vec![0.0f64; max_size];
        let mut work2 = vec![0.0f64; max_size];

        // Full pipeline
        for ff in factors {
            forward_solve_supernode(ff, &mut rhs, &mut work, &mut work2, Par::Seq);
        }
        for ff in factors {
            diagonal_solve_supernode(ff, &mut rhs, &mut work);
        }
        let n_sn = factors.len();
        for s in (0..n_sn).rev() {
            backward_solve_supernode(&factors[s], &mut rhs, &mut work, &mut work2, Par::Seq);
        }

        let mut x_computed = vec![0.0f64; n];
        for new in 0..n {
            x_computed[perm_fwd[new]] = rhs[new];
        }

        let err: f64 = x_computed
            .iter()
            .zip(x_exact.iter())
            .map(|(c, e)| (c - e).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12, "tridiag pipeline error: {:.2e}", err);
    }

    #[test]
    fn per_supernode_solve_indefinite_2x2() {
        // 4x4 indefinite: requires 2x2 pivots
        // [[ 1, 2, 0, 0],
        //  [ 2,-3, 1, 0],
        //  [ 0, 1, 5, 1],
        //  [ 0, 0, 1,-2]]
        let matrix = sparse_from_lower(
            4,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (1, 1, -3.0),
                (2, 1, 1.0),
                (2, 2, 5.0),
                (3, 2, 1.0),
                (3, 3, -2.0),
            ],
        );
        let (symbolic, numeric) = factor_identity(&matrix);
        let factors = numeric.front_factors();

        let x_exact = [1.0, -1.0, 2.0, 0.5];
        let b = matvec(&matrix, &x_exact);

        let (perm_fwd, _) = symbolic.perm_vecs();
        let n = 4;
        let mut rhs: Vec<f64> = (0..n).map(|i| b[perm_fwd[i]]).collect();

        let max_size = numeric.stats().max_front_size;
        let mut work = vec![0.0f64; max_size];
        let mut work2 = vec![0.0f64; max_size];

        for ff in factors {
            forward_solve_supernode(ff, &mut rhs, &mut work, &mut work2, Par::Seq);
        }
        for ff in factors {
            diagonal_solve_supernode(ff, &mut rhs, &mut work);
        }
        let n_sn = factors.len();
        for s in (0..n_sn).rev() {
            backward_solve_supernode(&factors[s], &mut rhs, &mut work, &mut work2, Par::Seq);
        }

        let mut x_computed = vec![0.0f64; n];
        for new in 0..n {
            x_computed[perm_fwd[new]] = rhs[new];
        }

        let err: f64 = x_computed
            .iter()
            .zip(x_exact.iter())
            .map(|(c, e)| (c - e).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12, "indefinite 2x2 pipeline error: {:.2e}", err);
    }

    #[test]
    fn front_factors_index_consistency() {
        // 8x8 arrow matrix: tests multi-supernode factorization
        let mut entries = Vec::new();
        for i in 0..8 {
            entries.push((i, i, if i < 7 { 10.0 } else { 20.0 }));
            if i < 7 {
                entries.push((7, i, 1.0));
            }
        }
        let matrix = sparse_from_lower(8, &entries);
        let (symbolic, numeric) = factor_identity(&matrix);
        let factors = numeric.front_factors();

        // Verify structural properties
        let n = symbolic.nrows();
        let mut total_eliminated = 0;
        for ff in factors {
            let ne = ff.num_eliminated();
            total_eliminated += ne;

            // L11 dimensions must match num_eliminated
            assert_eq!(ff.l11().dim(), ne);
            assert_eq!(ff.l11().dim(), ne);

            // L21 rows = row_indices length
            assert_eq!(ff.l21().nrows(), ff.row_indices().len());
            assert_eq!(ff.l21().ncols(), ne);

            // All col_indices must be valid
            for &ci in ff.col_indices() {
                assert!(ci < n, "col_index {} >= n={}", ci, n);
            }
            for &ri in ff.row_indices() {
                assert!(ri < n, "row_index {} >= n={}", ri, n);
            }
        }
        assert_eq!(total_eliminated, n, "total eliminated must equal n");
    }

    #[test]
    fn aptp_solve_dimension_mismatch() {
        let matrix = sparse_from_lower(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let (symbolic, numeric) = factor_identity(&matrix);

        let mut rhs = vec![1.0, 2.0]; // wrong size
        let scratch = aptp_solve_scratch(&numeric, 1);
        let mut mem = MemBuffer::new(scratch);
        let stack = MemStack::new(&mut mem);

        let result = aptp_solve(&symbolic, &numeric, &mut rhs, stack, Par::Seq);
        assert!(
            matches!(result, Err(SparseError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn aptp_solve_empty() {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![];
        let matrix = SparseColMat::try_new_from_triplets(0, 0, &triplets).expect("empty matrix");
        let symbolic =
            AptpSymbolic::analyze(matrix.symbolic(), SymmetricOrdering::Identity).expect("analyze");
        let numeric =
            AptpNumeric::factor(&symbolic, &matrix, &AptpOptions::default(), None).expect("factor");

        let mut rhs: Vec<f64> = vec![];
        let scratch = aptp_solve_scratch(&numeric, 1);
        let mut mem = MemBuffer::new(scratch);
        let stack = MemStack::new(&mut mem);

        let result = aptp_solve(&symbolic, &numeric, &mut rhs, stack, Par::Seq);
        assert!(result.is_ok(), "0x0 solve should succeed");
    }

    /// Lower triangle of a shifted 7-point Laplacian on a `k³` grid, plus
    /// `m` constraint rows (zero diagonal) tying pairs of grid points, which
    /// gives 2×2 pivots and delays.
    fn grid_kkt_lower(k: usize, shift: f64, m: usize) -> (usize, Vec<(usize, usize, f64)>) {
        let id = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
        let ng = k * k * k;
        let mut lower = Vec::new();
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    let i = id(x, y, z);
                    lower.push((i, i, 6.0 - shift));
                    if x + 1 < k {
                        lower.push((id(x + 1, y, z), i, -1.0));
                    }
                    if y + 1 < k {
                        lower.push((id(x, y + 1, z), i, -1.0));
                    }
                    if z + 1 < k {
                        lower.push((id(x, y, z + 1), i, -1.0));
                    }
                }
            }
        }
        for c in 0..m {
            let a = (c * 7919) % ng;
            let b = (c * 104_729 + 13) % ng;
            lower.push((ng + c, a, 1.0));
            if b != a {
                lower.push((ng + c, b, -0.5));
            }
        }
        (ng + m, lower)
    }

    fn factor_amd(matrix: &SparseColMat<usize, f64>) -> (AptpSymbolic, AptpNumeric) {
        let symbolic =
            AptpSymbolic::analyze(matrix.symbolic(), SymmetricOrdering::Amd).expect("analyze");
        let numeric =
            AptpNumeric::factor(&symbolic, matrix, &AptpOptions::default(), None).expect("factor");
        (symbolic, numeric)
    }

    fn solve_with(symbolic: &AptpSymbolic, numeric: &AptpNumeric, b: &[f64], par: Par) -> Vec<f64> {
        let mut x = b.to_vec();
        let mut mem = MemBuffer::new(aptp_solve_scratch(numeric, 1));
        aptp_solve(symbolic, numeric, &mut x, MemStack::new(&mut mem), par).expect("solve");
        x
    }

    /// The threaded solve splits the tree into several subtrees under a top
    /// part, and agrees with the sequential solve to rounding.
    #[test]
    fn tree_parallel_solve_matches_sequential() {
        for (k, shift, m) in [(14, 0.0, 0), (12, 0.7, 0), (11, 0.3, 200)] {
            let (n, lower) = grid_kkt_lower(k, shift, m);
            let a = sparse_from_lower(n, &lower);
            let (symbolic, numeric) = factor_amd(&a);

            let plan = SolvePlan::build(&numeric, 4).expect("plan");
            assert!(plan.tasks.len() >= 4, "only {} tasks", plan.tasks.len());
            assert!(!plan.top.is_empty());
            let mut seen = vec![false; numeric.front_factors().len()];
            for &s in plan.tasks.iter().flatten().chain(&plan.top) {
                assert!(!seen[s], "supernode {s} in two places");
                seen[s] = true;
            }
            assert!(seen.iter().all(|&x| x), "supernode missing from plan");

            let b: Vec<f64> = (0..n)
                .map(|i| ((i * 37) % 101) as f64 / 50.0 - 1.0)
                .collect();
            let x_seq = solve_with(&symbolic, &numeric, &b, Par::Seq);
            let scale = x_seq.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for par in [Par::rayon(2), Par::rayon(3), Par::rayon(4), Par::rayon(8)] {
                let x = solve_with(&symbolic, &numeric, &b, par);
                let diff = x
                    .iter()
                    .zip(&x_seq)
                    .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
                assert!(
                    diff <= 1e-10 * scale,
                    "k={k} m={m} {par:?}: max diff {diff:e} (scale {scale:e})"
                );
            }
        }
    }

    /// The fused diagonal + backward step equals the diagonal pass followed
    /// by the backward pass, bit for bit.
    #[test]
    fn fused_diag_backward_matches_separate() {
        let (n, lower) = grid_kkt_lower(8, 0.5, 40);
        let a = sparse_from_lower(n, &lower);
        let (_, numeric) = factor_amd(&a);
        let factors = numeric.front_factors();
        let max = numeric.stats().max_front_size;
        let (mut w1, mut w2) = (vec![0.0; max], vec![0.0; max]);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();

        let mut x1 = b.clone();
        for ff in factors {
            forward_solve_supernode(ff, &mut x1, &mut w1, &mut w2, Par::Seq);
        }
        let mut x2 = x1.clone();
        for ff in factors {
            diagonal_solve_supernode(ff, &mut x1, &mut w1);
        }
        for ff in factors.iter().rev() {
            backward_solve_supernode(ff, &mut x1, &mut w1, &mut w2, Par::Seq);
            diag_backward_solve_supernode(ff, &mut x2, &mut w1, &mut w2, Par::Seq);
        }
        assert_eq!(x1, x2);
    }
}
