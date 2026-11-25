//! Solver tests for non-`f64` scalar types: complex symmetric (`c64`, `c32`)
//! and single precision (`f32`).
//!
//! Complex symmetric means A = Aᵀ with complex entries, which is **not**
//! Hermitian (A ≠ Aᴴ). Every test matrix has complex off-diagonal entries, so
//! a factorization that conjugated anywhere (computing L D Lᴴ) would give a
//! large residual under the plain, non-conjugating matvec used here.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::{SparseColMat, Triplet};
use faer::traits::num_traits::ToPrimitive;
use faer::{Col, Par, c32, c64};

use rivrs_sparse::scalar::Scalar;
use rivrs_sparse::symmetric::{AnalyzeOptions, FactorOptions, OrderingStrategy, SparseLDLT};

/// Deterministic pseudo-random numbers in [0, 1) (no `rand` dependency needed).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Build a full symmetric CSC matrix from lower-triangle entries (i >= j).
fn symmetric_from_lower<T: Scalar>(
    n: usize,
    lower: &[(usize, usize, T)],
) -> SparseColMat<usize, T> {
    let mut trip = Vec::with_capacity(2 * lower.len());
    for &(i, j, v) in lower {
        trip.push(Triplet::new(i, j, v));
        if i != j {
            trip.push(Triplet::new(j, i, v));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &trip).expect("valid triplets")
}

/// y = A x with no conjugation.
fn matvec<T: Scalar>(a: &SparseColMat<usize, T>, x: &Col<T>) -> Col<T> {
    let mut y = Col::<T>::from_fn(a.nrows(), |_| T::zero());
    let cp = a.symbolic().col_ptr();
    let ri = a.symbolic().row_idx();
    let v = a.val();
    for j in 0..a.ncols() {
        for k in cp[j]..cp[j + 1] {
            y[ri[k]] += v[k] * x[j];
        }
    }
    y
}

/// Normwise backward error ‖Ax − b‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞), computed in f64.
fn backward_error<T: Scalar>(a: &SparseColMat<usize, T>, x: &Col<T>, b: &Col<T>) -> f64 {
    let ax = matvec(a, x);
    let n = a.nrows();
    let mut row_sum = vec![0.0f64; n];
    let cp = a.symbolic().col_ptr();
    let ri = a.symbolic().row_idx();
    for j in 0..n {
        for k in cp[j]..cp[j + 1] {
            row_sum[ri[k]] += mag64(a.val()[k]);
        }
    }
    let a_norm = row_sum.iter().cloned().fold(0.0, nan_max);
    let x_norm = (0..n).map(|i| mag64(x[i])).fold(0.0, nan_max);
    let b_norm = (0..n).map(|i| mag64(b[i])).fold(0.0, nan_max);
    let r_norm = (0..n).map(|i| mag64(ax[i] - b[i])).fold(0.0, nan_max);
    r_norm / (a_norm * x_norm + b_norm)
}

/// `max` that keeps NaN, so a non-finite residual or solution fails the
/// tolerance checks instead of being dropped by `f64::max`.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// |x| widened to f64 for error measurements.
fn mag64<T: Scalar>(x: T) -> f64 {
    x.mag().to_f64().expect("f32/f64 widen to f64")
}

/// A scalar with real part `re` and, for complex types, imaginary part `im`.
trait FromParts: Scalar {
    fn from_parts(re: f64, im: f64) -> Self;
}
impl FromParts for f64 {
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
}
impl FromParts for f32 {
    fn from_parts(re: f64, _im: f64) -> Self {
        re as f32
    }
}
impl FromParts for c64 {
    fn from_parts(re: f64, im: f64) -> Self {
        c64::new(re, im)
    }
}
impl FromParts for c32 {
    fn from_parts(re: f64, im: f64) -> Self {
        c32::new(re as f32, im as f32)
    }
}

/// 3D 7-point Helmholtz-like operator −Δ − k²(1 + iη) on an m³ grid with
/// complex, spatially varying coefficients and complex off-diagonals. For
/// real `T` the imaginary parts are dropped (real symmetric indefinite).
fn helmholtz_3d<T: FromParts>(m: usize, seed: u64) -> SparseColMat<usize, T> {
    let mut rng = Lcg(seed);
    let idx = |i: usize, j: usize, k: usize| i + m * (j + m * k);
    let k2 = 2.5; // shifts the spectrum so the operator is indefinite
    let mut lower = Vec::new();
    for k in 0..m {
        for j in 0..m {
            for i in 0..m {
                let p = idx(i, j, k);
                let eta = 0.1 + 0.4 * rng.next();
                lower.push((p, p, T::from_parts(6.0 - k2, -k2 * eta)));
                let mut off = |q: usize, rng: &mut Lcg| {
                    let c = 1.0 + 0.5 * rng.next();
                    lower.push((p, q, T::from_parts(-c, 0.2 * (rng.next() - 0.5))));
                };
                if i > 0 {
                    off(idx(i - 1, j, k), &mut rng);
                }
                if j > 0 {
                    off(idx(i, j - 1, k), &mut rng);
                }
                if k > 0 {
                    off(idx(i, j, k - 1), &mut rng);
                }
            }
        }
    }
    symmetric_from_lower(m * m * m, &lower)
}

/// Saddle-point [[H, Bᵀ], [B, 0]] with complex H and B. The zero (2,2) block
/// forces 2×2 pivots and exercises delayed pivoting.
fn kkt<T: FromParts>(nh: usize, nc: usize, seed: u64) -> SparseColMat<usize, T> {
    let mut rng = Lcg(seed);
    let mut lower = Vec::new();
    for i in 0..nh {
        lower.push((i, i, T::from_parts(2.0 + rng.next(), 0.5 * rng.next())));
        if i > 0 {
            lower.push((i, i - 1, T::from_parts(-1.0, 0.3 * (rng.next() - 0.5))));
        }
    }
    for c in 0..nc {
        for _ in 0..4 {
            let j = (rng.next() * nh as f64) as usize;
            lower.push((nh + c, j, T::from_parts(rng.next() - 0.5, rng.next() - 0.5)));
        }
    }
    // duplicate (row, col) pairs are summed by try_new_from_triplets
    symmetric_from_lower(nh + nc, &lower)
}

/// Solve A x = A·1 and return (backward error, max |x − 1|).
fn solve_ones<T: Scalar>(
    a: &SparseColMat<usize, T>,
    ordering: OrderingStrategy,
    par: Par,
) -> (f64, f64, SparseLDLT<T>) {
    let n = a.nrows();
    let ones = Col::<T>::from_fn(n, |_| T::one());
    let b = matvec(a, &ones);
    let mut solver =
        SparseLDLT::analyze_with_matrix(a, &AnalyzeOptions { ordering }).expect("analyze");
    solver
        .factor(
            a,
            &FactorOptions {
                par,
                ..Default::default()
            },
        )
        .expect("factor");
    let mut mem = MemBuffer::new(solver.solve_scratch(1));
    let x = solver
        .solve(&b, MemStack::new(&mut mem), par)
        .expect("solve");
    let fe = (0..n).map(|i| mag64(x[i] - T::one())).fold(0.0, nan_max);
    (backward_error(a, &x, &b), fe, solver)
}

fn orderings() -> [OrderingStrategy; 3] {
    [
        OrderingStrategy::Amd,
        OrderingStrategy::Metis,
        OrderingStrategy::MatchOrderMetis,
    ]
}

#[test]
fn test_matrices_are_symmetric_not_hermitian() {
    let a = helmholtz_3d::<c64>(4, 1);
    let at = a.as_ref().transpose().to_col_major().expect("transpose");
    let mut max_sym_diff = 0.0f64;
    let mut has_complex_offdiag = false;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let (x, y) = (a.get(i, j).copied(), at.get(i, j).copied());
            if let (Some(x), Some(y)) = (x, y) {
                max_sym_diff = max_sym_diff.max(mag64(x - y));
                has_complex_offdiag |= i != j && x.im != 0.0;
            }
        }
    }
    assert_eq!(max_sym_diff, 0.0, "A must equal Aᵀ");
    assert!(has_complex_offdiag, "A must differ from Aᴴ");
}

#[test]
fn c64_helmholtz_all_orderings() {
    let a = helmholtz_3d::<c64>(10, 7);
    for ord in orderings() {
        let name = format!("{ord:?}");
        let (be, fe, _) = solve_ones(&a, ord, Par::Seq);
        assert!(be < 1e-13, "c64 helmholtz {name}: backward error {be:e}");
        assert!(fe < 1e-8, "c64 helmholtz {name}: forward error {fe:e}");
    }
}

#[test]
fn c64_kkt_needs_two_by_two_pivots() {
    let a = kkt::<c64>(600, 150, 3);
    for ord in orderings() {
        let name = format!("{ord:?}");
        let (be, _, solver) = solve_ones(&a, ord, Par::Seq);
        assert!(be < 1e-12, "c64 kkt {name}: backward error {be:e}");
        let stats = solver.stats().expect("stats");
        assert!(
            stats.total_2x2_pivots > 0,
            "c64 kkt {name}: expected 2x2 pivots"
        );
    }
}

#[test]
fn c64_parallel_matches_sequential() {
    // 22^3 with METIS gives fronts above the intra-node parallel threshold
    // (256), so this exercises the parallel TRSM/GEMM and tree-level paths.
    let a = helmholtz_3d::<c64>(22, 11);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let (be_par, _, solver) =
        pool.install(|| solve_ones(&a, OrderingStrategy::Metis, Par::rayon(4)));
    let (be_seq, _, _) = solve_ones(&a, OrderingStrategy::Metis, Par::Seq);
    assert!(solver.stats().expect("stats").max_front_size > 256);
    assert!(be_par < 1e-13, "parallel backward error {be_par:e}");
    assert!(be_seq < 1e-13, "sequential backward error {be_seq:e}");
}

#[test]
fn complex_inertia_is_none() {
    let a = helmholtz_3d::<c64>(5, 2);
    let (_, _, solver) = solve_ones(&a, OrderingStrategy::Metis, Par::Seq);
    assert!(solver.inertia().is_none());
    let a = helmholtz_3d::<f64>(5, 2);
    let (_, _, solver) = solve_ones(&a, OrderingStrategy::Metis, Par::Seq);
    let inertia = solver.inertia().expect("real inertia");
    assert_eq!(inertia.positive + inertia.negative + inertia.zero, 125);
    assert!(inertia.negative > 0, "operator should be indefinite");
}

#[test]
fn c32_helmholtz_and_kkt() {
    for ord in orderings() {
        let name = format!("{ord:?}");
        let (be, _, _) = solve_ones(&helmholtz_3d::<c32>(8, 5), ord.clone(), Par::Seq);
        assert!(be < 1e-5, "c32 helmholtz {name}: backward error {be:e}");
        let (be, _, _) = solve_ones(&kkt::<c32>(300, 80, 9), ord, Par::Seq);
        assert!(be < 1e-5, "c32 kkt {name}: backward error {be:e}");
    }
}

/// Badly scaled c32 KKT matrices without MC64 scaling. The 2×2 pivots used to
/// divide by an unscaled determinant (order `|a|²`), and complex division
/// squares it again, so every solution entry overflowed to NaN/Inf for a
/// matrix scale of `1e±10`.
#[test]
fn c32_kkt_badly_scaled() {
    for scale in [1e10f32, 1e-10, 1e15] {
        let mut a = kkt::<c32>(300, 80, 9);
        for v in a.val_mut() {
            *v *= scale;
        }
        for ord in [OrderingStrategy::Amd, OrderingStrategy::Metis] {
            let name = format!("{ord:?}");
            let (be, _, solver) = solve_ones(&a, ord, Par::Seq);
            let n2x2 = solver.stats().expect("stats").total_2x2_pivots;
            assert!(n2x2 > 0, "c32 kkt x{scale:e} {name}: expected 2x2 pivots");
            // Without MC64 scaling, single precision on this KKT matrix lands
            // around 1e-5, so the bound only needs to catch NaN/Inf blow-up.
            assert!(
                be < 1e-4,
                "c32 kkt x{scale:e} {name}: backward error {be:e}"
            );
        }
    }
}

#[test]
fn f32_helmholtz_and_kkt() {
    for ord in orderings() {
        let name = format!("{ord:?}");
        let (be, _, _) = solve_ones(&helmholtz_3d::<f32>(8, 5), ord.clone(), Par::Seq);
        // f32 eps = 1.2e-7; the real (undamped) Helmholtz operator is the harder
        // indefinite case, so allow ~800 eps.
        assert!(be < 1e-4, "f32 helmholtz {name}: backward error {be:e}");
        let (be, _, _) = solve_ones(&kkt::<f32>(300, 80, 9), ord, Par::Seq);
        assert!(be < 1e-4, "f32 kkt {name}: backward error {be:e}");
    }
}

#[test]
fn f64_and_c64_agree_on_real_matrix() {
    // The same real matrix factored as f64 and as c64 (zero imaginary parts)
    // must give the same solution to rounding.
    let ar = helmholtz_3d::<f64>(7, 4);
    let lower: Vec<(usize, usize, c64)> = {
        let cp = ar.symbolic().col_ptr();
        let ri = ar.symbolic().row_idx();
        (0..ar.ncols())
            .flat_map(|j| (cp[j]..cp[j + 1]).map(move |k| (ri[k], j, k)))
            .filter(|&(i, j, _)| i >= j)
            .map(|(i, j, k)| (i, j, c64::new(ar.val()[k], 0.0)))
            .collect()
    };
    let ac = symmetric_from_lower(ar.nrows(), &lower);
    let (be_r, fe_r, _) = solve_ones(&ar, OrderingStrategy::Metis, Par::Seq);
    let (be_c, fe_c, _) = solve_ones(&ac, OrderingStrategy::Metis, Par::Seq);
    assert!(
        be_r < 1e-13 && be_c < 1e-13,
        "backward errors {be_r:e} {be_c:e}"
    );
    assert!(
        (fe_r - fe_c).abs() < 1e-10,
        "forward errors differ: {fe_r:e} vs {fe_c:e}"
    );
}

#[test]
fn metis_ordering_is_deterministic_under_concurrency() {
    // METIS has global state; concurrent analyze calls must still give the
    // same ordering as a single-threaded call.
    use rivrs_sparse::ordering::metis_ordering;
    let a = kkt::<f64>(300, 80, 9);
    let reference = metis_ordering(a.symbolic())
        .expect("metis")
        .arrays()
        .0
        .to_vec();
    let mismatches = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| {
                for _ in 0..25 {
                    let p = metis_ordering(a.symbolic())
                        .expect("metis")
                        .arrays()
                        .0
                        .to_vec();
                    if p != reference {
                        mismatches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    assert_eq!(
        mismatches.into_inner(),
        0,
        "concurrent METIS orderings differ"
    );
}

/// Shifted 3D Laplacian, 6 + σ on the diagonal: symmetric positive definite
/// for real `T`; for complex `T` the shift gets an imaginary part (complex
/// symmetric with positive definite real part, as in damped wave problems).
fn shifted_laplacian_3d<T: FromParts>(m: usize, seed: u64) -> SparseColMat<usize, T> {
    let mut rng = Lcg(seed);
    let idx = |i: usize, j: usize, k: usize| i + m * (j + m * k);
    let mut lower = Vec::new();
    for k in 0..m {
        for j in 0..m {
            for i in 0..m {
                let p = idx(i, j, k);
                lower.push((
                    p,
                    p,
                    T::from_parts(6.0 + 0.5 * rng.next(), 0.3 + rng.next()),
                ));
                for q in [
                    (i > 0).then(|| idx(i - 1, j, k)),
                    (j > 0).then(|| idx(i, j - 1, k)),
                    (k > 0).then(|| idx(i, j, k - 1)),
                ]
                .into_iter()
                .flatten()
                {
                    lower.push((p, q, T::from_parts(-1.0, 0.0)));
                }
            }
        }
    }
    symmetric_from_lower(m * m * m, &lower)
}

/// Solve A x = A·1 with the given pivoting choice (METIS ordering).
fn solve_ones_pivoting<T: Scalar>(
    a: &SparseColMat<usize, T>,
    pivoting: bool,
    par: Par,
) -> Result<(f64, f64), rivrs_sparse::error::SparseError> {
    let n = a.nrows();
    let ones = Col::<T>::from_fn(n, |_| T::one());
    let b = matvec(a, &ones);
    let mut solver = SparseLDLT::analyze_with_matrix(
        a,
        &AnalyzeOptions {
            ordering: OrderingStrategy::Metis,
        },
    )?;
    solver.factor(
        a,
        &FactorOptions {
            pivoting,
            par,
            ..Default::default()
        },
    )?;
    let mut mem = MemBuffer::new(solver.solve_scratch(1));
    let x = solver.solve(&b, MemStack::new(&mut mem), par)?;
    let fe = (0..n).map(|i| mag64(x[i] - T::one())).fold(0.0, f64::max);
    Ok((backward_error(a, &x, &b), fe))
}

#[test]
fn unpivoted_matches_pivoted_on_definite_matrices() {
    fn check<T: FromParts>(tol: f64, name: &str) {
        // 14^3 with METIS has fronts above 128 fully-summed columns, so the
        // blocked kernel and its rows-below TRSM are both exercised.
        let a = shifted_laplacian_3d::<T>(14, 3);
        for par in [Par::Seq, Par::rayon(4)] {
            let (be_u, fe_u) = solve_ones_pivoting(&a, false, par).expect("unpivoted factor");
            let (be_p, _) = solve_ones_pivoting(&a, true, par).expect("pivoted factor");
            assert!(be_u < tol, "{name} unpivoted backward error {be_u:e}");
            assert!(be_p < tol, "{name} pivoted backward error {be_p:e}");
            assert!(fe_u < 1e3 * tol, "{name} unpivoted forward error {fe_u:e}");
        }
    }
    check::<f64>(1e-13, "f64");
    check::<c64>(1e-13, "c64");
    check::<f32>(1e-5, "f32");
    check::<c32>(1e-5, "c32");
}

#[test]
fn unpivoted_saddle_point_is_refused_or_accurate() {
    // A zero (2,2) block needs pivoting in general. Without it, either a
    // (near-)zero pivot is hit and reported, or the order happened to avoid
    // one and the answer must still be accurate; never a silent bad factor.
    let a = kkt::<c64>(600, 150, 3);
    match solve_ones_pivoting(&a, false, Par::Seq) {
        Err(rivrs_sparse::error::SparseError::NumericalSingularity { .. }) => {}
        Err(e) => panic!("unexpected error {e:?}"),
        Ok((be, _)) => assert!(
            be < 1e-10,
            "unpivoted KKT accepted with backward error {be:e}"
        ),
    }
}
