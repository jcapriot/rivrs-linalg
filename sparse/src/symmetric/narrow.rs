//! The transposed product `W -= L21ᵀ · T` for a narrow right-hand side
//! block.
//!
//! The backward multi-rhs solve applies `L21ᵀ` (`L21` is `r × ne`,
//! column-major) to blocks of `k` right-hand sides. faer's matmul handles
//! `k = 1` with a fast matrix-vector kernel, but for this transposed shape
//! with `2 <= k <= 8` it runs at a fraction of the core's FMA rate. On
//! bigbox (i9-14900K, one P-core) and a cache-resident `1500 × 600` front it
//! moves 21 GB/s of `f64` at `k = 2` against 95 GB/s at `k = 1`. The
//! subtree tasks of the threaded solve are mostly such fronts.
//!
//! The kernel here takes two columns of `L` against up to four columns of
//! the right-hand side at a time: eight dot products in AVX2 registers, with
//! the rows split in blocks so the two column pieces stay in L1 across the
//! right-hand side groups. Each column of `L` is read from memory once.
//!
//! Complex symmetric factors (`c64`, no conjugation) use the same real
//! kernel. With `L = Lr + i Li` stored interleaved, the same memory read as
//! a real `2r × ne` matrix `Lv` has rows alternating `Lr`, `Li`, and
//! `Lvᵀ · [ (Re T, -Im T), (Im T, Re T) ]` (rows interleaved) is
//! `[Re, Im]` of `Lᵀ T`. So `L` is read once by the real kernel with `2k`
//! columns, plus `O(r k)` packing.
//!
//! [`sub_tr_narrow`] declines (returns `false`, and the caller uses faer)
//! for other scalar types, `k = 1`, more than [`NARROW_MAX_REAL_COLS`]
//! real columns, or a CPU without AVX2 + FMA.
//!
//! # References
//!
//! - Goto & van de Geijn (2008), "Anatomy of high-performance matrix
//!   multiplication": register and cache blocking of the micro-kernel
//! - Higham (2002), "Accuracy and Stability of Numerical Algorithms",
//!   Section 3.1: inner-product error bounds hold for any summation order
//! - The real embedding of complex products is the standard `2 × 2` real
//!   form of a complex scalar (e.g. Golub & Van Loan (2013), Section 1.3.7)

use std::cell::RefCell;

use faer::traits::ComplexField;
use faer::{MatMut, MatRef, Par, c64};

/// Most real right-hand side columns handled here (`k` for `f64`, `2k`
/// for `c64`); wider blocks go to faer's matmul.
pub(crate) const NARROW_MAX_REAL_COLS: usize = 8;

/// Rows of `L` per block in the transposed kernel: the `Y` block (up to
/// 32 KB) and a pair of `L` column pieces (8 KB) stay in L1.
#[cfg(target_arch = "x86_64")]
const TR_ROW_BLOCK: usize = 512;

/// Minimum `L` entries per parallel task.
const PAR_MIN_ENTRIES: usize = 1 << 15;

thread_local! {
    /// Packing and output scratch (`f64`), reused across calls per thread.
    static SCRATCH: RefCell<(Vec<f64>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Element type as seen by the real kernel.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Real,
    Complex,
}

fn kind<T: ComplexField>() -> Option<Kind> {
    if const { T::IS_NATIVE_F64 } && size_of::<T>() == size_of::<f64>() {
        Some(Kind::Real)
    } else if const { T::IS_NATIVE_C64 } && size_of::<T>() == size_of::<c64>() {
        Some(Kind::Complex)
    } else {
        None
    }
}

/// Column-major `f64` view: `len` covers every column, `ld` >= rows.
#[derive(Clone, Copy)]
struct RealRef<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
    ld: usize,
}

/// Reinterpret a column-major (row stride 1) `f64` or `c64` matrix as `f64`
/// (`c64`: `2 × rows` real rows, column stride doubled).
fn real_view<T>(m: MatRef<'_, T>, kind: Kind) -> Option<RealRef<'_>> {
    if m.nrows() == 0 || m.ncols() == 0 {
        return None;
    }
    if m.row_stride() != 1 {
        return None;
    }
    let ld = if m.ncols() == 1 {
        m.nrows()
    } else {
        usize::try_from(m.col_stride()).ok()?
    };
    if ld < m.nrows() {
        return None;
    }
    let len = (m.ncols() - 1) * ld + m.nrows();
    let f = if kind == Kind::Complex { 2 } else { 1 };
    // SAFETY: `T` is `f64` or `c64` (checked by `kind`, which also checks
    // the size), `c64` is two `f64` (`repr(C)`), and the view covers only
    // entries of `m`, which outlives the returned borrow.
    let data = unsafe { core::slice::from_raw_parts(m.as_ptr().cast::<f64>(), len * f) };
    Some(RealRef {
        data,
        rows: m.nrows() * f,
        cols: m.ncols(),
        ld: ld * f,
    })
}

/// `out (ne × k) -= Lᵀ (ne × r) · Y (r × k)`; `L` column-major, no
/// conjugation.
///
/// Returns `false` (and leaves `out` alone) if this case isn't handled.
pub(crate) fn sub_tr_narrow<T: ComplexField + Copy>(
    mut out: MatMut<'_, T>,
    l: MatRef<'_, T>,
    y: MatRef<'_, T>,
    par: Par,
) -> bool {
    let k = y.ncols();
    let Some(kind) = kind::<T>() else {
        return false;
    };
    let n = if kind == Kind::Complex { 2 * k } else { k };
    if k < 2 || n > NARROW_MAX_REAL_COLS || !simd_available() {
        return false;
    }
    let Some(lv) = real_view(l, kind) else {
        return false;
    };
    let (r, ne) = (l.nrows(), l.ncols());
    debug_assert_eq!(
        (out.nrows(), out.ncols(), y.nrows(), lv.cols),
        (ne, k, r, ne)
    );

    // Take the buffers out for the call instead of holding the borrow: a
    // rayon worker waiting in `run_chunks` can steal another job that calls
    // this function again on the same thread, and a held `RefCell` borrow
    // would panic. A nested call just starts from empty buffers.
    let (mut pack, mut zbuf) =
        SCRATCH.with_borrow_mut(|(p, z)| (std::mem::take(p), std::mem::take(z)));
    {
        let (pack, zbuf) = (&mut pack, &mut zbuf);
        // Real Y, `m × n` column-major (m = lv.rows).
        let m = lv.rows;
        pack.clear();
        pack.resize(m * n, 0.0);
        for j in 0..k {
            for i in 0..r {
                let (re, im) = parts(y[(i, j)], kind);
                match kind {
                    Kind::Real => pack[i + j * m] = re,
                    Kind::Complex => {
                        pack[2 * i + j * m] = re;
                        pack[2 * i + 1 + j * m] = -im;
                        pack[2 * i + (k + j) * m] = im;
                        pack[2 * i + 1 + (k + j) * m] = re;
                    }
                }
            }
        }
        // Z (ne × n) row-major: column tasks of L own contiguous rows of Z.
        let tasks = task_count(m * ne, par).min(ne.div_ceil(8)).max(1);
        let cols_per_task = ne.div_ceil(tasks);
        zbuf.clear();
        zbuf.resize(ne * n, 0.0);
        let job = |(t, z): (usize, &mut [f64])| {
            let c0 = t * cols_per_task;
            let cb = z.len() / n;
            tr_dispatch(z, n, &lv, c0, cb, pack, m);
        };
        run_chunks(zbuf, cols_per_task * n, tasks, job);

        for j in 0..k {
            for c in 0..ne {
                let (re, im) = parts(out[(c, j)], kind);
                let zim = if kind == Kind::Complex {
                    zbuf[c * n + k + j]
                } else {
                    0.0
                };
                out[(c, j)] = from_parts(re - zbuf[c * n + j], im - zim, kind);
            }
        }
    }
    SCRATCH.with_borrow_mut(|(p, z)| {
        *p = pack;
        *z = zbuf;
    });
    true
}

/// Whether the AVX2 kernel can run (tests also exercise the portable one).
fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return true;
    }
    cfg!(test)
}

/// Real and imaginary parts of an `f64` or `c64` entry.
#[inline(always)]
fn parts<T: Copy>(v: T, kind: Kind) -> (f64, f64) {
    // SAFETY: `kind` was derived from `T` by `kind()`, which checked that
    // `T` is `f64` (`Real`) or `c64` (`Complex`), sizes included.
    unsafe {
        match kind {
            Kind::Real => (core::mem::transmute_copy::<T, f64>(&v), 0.0),
            Kind::Complex => {
                let z = core::mem::transmute_copy::<T, c64>(&v);
                (z.re, z.im)
            }
        }
    }
}

/// Inverse of [`parts`].
#[inline(always)]
fn from_parts<T: Copy>(re: f64, im: f64, kind: Kind) -> T {
    // SAFETY: as in `parts`.
    unsafe {
        match kind {
            Kind::Real => core::mem::transmute_copy::<f64, T>(&re),
            Kind::Complex => core::mem::transmute_copy::<c64, T>(&c64::new(re, im)),
        }
    }
}

/// Number of tasks for `entries` entries of `L` (1 when sequential).
fn task_count(entries: usize, par: Par) -> usize {
    let threads = par.degree();
    if threads < 2 {
        return 1;
    }
    (4 * threads).min(entries / PAR_MIN_ENTRIES).max(1)
}

/// Run `job` on chunks of `buf`, in parallel when there is more than one.
fn run_chunks<F>(buf: &mut [f64], chunk: usize, tasks: usize, job: F)
where
    F: Fn((usize, &mut [f64])) + Sync + Send,
{
    if tasks > 1 {
        use rayon::prelude::*;
        buf.par_chunks_mut(chunk).enumerate().for_each(job);
    } else {
        buf.chunks_mut(chunk).enumerate().for_each(job);
    }
}

// ---------------------------------------------------------------------------
// Real kernels. `FMA` selects fused multiply-add (only when the target or
// the run-time check guarantees the instruction; otherwise `mul_add` would
// be a libm call).

#[inline(always)]
fn fma<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `z[(c - c0) * n + j] = L[:, c] · Y[:, j]` for `c` in `c0..c0 + cb`.
#[inline(always)]
fn tr_kernel<const FMA: bool>(
    z: &mut [f64],
    n: usize,
    l: &RealRef<'_>,
    c0: usize,
    cb: usize,
    y: &[f64],
    m: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        tr_block::<FMA, 4>(z, n, j, l, c0, cb, y, m);
        j += 4;
    }
    if j + 2 <= n {
        tr_block::<FMA, 2>(z, n, j, l, c0, cb, y, m);
        j += 2;
    }
    if j < n {
        tr_block::<FMA, 1>(z, n, j, l, c0, cb, y, m);
    }
}

/// [`tr_kernel`] for right-hand sides `j0..j0 + JB`: two columns of `L`
/// against `JB` columns of `Y`, `2 * JB` four-lane accumulators.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tr_block<const FMA: bool, const JB: usize>(
    z: &mut [f64],
    n: usize,
    j0: usize,
    l: &RealRef<'_>,
    c0: usize,
    cb: usize,
    y: &[f64],
    m: usize,
) {
    let lcol = |c: usize| &l.data[c * l.ld..][..m];
    let ys: [&[f64]; JB] = core::array::from_fn(|jj| &y[(j0 + jj) * m..][..m]);
    let m4 = m - m % 4;
    let mut c = 0;
    while c < cb {
        let two = c + 1 < cb;
        let la = lcol(c0 + c);
        let lb = if two { lcol(c0 + c + 1) } else { la };
        let mut a = [[0.0f64; 4]; JB];
        let mut b = [[0.0f64; 4]; JB];
        for ((la4, lb4), i) in la[..m4]
            .chunks_exact(4)
            .zip(lb[..m4].chunks_exact(4))
            .zip((0..m4).step_by(4))
        {
            for jj in 0..JB {
                let y4 = &ys[jj][i..i + 4];
                for q in 0..4 {
                    a[jj][q] = fma::<FMA>(la4[q], y4[q], a[jj][q]);
                    b[jj][q] = fma::<FMA>(lb4[q], y4[q], b[jj][q]);
                }
            }
        }
        for jj in 0..JB {
            let mut sa = (a[jj][0] + a[jj][1]) + (a[jj][2] + a[jj][3]);
            let mut sb = (b[jj][0] + b[jj][1]) + (b[jj][2] + b[jj][3]);
            for ii in m4..m {
                sa = fma::<FMA>(la[ii], ys[jj][ii], sa);
                sb = fma::<FMA>(lb[ii], ys[jj][ii], sb);
            }
            z[c * n + j0 + jj] = sa;
            if two {
                z[(c + 1) * n + j0 + jj] = sb;
            }
        }
        c += if two { 2 } else { 1 };
    }
}

fn tr_dispatch(
    z: &mut [f64],
    n: usize,
    l: &RealRef<'_>,
    c0: usize,
    cb: usize,
    y: &[f64],
    m: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // Row blocks keep the Y block and a pair of L column pieces in L1
        // while every right-hand side group runs against the pair; partial
        // dots accumulate in z.
        z.fill(0.0);
        let mut i0 = 0;
        while i0 < m {
            let i1 = (i0 + TR_ROW_BLOCK).min(m);
            // SAFETY: the features were just detected; bounds are checked
            // inside.
            unsafe { tr_rows_avx2(z, n, l, c0, cb, y, m, i0, i1) };
            i0 = i1;
        }
        return;
    }
    tr_kernel::<{ cfg!(target_arch = "aarch64") }>(z, n, l, c0, cb, y, m);
}

/// AVX2 kernel over rows `i0..i1`: for each pair of columns of `L`, every
/// group of up to four right-hand sides, adding the partial dots to `z`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn tr_rows_avx2(
    z: &mut [f64],
    n: usize,
    l: &RealRef<'_>,
    c0: usize,
    cb: usize,
    y: &[f64],
    m: usize,
    i0: usize,
    i1: usize,
) {
    assert!(z.len() >= cb * n && y.len() >= n * m);
    assert!(c0 + cb <= l.cols && m <= l.rows && (c0 + cb - 1) * l.ld + m <= l.data.len());
    assert!(i0 <= i1 && i1 <= m);
    let col = |c: usize| &l.data[(c0 + c) * l.ld + i0..(c0 + c) * l.ld + i1];
    let mut c = 0;
    while c < cb {
        let two = c + 1 < cb;
        let la = col(c);
        let lb = if two { col(c + 1) } else { la };
        let mut j = 0;
        while j < n {
            let jb = (n - j).min(4);
            // Groups of fewer than four repeat the last column (unused).
            let yc = |jj: usize| {
                let q = j + jj.min(jb - 1);
                &y[q * m + i0..q * m + i1]
            };
            let ys = [yc(0), yc(1), yc(2), yc(3)];
            let (sa, sb) = match jb {
                4 => pair_dots_avx2::<4>(la, lb, ys),
                3 => pair_dots_avx2::<3>(la, lb, ys),
                2 => pair_dots_avx2::<2>(la, lb, ys),
                _ => pair_dots_avx2::<1>(la, lb, ys),
            };
            for jj in 0..jb {
                z[c * n + j + jj] += sa[jj];
                if two {
                    z[(c + 1) * n + j + jj] += sb[jj];
                }
            }
            j += jb;
        }
        c += if two { 2 } else { 1 };
    }
}

/// Dot products of `la` and `lb` with the first `JB` of `ys` (all the same
/// length), as `([la · y], [lb · y])`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn pair_dots_avx2<const JB: usize>(
    la: &[f64],
    lb: &[f64],
    ys: [&[f64]; 4],
) -> ([f64; 4], [f64; 4]) {
    use core::arch::x86_64::*;
    let len = la.len();
    assert!(lb.len() == len && ys.iter().all(|y| y.len() == len));
    let m4 = len - len % 4;
    let hsum = |v: __m256d| {
        let lo = _mm256_castpd256_pd128(v);
        let hi = _mm256_extractf128_pd::<1>(v);
        let s = _mm_add_pd(lo, hi);
        _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
    };
    let (pa, pb) = (la.as_ptr(), lb.as_ptr());
    let py: [*const f64; 4] = core::array::from_fn(|jj| ys[jj].as_ptr());
    let mut a = [_mm256_setzero_pd(); JB];
    let mut b = [_mm256_setzero_pd(); JB];
    let mut i = 0;
    while i < m4 {
        // SAFETY: i + 4 <= len for every slice (asserted equal lengths).
        unsafe {
            let va = _mm256_loadu_pd(pa.add(i));
            let vb = _mm256_loadu_pd(pb.add(i));
            for jj in 0..JB {
                let yv = _mm256_loadu_pd(py[jj].add(i));
                a[jj] = _mm256_fmadd_pd(va, yv, a[jj]);
                b[jj] = _mm256_fmadd_pd(vb, yv, b[jj]);
            }
        }
        i += 4;
    }
    let mut sa = [0.0; 4];
    let mut sb = [0.0; 4];
    for jj in 0..JB {
        sa[jj] = hsum(a[jj]);
        sb[jj] = hsum(b[jj]);
        for ii in m4..len {
            sa[jj] = la[ii].mul_add(ys[jj][ii], sa[jj]);
            sb[jj] = lb[ii].mul_add(ys[jj][ii], sb[jj]);
        }
    }
    (sa, sb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    fn check<T: ComplexField<Real = f64> + Copy>(
        r: usize,
        ne: usize,
        k: usize,
        f: impl Fn(usize, usize, usize) -> T,
        par: Par,
    ) {
        let l = Mat::<T>::from_fn(r, ne, |i, j| f(i, j, 0));
        let y = Mat::<T>::from_fn(r, k, |i, j| f(i, j, 2));
        let o0 = Mat::<T>::from_fn(ne, k, |i, j| f(i, j, 3));
        let mut want = o0.clone();
        matmul(
            want.as_mut(),
            Accum::Add,
            l.as_ref().transpose(),
            y.as_ref(),
            T::from_f64_impl(-1.0),
            Par::Seq,
        );
        let mut got = o0.clone();
        assert!(sub_tr_narrow(got.as_mut(), l.as_ref(), y.as_ref(), par));
        let err = (&got - &want).norm_max() / want.norm_max().max(1.0);
        assert!(err < 1e-13, "sub_tr r={r} ne={ne} k={k}: {err:e}");
    }

    fn val(i: usize, j: usize, s: usize) -> f64 {
        (((i * 31 + j * 17 + s * 7) % 23) as f64 - 11.0) / 7.0
    }

    /// Called from inside another parallel loop, a worker waiting on its
    /// chunks can steal a job that re-enters `sub_tr_narrow` on the same
    /// thread. That used to panic on the held scratch `RefCell` borrow.
    #[test]
    fn narrow_reentrant_under_nested_rayon() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..32).into_par_iter().for_each(|s| {
            check::<f64>(2500, 300, 4, |i, j, t| val(i, j, t + s), Par::rayon(8));
        });
    }

    #[test]
    fn narrow_transposed_product_matches_matmul() {
        // Shapes cover row remainders, odd column counts, and more rows
        // than one row block.
        let shapes = [(1, 1), (3, 2), (7, 5), (130, 9), (257, 131), (2500, 300)];
        for &(r, ne) in &shapes {
            for par in [Par::Seq, Par::rayon(4)] {
                for k in 2..=NARROW_MAX_REAL_COLS {
                    check::<f64>(r, ne, k, val, par);
                }
                for k in 2..=NARROW_MAX_REAL_COLS / 2 {
                    let z = |i, j, s| c64::new(val(i, j, s), val(j, i, s + 5));
                    check::<c64>(r, ne, k, z, par);
                }
            }
        }
    }

    #[test]
    fn narrow_declines_other_cases() {
        let l = Mat::<f64>::zeros(4, 3);
        let y1 = Mat::<f64>::zeros(4, 1);
        let mut o1 = Mat::<f64>::zeros(3, 1);
        assert!(!sub_tr_narrow(
            o1.as_mut(),
            l.as_ref(),
            y1.as_ref(),
            Par::Seq
        ));
        let lf = Mat::<f32>::zeros(4, 3);
        let yf = Mat::<f32>::zeros(4, 2);
        let mut of = Mat::<f32>::zeros(3, 2);
        assert!(!sub_tr_narrow(
            of.as_mut(),
            lf.as_ref(),
            yf.as_ref(),
            Par::Seq
        ));
        let k = NARROW_MAX_REAL_COLS + 1;
        let yw = Mat::<f64>::zeros(4, k);
        let mut ow = Mat::<f64>::zeros(3, k);
        assert!(!sub_tr_narrow(
            ow.as_mut(),
            l.as_ref(),
            yw.as_ref(),
            Par::Seq
        ));
        let lc = Mat::<c64>::zeros(4, 3);
        let kc = NARROW_MAX_REAL_COLS / 2 + 1;
        let yc = Mat::<c64>::zeros(4, kc);
        let mut oc = Mat::<c64>::zeros(3, kc);
        assert!(!sub_tr_narrow(
            oc.as_mut(),
            lc.as_ref(),
            yc.as_ref(),
            Par::Seq
        ));
    }
}
