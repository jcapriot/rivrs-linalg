//! Scalar types supported by the symmetric solver.
//!
//! The numeric core is generic over [`Scalar`]: any faer [`ComplexField`]
//! that is `Copy` and whose real type implements [`Float`] (num-traits,
//! re-exported by faer as `faer::traits::num_traits`). Today that is `f64` and `f32` (real symmetric
//! indefinite) and [`c64`] and [`c32`] (complex
//! *symmetric*, A = Aᵀ, **not** Hermitian). The factorization computed is always
//! Pᵀ A P = L D Lᵀ with plain transposes; no conjugation is ever applied.
//!
//! Pivot magnitudes and stability tests are computed in the scalar's real type
//! `T::Real`, so single precision factors pivot in single precision. User-facing
//! thresholds (`AptpOptions::threshold`, `small`) stay `f64` and are converted
//! once. MC64 matching takes logarithms and always runs in `f64` on |A|.
//!
//! # Why these bounds
//!
//! - [`ComplexField`] supplies arithmetic (by value and by reference, via its
//!   `RefOps` supertrait), `Debug`, `PartialEq`, `Send + Sync` and the
//!   constructors used here.
//! - `Copy`: the dense kernels index and reuse scalars by value; faer only
//!   guarantees `Clone`.
//! - `Real: Float`: logarithms/exponentials for MC64 scaling and `to_f64` for
//!   reporting statistics. faer's `RealField` provides neither. This excludes
//!   faer's double-double `fx128`, which does not implement `Float`.
//!
//! # References
//!
//! - Duff & Reid (1983), "The multifrontal solution of indefinite sparse
//!   symmetric linear equations": the method applies unchanged to complex
//!   symmetric matrices when transposes are not conjugated.
//! - Amestoy et al. (2001), MUMPS: complex symmetric (SYM=2) LDLᵀ with
//!   1×1/2×2 pivots selected on moduli.

use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, ToPrimitive};
#[allow(unused_imports)] // c32/c64 are referenced by the docs
use faer::{c32, c64};

/// Element type of matrices the symmetric solver can factor.
///
/// Blanket-implemented for every `Copy` faer [`ComplexField`] whose real type
/// is a [`Float`]; there is no per-type code. The provided methods are thin
/// names for faer's `ComplexField` functions.
pub trait Scalar: ComplexField<Real: Float> + Copy {
    /// Modulus `|x|` in the real type (complex modulus for complex scalars).
    #[inline(always)]
    fn mag(self) -> Self::Real {
        Self::abs_impl(&self)
    }

    /// Embed a real `f64` constant.
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        Self::from_f64_impl(x)
    }

    /// Additive identity.
    #[inline(always)]
    fn zero() -> Self {
        Self::zero_impl()
    }

    /// Multiplicative identity.
    #[inline(always)]
    fn one() -> Self {
        Self::one_impl()
    }
}

impl<T: ComplexField<Real: Float> + Copy> Scalar for T {}

/// Convert an `f64` threshold or constant to the real type of `T`.
#[inline(always)]
pub(crate) fn real<T: Scalar>(x: f64) -> T::Real {
    <T::Real as ComplexField>::from_f64_impl(x)
}

/// Determinant of the symmetric 2×2 block `[[a11, a21], [a21, a22]]`, as
/// `(s, det_s)` with `det = det_s / s²`.
///
/// `s` is a power of two near `1/max|a_ij|`, so `det_s` is of order one.
/// Unscaled, `det = a11·a22 − a21²` is of order `|a|²`, and complex division
/// by it squares that again (`re² + im²`), which overflows or underflows
/// single precision for `|a|` around `1e±9`. A power of two scales exactly,
/// so wherever the unscaled formula stays in range, dividing by `det_s` and
/// rescaling by `s` gives bitwise the same result. `s = 1` for a zero or
/// non-finite block.
#[inline]
pub(crate) fn det_2x2_scaled<T: Scalar>(a11: T, a21: T, a22: T) -> (T, T) {
    let maxpiv = a11.mag().max(a21.mag()).max(a22.mag());
    let s = if maxpiv == real::<T>(0.0) || !maxpiv.is_finite() {
        T::one()
    } else {
        let e = maxpiv.log2().floor().to_i32().unwrap_or(0);
        T::from_real_impl(&real::<T>(2.0).powi(-e))
    };
    (s, (a11 * s) * (a22 * s) - (a21 * s) * (a21 * s))
}

/// `1/det` for the symmetric 2×2 block, without overflow (see
/// [`det_2x2_scaled`]).
#[inline]
pub(crate) fn inv_det_2x2<T: Scalar>(a11: T, a21: T, a22: T) -> T {
    let (s, det_s) = det_2x2_scaled(a11, a21, a22);
    (s / det_s) * s
}

/// Widen a real value to `f64` for reporting (statistics, MC64 input).
///
/// Every `Float` used here (`f32`, `f64`) converts losslessly; a type that
/// cannot is reported as NaN rather than panicking.
#[inline(always)]
pub(crate) fn to_f64<R: Float>(x: R) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}
