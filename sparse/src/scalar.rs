//! Scalar types supported by the symmetric solver.
//!
//! The numeric core is generic over [`Scalar`], implemented for `f64` (real
//! symmetric indefinite) and [`c64`](faer::c64) (complex *symmetric*,
//! A = Aᵀ, **not** Hermitian), plus the single-precision `f32` and
//! [`c32`](faer::c32). The factorization computed is always
//! Pᵀ A P = L D Lᵀ with plain transposes; no conjugation is ever applied.
//!
//! Pivot magnitudes, thresholds and scaling factors are `f64` for every scalar
//! type: APTP compares moduli, so the pivoting logic is shared. Single
//! precision types widen their moduli to `f64` (exact) for those tests.
//!
//! # References
//!
//! - Duff & Reid (1983), "The multifrontal solution of indefinite sparse
//!   symmetric linear equations": the method applies unchanged to complex
//!   symmetric matrices when transposes are not conjugated.
//! - Amestoy et al. (2001), MUMPS: complex symmetric (SYM=2) LDLᵀ with
//!   1×1/2×2 pivots selected on moduli.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use faer::traits::ComplexField;
use faer::{c32, c64};

/// Element type of matrices the symmetric solver can factor.
///
/// This is a thin extension of faer's [`ComplexField`]: it adds by-value
/// arithmetic operators (the dense kernels are written with `a * b` on
/// values), `Copy + Send + Sync` for rayon, and an `f64` modulus so all APTP
/// thresholds stay `f64` across precisions. Every method delegates to faer.
///
/// Implemented for `f64`, `f32`, `c64` and `c32`. All arithmetic in the solver goes through
/// the standard operator traits, so any implementor must behave like a field
/// under `+ - * /`; [`mag`](Scalar::mag) must be a norm (|x| or the complex
/// modulus) because pivot acceptance tests compare it against `f64` bounds.
pub trait Scalar:
    ComplexField
    + Copy
    + Send
    + Sync
    + Debug
    + PartialEq
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Whether this is a complex type. Inertia is only defined when `false`.
    const IS_COMPLEX: bool;

    /// Modulus as `f64`: |x| for reals, hypot(re, im) for complex.
    fn mag(self) -> f64;

    /// Embed a real number.
    fn from_f64(x: f64) -> Self;

    /// Additive identity.
    #[inline(always)]
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// Multiplicative identity.
    #[inline(always)]
    fn one() -> Self {
        Self::from_f64(1.0)
    }

    /// Real part (the value itself for `f64`).
    fn real_part(self) -> f64;
}

macro_rules! impl_scalar {
    ($t:ty, $complex:expr) => {
        impl Scalar for $t {
            const IS_COMPLEX: bool = $complex;

            #[inline(always)]
            fn mag(self) -> f64 {
                <$t as ComplexField>::abs_impl(&self) as f64
            }

            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                <$t as ComplexField>::from_f64_impl(x)
            }

            #[inline(always)]
            fn zero() -> Self {
                <$t as ComplexField>::zero_impl()
            }

            #[inline(always)]
            fn one() -> Self {
                <$t as ComplexField>::one_impl()
            }

            #[inline(always)]
            fn real_part(self) -> f64 {
                <$t as ComplexField>::real_part_impl(&self) as f64
            }
        }
    };
}

// Thin layer over faer's `ComplexField`: values, operators and f64 moduli.
impl_scalar!(f64, false);
impl_scalar!(f32, false);
impl_scalar!(c64, true);
impl_scalar!(c32, true);
