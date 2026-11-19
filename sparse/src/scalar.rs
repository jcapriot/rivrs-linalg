//! Scalar type switch for the symmetric solver (prototype).
//!
//! With the `complex` feature the numeric core works on `c64` and computes a
//! complex *symmetric* (A = A^T, not Hermitian) LDL^T. Without it, `S = f64`
//! and behaviour is unchanged. Pivot magnitudes, thresholds and scaling
//! factors stay `f64` in both modes.

#[cfg(not(feature = "complex"))]
pub type S = f64;
#[cfg(feature = "complex")]
pub type S = faer::c64;

/// Modulus of a scalar as `f64` (|x| for reals, hypot(re, im) for complex).
pub trait Mag: Copy {
    fn mag(self) -> f64;
}
impl Mag for f64 {
    #[inline(always)]
    fn mag(self) -> f64 {
        self.abs()
    }
}
impl Mag for faer::c64 {
    #[inline(always)]
    fn mag(self) -> f64 {
        self.re.hypot(self.im)
    }
}

#[inline(always)]
pub fn s(x: f64) -> S {
    S::from(x)
}

#[inline(always)]
pub fn zero() -> S {
    s(0.0)
}

/// Real part (identity for `f64`). Used only where the real solver needs a
/// sign (inertia); for complex symmetric matrices inertia is not defined and
/// the counts it produces are not meaningful.
#[cfg(not(feature = "complex"))]
#[inline(always)]
pub fn re(x: S) -> f64 {
    x
}
#[cfg(feature = "complex")]
#[inline(always)]
pub fn re(x: S) -> f64 {
    x.re
}
