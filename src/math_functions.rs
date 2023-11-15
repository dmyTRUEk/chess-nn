//! Math functions.

use crate::{
    float_type::float,
    linalg_types::RowVector,
};


// mathods to functions aliases
pub fn abs(x: float) -> float { x.abs() }
pub fn exp(x: float) -> float { x.exp() }
pub fn ln(x: float) -> float { x.ln() }
pub fn signum(x: float) -> float { x.signum() }
pub fn sqrt(x: float) -> float { x.sqrt() }
pub fn tanh(x: float) -> float { x.tanh() }

// pub fn kronecker_delta_f<T: Eq>(i: T, j: T) -> float {
//     if i == j { 1. } else { 0. }
// }


// Activation functions:

// pub fn abs(x: float) -> float {
//     abs(x)
// }
pub fn abs_prime(x: float) -> float {
    signum(x)
}

pub fn binary_step(x: float) -> float {
    if x < 0. { 0. } else { 1. }
}
pub fn binary_step_prime(_x: float) -> float {
    0.
}

pub fn elu(x: float) -> float {
    const ALPHA: float = 1.;
    if x < 0. { ALPHA * (exp(x) - 1.) } else { x }
}
pub fn elu_prime(x: float) -> float {
    const ALPHA: float = 1.; // MUST BE IN SYNC WITH [`elu::ALPHA`]
    if x < 0. { ALPHA * exp(x) } else { 1. }
}

pub fn gaussian(x: float) -> float {
    exp(-x.powi(2))
}
pub fn gaussian_prime(x: float) -> float {
    -2. * x * gaussian(x)
}

pub fn leaky_relu(x: float) -> float {
    const ALPHA: float = 0.01;
    if x < 0. { ALPHA * x } else { x }
}
pub fn leaky_relu_prime(x: float) -> float {
    const ALPHA: float = 0.01; // MUST BE IN SYNC WITH [`leaky_relu::ALPHA`]
    if x < 0. { ALPHA } else { 1. }
}

// Only vectorized forms exists
// pub fn max_out(x: float) -> float {}
// pub fn max_out_prime(x: float) -> float {}

pub fn relu(x: float) -> float {
    x.max(0.)
}
pub fn relu_prime(x: float) -> float {
    binary_step(x)
}

pub fn silu(x: float) -> float {
    x / (1. + exp(-x))
}
pub fn silu_prime(x: float) -> float {
    let exp_m_x: float = exp(-x);
    (1. + exp_m_x + x*exp_m_x) / (1. + exp_m_x).powi(2)
}

pub fn sigmoid(x: float) -> float {
    1. / (1. + exp(-x))
}
pub fn sigmoid_prime(x: float) -> float {
    let y = sigmoid(x);
    y * (1. - y)
}

// pub fn signum(x: float) -> float {
//     signum(x)
// }
pub fn signum_prime(_x: float) -> float {
    0.
}

pub fn sign_sqrt_abs(x: float) -> float {
    signum(x) * sqrt(abs(x))
}
pub fn sign_sqrt_abs_prime(x: float) -> float {
    const MAX_VALUE: float = 1e1;
    0.5 * signum(x) / sqrt(abs(x)).min(MAX_VALUE)
}

// Only vectorized forms exists
// pub fn soft_max(x: float) -> float {}
// pub fn soft_max_prime(x: float) -> float {}

pub fn soft_plus(x: float) -> float {
    ln(1. + exp(x))
}
pub fn soft_plus_prime(x: float) -> float {
    1. / (1. + exp(-x))
}

// pub fn tanh(x: float) -> float {
//     tanh(x)
// }
pub fn tanh_prime(x: float) -> float {
    1. - tanh(x).powi(2)
}



// Vectorized activation functions:

// TODO(optimize): vectorize them?
// TODO(refactor): write macros, that must be applied on not vectorized function, and it will automatically create vectorized function

pub fn abs_v(v: RowVector) -> RowVector {
    v.map(abs)
}
pub fn abs_prime_v(v: RowVector) -> RowVector {
    v.map(abs_prime)
}

pub fn binary_step_v(v: RowVector) -> RowVector {
    v.map(binary_step)
}
pub fn binary_step_prime_v(v: RowVector) -> RowVector {
    v.map(binary_step_prime)
}

pub fn elu_v(v: RowVector) -> RowVector {
    v.map(elu)
}
pub fn elu_prime_v(v: RowVector) -> RowVector {
    v.map(elu_prime)
}

pub fn gaussian_v(v: RowVector) -> RowVector {
    v.map(gaussian)
}
pub fn gaussian_prime_v(v: RowVector) -> RowVector {
    v.map(gaussian_prime)
}

pub fn leaky_relu_v(v: RowVector) -> RowVector {
    v.map(leaky_relu)
}
pub fn leaky_relu_prime_v(v: RowVector) -> RowVector {
    v.map(leaky_relu_prime)
}

pub fn max_out_v(v: RowVector) -> RowVector {
    // TODO: recheck
    RowVector::from_element(v.len(), v.max())
}
pub fn max_out_prime_v(v: RowVector) -> RowVector {
    // TODO: recheck
    let mut res = RowVector::zeros(v.len());
    let v_max = v.max();
    let index_of_max = v.into_iter().position(|&el| el == v_max).unwrap_or(0); // None if v==[NaN, …]
    res[index_of_max] = 1.;
    res
}

pub fn relu_v(v: RowVector) -> RowVector {
    v.map(relu)
}
pub fn relu_prime_v(v: RowVector) -> RowVector {
    v.map(relu_prime)
}

pub fn silu_v(v: RowVector) -> RowVector {
    v.map(silu)
}
pub fn silu_prime_v(v: RowVector) -> RowVector {
    v.map(silu_prime)
}

pub fn sigmoid_v(v: RowVector) -> RowVector {
    v.map(sigmoid)
}
pub fn sigmoid_prime_v(v: RowVector) -> RowVector {
    v.map(sigmoid_prime)
}

pub fn signum_v(v: RowVector) -> RowVector {
    v.map(signum)
}
pub fn signum_prime_v(v: RowVector) -> RowVector {
    v.map(signum_prime)
}

pub fn sign_sqrt_abs_v(v: RowVector) -> RowVector {
    v.map(sign_sqrt_abs)
}
pub fn sign_sqrt_abs_prime_v(v: RowVector) -> RowVector {
    v.map(sign_sqrt_abs_prime)
}

// pub fn soft_max_v(v: RowVector) -> RowVector {
//     let exp_v = v.map(exp);
//     let exp_sum: float = exp_v.sum();
//     exp_v / exp_sum
// }
/// This is stable version of SoftMax.
/// src: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
pub fn soft_max_v(mut v: RowVector) -> RowVector {
    v -= RowVector::from_element(v.len(), v.max());
    let exp_v = v.map(exp);
    let exp_sum: float = exp_v.sum();
    exp_v / exp_sum
}
pub fn soft_max_prime_v(v: RowVector) -> RowVector {
    todo!()
}

pub fn soft_plus_v(v: RowVector) -> RowVector {
    v.map(soft_plus)
}
pub fn soft_plus_prime_v(v: RowVector) -> RowVector {
    v.map(soft_plus_prime)
}

pub fn tanh_v(v: RowVector) -> RowVector {
    v.map(tanh)
}
pub fn tanh_prime_v(v: RowVector) -> RowVector {
    v.map(tanh_prime)
}

