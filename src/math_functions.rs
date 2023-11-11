//! Math functions.

use crate::{
    float_type::float,
    linalg_types::RowVector,
};


pub fn exp(x: float) -> float {
    x.exp()
}

pub fn sqrt(x: float) -> float {
    x.sqrt()
}


// Activation functions:

pub fn sigmoid(x: float) -> float {
    1. / (1. + exp(-x))
}
pub fn sigmoid_prime(x: float) -> float {
    let y = sigmoid(x);
    y * (1. - y)
}

pub fn tanh(x: float) -> float {
    x.tanh()
}
pub fn tanh_prime(x: float) -> float {
    1. - tanh(x).powi(2)
}



// Vectorized activation functions:

// TODO(optimize): vectorize them

pub fn sigmoid_v(v: RowVector) -> RowVector {
    v.map(|x| sigmoid(x))
}
pub fn sigmoid_prime_v(v: RowVector) -> RowVector {
    v.map(|x| sigmoid_prime(x))
}

pub fn tanh_v(v: RowVector) -> RowVector {
    v.map(|x| tanh(x))
}
pub fn tanh_prime_v(v: RowVector) -> RowVector {
    v.map(|x| tanh_prime(x))
}

