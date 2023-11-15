//! Activation Function Tanh

use crate::{
    float_type::float,
    math_functions::{abs_prime_v, abs_v, binary_step_prime_v, binary_step_v, elu_prime_v, elu_v, gaussian_prime_v, gaussian_v, leaky_relu_prime_v, leaky_relu_v, max_out_prime_v, max_out_v, relu_prime_v, relu_v, sigmoid_prime_v, sigmoid_v, sign_sqrt_abs_prime_v, sign_sqrt_abs_v, signum_prime_v, signum_v, silu_prime_v, silu_v, soft_max_prime_v, soft_max_v, soft_plus_prime_v, soft_plus_v, tanh_prime_v, tanh_v},
};

use super::super::vector_type::Vector;

use super::Layer;


// Yeah, this code style may be unideomatic, but in this scenario it's just so much better.
#[allow(non_camel_case_types)] pub(super) type AF_Abs = ActivationFunction<ABS>;
#[allow(non_camel_case_types)] pub(super) type AF_BinaryStep = ActivationFunction<BINARY_STEP>;
#[allow(non_camel_case_types)] pub(super) type AF_Elu = ActivationFunction<ELU>;
#[allow(non_camel_case_types)] pub(super) type AF_Gaussian = ActivationFunction<GAUSSIAN>;
#[allow(non_camel_case_types)] pub(super) type AF_LeakyRelu = ActivationFunction<LEAKY_RELU>;
#[allow(non_camel_case_types)] pub(super) type AF_MaxOut = ActivationFunction<MAX_OUT>;
#[allow(non_camel_case_types)] pub(super) type AF_Relu = ActivationFunction<RELU>;
#[allow(non_camel_case_types)] pub(super) type AF_Sigmoid = ActivationFunction<SIGMOID>;
#[allow(non_camel_case_types)] pub(super) type AF_Signum = ActivationFunction<SIGNUM>;
#[allow(non_camel_case_types)] pub(super) type AF_SignSqrtAbs = ActivationFunction<SIGN_SQRT_ABS>;
#[allow(non_camel_case_types)] pub(super) type AF_Silu = ActivationFunction<SILU>;
#[allow(non_camel_case_types)] pub(super) type AF_SoftMax = ActivationFunction<SOFT_MAX>;
#[allow(non_camel_case_types)] pub(super) type AF_SoftPlus = ActivationFunction<SOFT_PLUS>;
#[allow(non_camel_case_types)] pub(super) type AF_Tanh = ActivationFunction<TANH>;

const ABS          : u8 = 0;
const BINARY_STEP  : u8 = 1;
const ELU          : u8 = 2;
const GAUSSIAN     : u8 = 3;
const LEAKY_RELU   : u8 = 4;
const MAX_OUT      : u8 = 5;
const RELU         : u8 = 6;
const SIGMOID      : u8 = 7;
const SIGNUM       : u8 = 8;
const SIGN_SQRT_ABS: u8 = 9;
const SILU         : u8 = 10;
const SOFT_MAX     : u8 = 11;
const SOFT_PLUS    : u8 = 12;
const TANH         : u8 = 13;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ActivationFunction<const F_INDEX: u8> {
    size: usize,
    input: Option<Vector>,
    output: Option<Vector>,
}

impl<const F_INDEX: u8> ActivationFunction<F_INDEX> {
    const F: fn(Vector) -> Vector = match F_INDEX {
        ABS => abs_v,
        BINARY_STEP => binary_step_v,
        ELU => elu_v,
        GAUSSIAN => gaussian_v,
        LEAKY_RELU => leaky_relu_v,
        MAX_OUT => max_out_v,
        RELU => relu_v,
        SIGMOID => sigmoid_v,
        SIGNUM => signum_v,
        SIGN_SQRT_ABS => sign_sqrt_abs_v,
        SILU => silu_v,
        SOFT_MAX => soft_max_v,
        SOFT_PLUS => soft_plus_v,
        TANH => tanh_v,
        _ => unreachable!()
    };

    const F_PRIME: fn(Vector) -> Vector = match F_INDEX {
        ABS => abs_prime_v,
        BINARY_STEP => binary_step_prime_v,
        ELU => elu_prime_v,
        GAUSSIAN => gaussian_prime_v,
        LEAKY_RELU => leaky_relu_prime_v,
        MAX_OUT => max_out_prime_v,
        RELU => relu_prime_v,
        SIGMOID => sigmoid_prime_v,
        SIGNUM => signum_prime_v,
        SIGN_SQRT_ABS => sign_sqrt_abs_prime_v,
        SILU => silu_prime_v,
        SOFT_MAX => soft_max_prime_v,
        SOFT_PLUS => soft_plus_prime_v,
        TANH => tanh_prime_v,
        _ => unreachable!()
    };

    pub(super) fn new(size: usize) -> Self {
        Self {
            size,
            input: None,
            output: None,
        }
    }

    fn forward_propagation(&self, input: Vector) -> Vector {
        // assert_eq!(self.size, input.len());
        // println!("input: {input:?}");
        let output = Self::F(input);
        // println!("output: {output:?}");
        output
    }

    fn forward_propagation_for_training(&mut self, input: Vector) -> Vector {
        // assert_eq!(self.size, input.len());
        self.input = Some(input.clone());
        let output = self.forward_propagation(input);
        self.output = Some(output.clone());
        output
    }

    fn backward_propagation(&mut self, output_error: Vector, _learning_rate: float) -> Vector {
        // assert_eq!(self.size, output_error.len());
        Self::F_PRIME(self.input.clone().unwrap()).component_mul(&output_error)
    }
}

impl<const F_INDEX: u8> Layer for ActivationFunction<F_INDEX> {
    fn forward_propagation(&self, input: Vector) -> Vector {
        self.forward_propagation(input)
    }

    fn forward_propagation_for_training(&mut self, input: Vector) -> Vector {
        self.forward_propagation_for_training(input)
    }

    fn backward_propagation(&mut self, output_error: Vector, learning_rate: float) -> Vector {
        self.backward_propagation(output_error, learning_rate)
    }
}

impl<const F_INDEX: u8> ToString for ActivationFunction<F_INDEX> {
    fn to_string(&self) -> String {
        match F_INDEX {
            ABS => format!("{}({})", stringify!(AF_Abs), self.size),
            BINARY_STEP => format!("{}({})", stringify!(AF_BinaryStep), self.size),
            ELU => format!("{}({})", stringify!(AF_Elu), self.size),
            GAUSSIAN => format!("{}({})", stringify!(AF_Gaussian), self.size),
            LEAKY_RELU => format!("{}({})", stringify!(AF_LeakyRelu), self.size),
            RELU => format!("{}({})", stringify!(AF_Relu), self.size),
            SIGMOID => format!("{}({})", stringify!(AF_Sigmoid), self.size),
            SIGNUM => format!("{}({})", stringify!(AF_Signum), self.size),
            SIGN_SQRT_ABS => format!("{}({})", stringify!(AF_SignSqrtAbs), self.size),
            SILU => format!("{}({})", stringify!(AF_Silu), self.size),
            SOFT_MAX => format!("{}({})", stringify!(AF_SoftMax), self.size),
            SOFT_PLUS => format!("{}({})", stringify!(AF_SoftPlus), self.size),
            TANH => format!("{}({})", stringify!(AF_Tanh), self.size),
            _ => unreachable!()
        }
    }
}



// RESEARCH ZONE:

// requires `adt_const_params` feature
// use std::marker::ConstParamTy;

// #[derive(PartialEq, Eq)]
// enum Type {
//     A, B
// }
// impl ConstParamTy for Type {}

// struct Foo<const T: Type> {}

// impl<const T: Type> Foo<T> {
//     const F: fn(float) -> float = match T {
//         Type::A => abs,
//         Type::B => todo!(),
//     };
// }

// const FOO_A: Foo = Foo<Type::A>;
// const FOO_B: Foo = Foo<Type::B>;

