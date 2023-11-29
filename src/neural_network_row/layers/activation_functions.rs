//! Activation Function Tanh

use crate::{
    float_type::float,
    math_functions::{
        abs_prime_v,
        abs_v,
        binary_step_prime_v,
        binary_step_v,
        elu_prime_v,
        elu_v,
        gaussian_prime_v,
        gaussian_v,
        leaky_relu_prime_v,
        leaky_relu_v,
        max_out_prime_v,
        max_out_v,
        relu_prime_v,
        relu_v,
        sigmoid_prime_v,
        sigmoid_v,
        signum_prime_v,
        signum_v,
        silu_prime_v,
        silu_v,
        soft_max_prime_v,
        soft_max_v,
        soft_plus_prime_v,
        soft_plus_v,
        sym_ln_prime_v,
        sym_ln_v,
        sym_sqrt_prime_v,
        sym_sqrt_v,
        tanh_prime_v,
        tanh_v,
    },
};

use super::super::vector_type::Vector;

use super::Layer;


// Yeah, this code style may be unideomatic, but in this scenario it's just so much better.
#[expect(non_camel_case_types)] pub(super) type AF_Abs = ActivationFunction<ABS>;
#[expect(non_camel_case_types)] pub(super) type AF_BinaryStep = ActivationFunction<BINARY_STEP>;
#[expect(non_camel_case_types)] pub(super) type AF_Elu = ActivationFunction<ELU>;
#[expect(non_camel_case_types)] pub(super) type AF_Gaussian = ActivationFunction<GAUSSIAN>;
#[expect(non_camel_case_types)] pub(super) type AF_LeakyRelu = ActivationFunction<LEAKY_RELU>;
#[expect(non_camel_case_types)] pub(super) type AF_MaxOut = ActivationFunction<MAX_OUT>;
#[expect(non_camel_case_types)] pub(super) type AF_Relu = ActivationFunction<RELU>;
#[expect(non_camel_case_types)] pub(super) type AF_Sigmoid = ActivationFunction<SIGMOID>;
#[expect(non_camel_case_types)] pub(super) type AF_Signum = ActivationFunction<SIGNUM>;
#[expect(non_camel_case_types)] pub(super) type AF_SymLn = ActivationFunction<SYM_LN>;
#[expect(non_camel_case_types)] pub(super) type AF_SymSqrt = ActivationFunction<SYM_SQRT>;
#[expect(non_camel_case_types)] pub(super) type AF_Silu = ActivationFunction<SILU>;
#[expect(non_camel_case_types)] pub(super) type AF_SoftMax = ActivationFunction<SOFT_MAX>;
#[expect(non_camel_case_types)] pub(super) type AF_SoftPlus = ActivationFunction<SOFT_PLUS>;
#[expect(non_camel_case_types)] pub(super) type AF_Tanh = ActivationFunction<TANH>;

// !!! NOTE !!!: When adding new AF, also add it to [`super::LayerSpecs`] & [`ais_generator::ActivationFunctions::gen_with_rng`].
// TODO: rewrite using ("const") enum to avoid manually managing numbers.
const ABS          : u8 = 0;
const BINARY_STEP  : u8 = 1;
const ELU          : u8 = 2;
const GAUSSIAN     : u8 = 3;
const LEAKY_RELU   : u8 = 4;
// TODO: check if implemented correctly.
const MAX_OUT      : u8 = 5;
const RELU         : u8 = 6;
const SIGMOID      : u8 = 7;
const SIGNUM       : u8 = 8;
const SILU         : u8 = 9;
const SOFT_MAX     : u8 = 10;
const SOFT_PLUS    : u8 = 11;
const SYM_LN       : u8 = 12;
const SYM_SQRT     : u8 = 13;
const TANH         : u8 = 14;

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
        SILU => silu_v,
        SOFT_MAX => soft_max_v,
        SOFT_PLUS => soft_plus_v,
        SYM_LN => sym_ln_v,
        SYM_SQRT => sym_sqrt_v,
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
        SILU => silu_prime_v,
        SOFT_MAX => soft_max_prime_v,
        SOFT_PLUS => soft_plus_prime_v,
        SYM_LN => sym_ln_prime_v,
        SYM_SQRT => sym_sqrt_prime_v,
        TANH => tanh_prime_v,
        _ => unreachable!()
    };

    pub(super) const fn new(size: usize) -> Self {
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

    pub fn get_name(&self) -> String {
        match F_INDEX {
            ABS => stringify!(AF_Abs),
            BINARY_STEP => stringify!(AF_BinaryStep),
            ELU => stringify!(AF_Elu),
            GAUSSIAN => stringify!(AF_Gaussian),
            LEAKY_RELU => stringify!(AF_LeakyRelu),
            RELU => stringify!(AF_Relu),
            SIGMOID => stringify!(AF_Sigmoid),
            SIGNUM => stringify!(AF_Signum),
            SILU => stringify!(AF_Silu),
            SOFT_MAX => stringify!(AF_SoftMax),
            SOFT_PLUS => stringify!(AF_SoftPlus),
            SYM_LN => stringify!(AF_SymLn),
            SYM_SQRT => stringify!(AF_SymSqrt),
            TANH => stringify!(AF_Tanh),
            _ => unreachable!()
        }.to_string()
    }

    #[expect(dead_code)]
    pub fn to_string(&self) -> String {
        let name = self.get_name();
        let size = self.size;
        format!("{name}({size})")
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

