//! Activation Function Tanh

use crate::{
    float_type::float,
    linalg_types::RowVector,
    math_functions::{sigmoid_prime_v, sigmoid_v, tanh_prime_v, tanh_v},
};

use super::Layer;


pub type AFSigmoid<const SIZE: usize> = ActivationFunction<SIZE, SIGMOID>;
pub type AFTanh<const SIZE: usize> = ActivationFunction<SIZE, TANH>;

const SIGMOID: u8 = 0;
const TANH: u8 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct ActivationFunction<const SIZE: usize, const F_INDEX: u8> {
    input: Option<RowVector>,
    output: Option<RowVector>,
}

impl<const SIZE: usize, const F_INDEX: u8> ActivationFunction<SIZE, F_INDEX> {
    const F: fn(RowVector) -> RowVector = match F_INDEX {
        SIGMOID => sigmoid_v,
        TANH => tanh_v,
        _ => unreachable!()
    };

    const F_PRIME: fn(RowVector) -> RowVector = match F_INDEX {
        SIGMOID => sigmoid_prime_v,
        TANH => tanh_prime_v,
        _ => unreachable!()
    };

    fn forward_propagation(&self, input: RowVector) -> RowVector {
        assert_eq!(SIZE, input.len());
        // println!("input: {input:?}");
        let output = Self::F(input);
        // println!("output: {output:?}");
        output
    }

    fn forward_propagation_for_training(&mut self, input: RowVector) -> RowVector {
        assert_eq!(SIZE, input.len());
        self.input = Some(input.clone());
        let output = self.forward_propagation(input);
        self.output = Some(output.clone());
        output
    }

    fn backward_propagation(&mut self, output_error: RowVector, _learning_rate: float) -> RowVector {
        assert_eq!(SIZE, output_error.len());
        Self::F_PRIME(self.input.clone().unwrap()).component_mul(&output_error)
    }
}

impl<const SIZE: usize, const F_INDEX: u8> Layer for ActivationFunction<SIZE, F_INDEX> {
    fn new() -> Self {
        Self {
            input: None,
            output: None,
        }
    }

    fn forward_propagation(&self, input: RowVector) -> RowVector {
        self.forward_propagation(input)
    }

    fn forward_propagation_for_training(&mut self, input: RowVector) -> RowVector {
        self.forward_propagation_for_training(input)
    }

    fn backward_propagation(&mut self, output_error: RowVector, learning_rate: float) -> RowVector {
        self.backward_propagation(output_error, learning_rate)
    }
}

impl<const SIZE: usize, const F_INDEX: u8> ToString for ActivationFunction<SIZE, F_INDEX> {
    fn to_string(&self) -> String {
        match F_INDEX {
            SIGMOID => format!("AFSigmoid({SIZE})"),
            TANH => format!("AFTanh({SIZE})"),
            _ => unreachable!()
        }
    }
}

