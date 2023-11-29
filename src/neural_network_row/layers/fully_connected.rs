//! Fully Connected Layer

use rand::{Rng, thread_rng};

use crate::{
    float_type::float,
    fully_connected_layer_initial_values::{S_MAX, S_MIN, W_MAX, W_MIN},
    linalg_types::Matrix,
};

use super::super::vector_type::Vector;

use super::Layer;


#[derive(Debug, Clone, PartialEq)]
pub(super) struct FullyConnected {
    weights_matrix: Matrix,
    shift_vector: Vector,
    input: Option<Vector>,
    output: Option<Vector>,
}

impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        let weights_matrix = Matrix::from_fn(
            input_size,
            output_size,
            |_i, _j| rng.gen_range(W_MIN .. W_MAX),
        );
        let shift_vector = Vector::from_fn(
            output_size,
            |_i, _| rng.gen_range(S_MIN .. S_MAX),
        );
        Self {
            weights_matrix,
            shift_vector,
            input: None,
            output: None,
        }
    }
}

impl Layer for FullyConnected {
    fn forward_propagation(&self, input: Vector) -> Vector {
        let output: Vector = input * &self.weights_matrix + &self.shift_vector;
        output
    }

    fn forward_propagation_for_training(&mut self, input: Vector) -> Vector {
        self.input = Some(input.clone());
        // println!();
        // println!("FORWARD PROPAGATION");
        // println!("input: {:?}", input);
        // println!("weights_matrix: {:?}", self.weights_matrix);
        // println!("shift_vector: {:?}", self.shift_vector);
        let output: Vector = input * &self.weights_matrix + &self.shift_vector;
        // println!("output: {:?}", output);
        self.output = Some(output.clone());
        output
    }

    fn backward_propagation(&mut self, output_error: Vector, learning_rate: float) -> Vector {
        // println!();
        // println!("BACKWARD PROPAGATION");
        // println!("output_error: {}", output_error.len());
        // println!("weights_matrix: {:?}", self.weights_matrix.shape());
        let input_error: Vector = &output_error * &self.weights_matrix.transpose();
        let weights_error: Matrix = self.input.as_ref().unwrap().transpose() * &output_error;
        // println!("weights_error: {:?}", weights_error);
        // println!("input_error: {:?}", input_error);

        // update self params
        self.weights_matrix -= learning_rate * weights_error;
        self.shift_vector -= learning_rate * output_error;

        input_error
    }
}

impl ToString for FullyConnected {
    fn to_string(&self) -> String {
        [
            "FCLayer:",
            &format!("weights_matrix: {}", self.weights_matrix.to_string()),
            &format!("shift_vector: {}", self.shift_vector.to_string()),
        ].join("\n")
    }
}

