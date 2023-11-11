//! Fully Connected Layer

use rand::{thread_rng, Rng};

use crate::{
    float_type::float,
    linalg_types::{Matrix, RowVector},
};

use super::Layer;


#[derive(Debug, Clone, PartialEq)]
pub struct FCLayer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights_matrix: Matrix,
    shift_vector: RowVector,
    input: Option<RowVector>,
    output: Option<RowVector>,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer for FCLayer<INPUT_SIZE, OUTPUT_SIZE> {
    fn new() -> Self {
        const W_MIN: float = -0.003;
        const W_MAX: float =  0.003;
        const S_MIN: float = -0.003;
        const S_MAX: float =  0.003;
        let mut rng = thread_rng();
        let weights_matrix = Matrix::from_fn(
            INPUT_SIZE,
            OUTPUT_SIZE,
            |_i, _j| rng.gen_range(W_MIN .. W_MAX),
        );
        let shift_vector = RowVector::from_fn(
            OUTPUT_SIZE,
            |_i, _| rng.gen_range(S_MIN .. S_MAX),
        );
        Self {
            weights_matrix,
            shift_vector,
            input: None,
            output: None,
        }
    }

    fn forward_propagation(&self, input: RowVector) -> RowVector {
        let output: RowVector = input * &self.weights_matrix + &self.shift_vector;
        output
    }

    fn forward_propagation_for_training(&mut self, input: RowVector) -> RowVector {
        self.input = Some(input.clone());
        // println!();
        // println!("FORWARD PROPAGATION");
        // println!("input: {:?}", input);
        // println!("weights_matrix: {:?}", self.weights_matrix);
        // println!("shift_vector: {:?}", self.shift_vector);
        let output: RowVector = input * &self.weights_matrix + &self.shift_vector;
        // println!("output: {:?}", output);
        self.output = Some(output.clone());
        output
    }

    fn backward_propagation(&mut self, output_error: RowVector, learning_rate: float) -> RowVector {
        // println!();
        // println!("BACKWARD PROPAGATION");
        // println!("output_error: {}", output_error.len());
        // println!("weights_matrix: {:?}", self.weights_matrix.shape());
        let input_error: RowVector = &output_error * &self.weights_matrix.transpose();
        let weights_error: Matrix = self.input.as_ref().unwrap().transpose() * &output_error;
        // println!("weights_error: {:?}", weights_error);
        // println!("input_error: {:?}", input_error);

        // update self params
        self.weights_matrix -= learning_rate * weights_error;
        self.shift_vector -= learning_rate * output_error;

        input_error
    }
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> ToString for FCLayer<INPUT_SIZE, OUTPUT_SIZE> {
    fn to_string(&self) -> String {
        [
            "FCLayer:",
            &format!("weights_matrix: {}", self.weights_matrix.to_string()),
            &format!("shift_vector: {}", self.shift_vector.to_string()),
        ].join("\n")
    }
}

