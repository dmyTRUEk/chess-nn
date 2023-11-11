/// All NN functions

pub mod layers;

use std::fmt;

use crate::{
    float_type::float,
    linalg_types::RowVector,
};

use self::layers::Layer;



#[derive(Clone)]
pub struct ChessNeuralNetwork {
    pub layers: Vec<Box<dyn Layer + Send + Sync>>,
}

impl ChessNeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer + Send + Sync>>) -> Self {
        Self { layers }
    }

    pub fn process_input(&self, input: RowVector) -> float {
        let mut input = input;
        let mut output = None;
        for layer in self.layers.iter() {
            let o = layer.forward_propagation(input);
            // TODO(optimize)
            output = Some(o.clone());
            input = o;
        }
        let output = output.unwrap();
        assert_eq!(1, output.len(), "NN output must be one, but it was not");
        output[0]
    }

    pub fn process_input_for_training(&mut self, input: RowVector) -> float {
        let mut input = input;
        let mut output = None;
        for layer in self.layers.iter_mut() {
            // println!("input: {input:?}");
            let o = layer.forward_propagation_for_training(input);
            // TODO(optimize)
            output = Some(o.clone());
            input = o;
        }
        // println!("output: {:?}", output.clone().unwrap());
        // panic!();
        output.unwrap()[0]
    }

    pub fn process_multiple_input(&self, inputs: Vec<RowVector>) -> Vec<float> {
        inputs
            .into_iter()
            .map(|input| self.process_input(input))
            .collect()
    }

    pub fn process_multiple_input_for_training(&mut self, inputs: Vec<RowVector>) -> Vec<float> {
        inputs
            .into_iter()
            .map(|input| self.process_input_for_training(input))
            .collect()
    }

    pub fn loss(&self, output_actual: float, output_expected: float) -> float {
        let error = output_expected - output_actual;
        let error_squared = error.powi(2);
        error_squared
    }

    pub fn loss_prime(&self, output_actual: float, output_expected: float) -> float {
        2. * (output_actual - output_expected)
    }

    pub fn get_total_neurons(&self) -> usize {
        todo!()
    }
}



impl fmt::Display for ChessNeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = "ChessNeuralNetwork {".to_string();
        for (i, layer) in self.layers.iter().enumerate() {
            res += &format!("\nlayer[{i}]: {},", layer.to_string());
            res += "\n";
        }
        res += "}\n";
        write!(f, "{res}")
    }
}

