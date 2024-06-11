/// All NN functions

pub mod layers;
pub mod vector_type;


use std::fmt;

use crate::{NN_INPUT_SIZE, float_type::float};

use self::{
	layers::{BoxDynLayer, LayerSpecs},
	vector_type::Vector,
};



#[derive(Clone)]
pub struct ChessNeuralNetwork {
	pub layers: Vec<BoxDynLayer>,
}

impl ChessNeuralNetwork {
	/// Create Neural Network from layers specs.
	pub fn from_layers_specs(layers_specs: Vec<LayerSpecs>) -> Self {
		let mut layers = Vec::<BoxDynLayer>::with_capacity(layers_specs.len());
		let mut input_size: usize = NN_INPUT_SIZE;
		for layer_specs in layers_specs {
			// TODO(refactor): dont pass `&mut`, but get updated and set it here.
			layers.push(layer_specs.to_layer(&mut input_size));
		}
		Self { layers }
	}

	/// Returns predicion.
	pub fn process_input(&self, input: Vector) -> float {
		let mut input = input;
		let mut output = None;
		for layer in self.layers.iter() {
			let o = layer.forward_propagation(input);
			// TODO(optimize)
			output = Some(o.clone());
			input = o;
		}
		let output = output.unwrap();
		// TODO?: assert_eq 1 and NN_OUTPUT_SIZE
		assert_eq!(1, output.len(), "NN output must be one, but it was not");
		output[0]
	}

	/// Returns predicion and saves input & output in every layer.
	pub fn process_input_for_training(&mut self, input: Vector) -> float {
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

	// /// Returns vec of predicions.
	// pub fn process_v(&self, inputs: Vec<Vector>) -> Vec<float> {
	//     inputs
	//         .into_iter()
	//         .map(|input| self.process_input(input))
	//         .collect()
	// }

	// THIS IS WRONG
	// pub fn process_multiple_input_for_training(&mut self, inputs: Vec<Vector>) -> Vec<float> {
	//     inputs
	//         .into_iter()
	//         .map(|input| self.process_input_for_training(input))
	//         .collect()
	// }

	/// Loss function = Mean Square Error.
	pub fn loss(&self, output_actual: float, output_expected: float) -> float {
		let error = output_expected - output_actual;
		let error_squared = error.powi(2);
		error_squared
	}

	/// Derivative of loss function.
	pub fn loss_prime(&self, output_actual: float, output_expected: float) -> float {
		2. * (output_actual - output_expected)
	}

	// pub fn loss_v(&self, outputs_actual: Vec<float>, outputs_expected: Vec<float>) -> Vec<float> {
	//     // TODO(optimize): rewrite using `nalgebra`
	//     assert_eq!(outputs_actual.len(), outputs_expected.len());
	//     outputs_actual.into_iter().zip(outputs_expected)
	//         .map(|(output_actual, outputs_expected)| self.loss(output_actual, outputs_expected))
	//         .collect()
	// }

	// /// Returns vec of losses.
	// pub fn loss_from_input_v(&self, inputs: Vec<Vector>, outputs_expected: Vec<float>) -> Vec<float> {
	//     // TODO(optimize): rewrite using `nalgebra`
	//     assert_eq!(inputs.len(), outputs_expected.len());
	//     let outputs = self.process_v(inputs);
	//     self.loss_v(outputs, outputs_expected)
	// }

	// /// Returns sum of vec of losses = sum . loss_v
	// /// where `.` denotes composition.
	// pub fn loss_from_input_v_sum(&self, inputs: Vec<Vector>, outputs_expected: Vec<float>) -> float {
	//     assert_eq!(inputs.len(), outputs_expected.len());
	//     self.loss_from_input_v(inputs, outputs_expected).into_iter().sum()
	// }

	// /// Returns avg of vec of losses = avg . loss_v
	// /// where `.` denotes composition.
	// pub fn loss_from_input_v_avg(&self, inputs: Vec<Vector>, outputs_expected: Vec<float>) -> float {
	//     assert_eq!(inputs.len(), outputs_expected.len());
	//     let inputs_len_f = inputs.len() as float;
	//     self.loss_from_input_v_sum(inputs, outputs_expected) / inputs_len_f
	// }

	// /// Returns sqrt of avg of vec of losses = sqrt . avg . loss_v
	// /// where `.` denotes composition.
	// pub fn loss_from_input_v_avg_sqrt(&self, inputs: Vec<Vector>, outputs_expected: Vec<float>) -> float {
	//     assert_eq!(inputs.len(), outputs_expected.len());
	//     self.loss_from_input_v_avg(inputs, outputs_expected).sqrt()
	// }
}



impl fmt::Display for ChessNeuralNetwork {
	#[expect(unreachable_code, unused_variables)]
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut res = "ChessNeuralNetwork {".to_string();
		for (i, layer) in self.layers.iter().enumerate() {
			unimplemented!();
			// res += &format!("\nlayer[{i}]: {},", layer.to_string());
			res += "\n";
		}
		res += "}\n";
		write!(f, "{res}")
	}
}

