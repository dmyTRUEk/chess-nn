//! AIs generator.

use rand::{Rng, rngs::ThreadRng, thread_rng};

use crate::{
	NN_INPUT_SIZE,
	NN_OUTPUT_SIZE,
	ais_generator_consts,
	float_type::float,
	neural_network_row::{ChessNeuralNetwork, layers::LayerSpecs},
};

use super::ai::AI;



pub struct AIsGenerator {
	/// Probability of using different Activation Functions in NN.
	pub multi_af_prob: float,
	pub activation_functions: ActivationFunctions,
	pub layers_number: LayersNumber,
	pub layers_sizes: &'static [usize],
}

impl AIsGenerator {
	pub fn generate(&self, n: usize) -> Vec<AI> {
		if let ActivationFunctions::Specific(activation_functions) = &self.activation_functions {
			assert!(activation_functions.iter().all(|maybe_af| maybe_af.is_activation_function()))
		}
		match self.layers_number {
			LayersNumber::All => {}
			LayersNumber::Range { min, max } => {
				assert!(min <= max);
				assert!(max <= self.layers_sizes.len());
			}
			LayersNumber::Specific(ref layers_numbers) => {
				assert!(layers_numbers.iter().all(|&layers_number| layers_number <= self.layers_sizes.len()));
			}
		}
		(0..n)
			.into_iter()
			// .into_par_iter() // this function shouldn't be long, so no need
			.map(|_| self.generate_one_ai())
			.collect()
	}

	fn generate_one_ai(&self) -> AI {
		const WRITE_FIRST_AND_LAST_LAYERS_IN_NAME: bool = false;
		let mut rng = thread_rng();
		let layers_number = match &self.layers_number {
			LayersNumber::All => rng.gen_range(0..=self.layers_sizes.len()),
			LayersNumber::Specific(layers_numbers) => layers_numbers[rng.gen_range(0..layers_numbers.len())],
			LayersNumber::Range { min, max } => rng.gen_range(*min..=*max),
		};
		let mut layers_sizes = vec![];
		while layers_sizes.len() < layers_number {
			layers_sizes.push(self.layers_sizes[rng.gen_range(0..self.layers_sizes.len())]);
		}
		layers_sizes.sort();
		layers_sizes.reverse();
		let layers_sizes = layers_sizes; // unmut
		let gen_af = || { self.activation_functions.gen() };
		let activation_function: Option<LayerSpecs> = if rng.gen_bool(self.multi_af_prob) { None } else { Some(gen_af()) };
		let af_to_string = |afs: LayerSpecs| { format!("{afs:?}").trim_start_matches("AF_").to_string() };
		let mut layers = Vec::<LayerSpecs>::with_capacity(2*layers_number+1);
		// TODO(refactor): rename this var.
		let name: Option<String> = activation_function.map(af_to_string);
		let mut name_parts: Vec<String> = vec![];
		if WRITE_FIRST_AND_LAST_LAYERS_IN_NAME {
			name_parts.push(NN_INPUT_SIZE.to_string());
		}
		for layer_size in layers_sizes {
			layers.push(LayerSpecs::FullyConnected(layer_size));
			name_parts.push(layer_size.to_string());
			let af = activation_function.unwrap_or_else(gen_af);
			layers.push(af);
			if activation_function.is_none() {
				name_parts.push(af_to_string(af));
			}
		}
		layers.push(LayerSpecs::FullyConnected(NN_OUTPUT_SIZE));
		if WRITE_FIRST_AND_LAST_LAYERS_IN_NAME {
			name_parts.push(NN_OUTPUT_SIZE.to_string());
		}
		const NAME_PARTS_SEP: &str = "-";
		let name_parts_joined = name_parts.join(NAME_PARTS_SEP);
		let final_name = if let Some(name) = name {
			format!("{name} {name_parts_joined}")
		} else {
			name_parts_joined
		};
		AI::new_for_training(
			final_name,
			ChessNeuralNetwork::from_layers_specs(layers),
		)
	}
}

impl Default for AIsGenerator {
	fn default() -> Self {
		AIsGenerator {
			multi_af_prob: ais_generator_consts::MULTI_AF_PROB,
			activation_functions: ais_generator_consts::ACTIVATION_FUNCTIONS,
			layers_number: ais_generator_consts::LAYERS_NUMBER,
			layers_sizes: ais_generator_consts::LAYERS_SIZES,
		}
	}
}

#[expect(dead_code)]
pub enum LayersNumber {
	All,
	/// Inclusive range
	Range {
		min: usize,
		max: usize,
	},
	Specific(Vec<usize>),
}

#[expect(dead_code)]
pub enum ActivationFunctions {
	All,
	Specific(Vec<LayerSpecs>),
}
impl ActivationFunctions {
	fn gen(&self) -> LayerSpecs {
		self.gen_with_rng(&mut thread_rng())
	}
	fn gen_with_rng(&self, rng: &mut ThreadRng) -> LayerSpecs {
		match self {
			Self::All => {
				use LayerSpecs::*;
				const AFS: &[LayerSpecs] = &[
					AF_Abs,
					AF_BinaryStep,
					AF_Elu,
					AF_Gaussian,
					AF_LeakyRelu,
					// AF_MaxOut,
					AF_Relu,
					AF_Sigmoid,
					AF_Signum,
					AF_Silu,
					// AF_SoftMax,
					AF_SoftPlus,
					AF_SymLn,
					AF_SymSqrt,
					AF_Tanh,
				];
				AFS[rng.gen_range(0..AFS.len())]
			}
			Self::Specific(activation_functions) => activation_functions[rng.gen_range(0..activation_functions.len())],
		}
	}
}

