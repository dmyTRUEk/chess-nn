/// This file contains all NN functions

use std::fmt;

use arrayfire::{Array, Dim4, matmul, div, constant, exp};
use rand::{Rng, thread_rng, prelude::ThreadRng};

use crate::{ComputingUnit, COMPUTING_UNIT};
// use crate::activation_functions::activation_function;



#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    pub weight: Vec<Vec<Vec<f32>>>,
}

impl NeuralNetwork {
    pub fn new(heights: &Vec<usize>) -> Self {
        let layers_amount = heights.len();

        // set layers amount
        let mut nn: NeuralNetwork = NeuralNetwork {
            weight: vec![
                vec![];
                layers_amount
            ]
        };

        // set up weights for every layer
        for l in 0..layers_amount {
            nn.weight[l] = Vec::with_capacity(heights[l]);
            for _ in 0..heights[l] {
                nn.weight[l].push(
                    vec![
                        1.0;
                        if l == 0 {
                            // set weights size = 0 for every neuron in first (0) layer
                            // set zero's layer connections amount
                            0
                        } else {
                            // set weights size for neurons in all other layers
                            heights[l-1]
                        }
                    ]
                );
            }
        }

        nn
    }

    pub fn with_const_weights(heights: &Vec<usize>, value: f32) -> Self {
        let mut nn = NeuralNetwork::new(heights);
        for l in 0..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = value;
                }
            }
        }
        nn
    }

    pub fn with_random_weights(heights: &Vec<usize>, value_min: f32, value_max: f32) -> Self {
        let mut rng = thread_rng();
        let mut nn = NeuralNetwork::new(heights);
        for l in 0..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = rng.gen_range(value_min..value_max);
                }
            }
        }
        nn
    }


    pub fn process_input(&self, input: &Vec<f32>) -> Vec<f32> {
        assert_eq!(64, input.len());
        let layers = self.weight.len();
        
        match COMPUTING_UNIT {
            ComputingUnit::CPU => {
                let mut input: Vec<f32> = input.to_vec();

                for l in 1..layers {
                    let mut res: Vec<f32> = Vec::with_capacity(self.weight[l].len());
                    for h in 0..self.weight[l].len() {
                        let mut sum: f32 = 0.0;
                        // assert_eq!(self.weight[l-1].len(), self.weight[l][h].len());
                        for c in 0..self.weight[l][h].len() {
                            sum += self.weight[l][h][c] * input[c]
                        }
                        sum = 1.0 / (1.0 + (-sum).exp());
                        res.push(sum);
                    }
                    // assert_eq!(input.len(), self.weight[l-1].len());

                    input = res;
                }

                return input;
            }
            ComputingUnit::GPU => {
                // this data is on gpu
                let mut input: Array<f32> = Array::new(input, Dim4::new(&[64, 1, 1, 1]));

                for l in 1..layers {
                    let weights_dims: Dim4 = Dim4::new(&[self.weight[l].len() as u64, self.weight[l-1].len() as u64, 1, 1]);
                    // let dims_input: Dim4 = Dim4::new(&[self.weight[l-1].len() as u64, 1, 1, 1]);

                    // let mut weights_flat: Vec<f32> = Vec::with_capacity(self.weight[l].len() * self.weight[l-1].len());
                    // for h in 0..self.weight[l].len() {
                    //     for c in 0..self.weight[l][h].len() {
                    //         weights_flat.push(self.weight[l][h][c]);
                    //     }
                    // }
                    let weights_flat: Vec<f32> = self.weight[l]
                        // .clone()
                        // .into_iter()
                        .iter()
                        .flatten()
                        .map(|&x| x)
                        .collect();
                    let weights_flat_slice: &[f32] = weights_flat.as_slice();

                    // this data is on gpu
                    let weights = Array::new(weights_flat_slice, weights_dims);

                    // assert_eq!(
                    //     input_on_gpu.dims().get()[0],
                    //     weights_on_gpu.dims().get()[1]
                    // );

                    // println!("{:?}", input_on_gpu.dims().get());
                    // println!("{:?}", weights_on_gpu.dims().get());

                    // calc result
                    input = matmul(&weights, &input, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);

                    // apply activation function: x -> 1 / (1 + e^-x)
                    // input = constant(0.0_f32, input.dims()) - input;
                    input = -input;
                    input = exp(&input);
                    input += constant(1.0, input.dims());
                    input = div(&1.0_f32, &input, false);

                    // assert_eq!(
                    //     self.weight[l].len(),
                    //     input_on_gpu.elements()
                    // );
                }

                // assert_eq!(
                //     1,
                //     input.elements()
                // );

                let mut result: Vec<f32> = vec![0.0; self.weight.last().unwrap().len()];
                input.host(&mut result);

                return result;
            }
        }
    }

    pub fn get_total_neurons(&self) -> u32 {
        let layers = self.weight.len();
        let mut res: u32 = 0;
        // from 1, because 0's layer doesnt calculate anything
        for l in 1..layers {
            res += self.weight[l].len() as u32;
        }
        res
    }

    fn choose_random_neuron(&self, rng: &mut ThreadRng) -> (usize, usize) {
        let total_neurons: u32 = self.get_total_neurons();
        let neuron_id_to_evolve: usize = rng.gen_range(0..total_neurons) as usize;
        let mut l: usize = 1;
        let mut h: usize = 0;
        for _j in 0..neuron_id_to_evolve {
            if h < self.weight[l].len()-1 {
                h += 1;
            }
            else {
                l += 1;
                h = 0;
            }
        }
        (l, h)
    }

    pub fn evolve(&mut self, evolution_factor: f32) {
        assert!(0.0 <= evolution_factor && evolution_factor <= 1.0);

        let mut rng = thread_rng();

        let total_neurons: u32 = self.get_total_neurons();
        let neurons_to_evolve: u32 = ((total_neurons as f32) * evolution_factor) as u32;
        let neurons_to_evolve: u32 = rng.gen_range(1..=neurons_to_evolve.max(1));
        // println!("total_neurons = {}", total_neurons);
        // println!("neurons_to_evolve = {}", neurons_to_evolve);

        for _i in 0..neurons_to_evolve {
            let (l, h) = self.choose_random_neuron(&mut rng);

            let total_weights: u32 = self.weight[l][h].len() as u32;
            let weights_to_evolve: u32 = ((total_weights as f32) * evolution_factor) as u32;
            let weights_to_evolve: u32 = rng.gen_range(1..=weights_to_evolve.max(1));
            // println!("total_weights = {}", total_weights);
            // println!("weights_to_evolve = {}", weights_to_evolve);

            for _ in 0..weights_to_evolve {
                let c: usize = rng.gen_range(0..total_weights) as usize;

                let sign: f32 = if rng.gen_range(0.0_f32..=1.0_f32) < 0.01_f32 { -1.0 } else { 1.0 };

                // println!("old value: {}", self.weights[h]);
                self.weight[l][h][c] *= sign *
                    if rng.gen_range(-1.0_f32..=1.0_f32) > 0.0 {
                        1.0 * (1.0 + evolution_factor)
                        // 1.0 * (1.1)
                    } else {
                        1.0 / (1.0 + evolution_factor)
                        // 1.0 / (1.1)
                    };
                // println!("new value: {}", self.weights[h]);
            }
        }
    }
}



impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = String::from("{");

        let layers = self.weight.len();
        let mut height: usize;
        for l in 0..layers {
            height = self.weight[l].len();
            for h in 0..height {
                res += &format!("\n    neuron[{}][{}] = {:?},", l, h, self.weight[l][h]);
            }
            res += "\n";
        }
        res += "}\n";

        write!(f, "{}", res)
    }
}

