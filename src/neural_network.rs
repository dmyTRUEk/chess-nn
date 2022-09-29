/// This file contains all NN functions

use std::fmt;

use arrayfire::{Array, Dim4, matmul, div, constant, exp, sign, abs, sqrt};
use rand::{Rng, thread_rng, prelude::ThreadRng};

use crate::{
    activation_functions::*,
    ComputingUnit,
    COMPUTING_UNIT,
    NEURONS_IN_FIRST_LAYER,
};



#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    weight: Vec<Vec<Vec<f32>>>,
    consts: Vec<Vec<f32>>,
    activation_function: ActivationFunction,
}

impl NeuralNetwork {
    pub fn get_activation_function(&self) -> ActivationFunction {
        self.activation_function
    }


    pub fn new(heights: &Vec<usize>) -> Self {
        let layers = heights.len();
        // set layers amount:
        let mut nn: NeuralNetwork = NeuralNetwork {
            weight: vec![vec![]; layers],
            consts: vec![vec![]; layers],
            activation_function: ActivationFunction::Sigmoid,
        };
        // set up weights for every layer:
        for l in 0..layers {
            nn.weight[l] = Vec::with_capacity(heights[l]);
            nn.consts[l] = Vec::with_capacity(heights[l]);
            for _ in 0..heights[l] {
                nn.consts[l].push(0.0);
                nn.weight[l].push(vec![
                    1.0;
                    if l == 0 {
                        // set weights size = 0 for every neuron in first (0) layer
                        // set zero's layer connections amount
                        0
                    } else {
                        // set weights size for neurons in all other layers
                        heights[l-1]
                    }
                ]);
            }
        }
        nn
    }

    pub fn with_consts(
        heights: &Vec<usize>,
        weight: f32,
        consts: f32,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut nn = NeuralNetwork::new(heights);
        nn.activation_function = activation_function;
        for l in 0..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                nn.consts[l][h] = consts;
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = weight;
                }
            }
        }
        nn
    }

    pub fn with_random(
        heights: &Vec<usize>,
        weight_min: f32, weight_max: f32,
        consts_min: f32, consts_max: f32,
    ) -> Self {
        let mut rng = thread_rng();
        let mut nn = NeuralNetwork::new(heights);
        nn.activation_function = get_random_activation_function(&mut rng);
        for l in 0..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                nn.consts[l][h] = rng.gen_range(consts_min..=consts_max);
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = rng.gen_range(weight_min..=weight_max);
                }
            }
        }
        nn
    }

    #[deprecated]
    pub fn with_smart_random(heights: &Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let mut nn = NeuralNetwork::new(heights);
        nn.activation_function = get_random_activation_function(&mut rng);
        for l in 1..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                nn.consts[l][h] = rng.gen_range(-2.0..2.0_f32).powi(2);
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = rng.gen_range(
                        -1.0/(nn.weight[l-1].len() as f32)..1.0/(nn.weight[l-1].len() as f32)
                    );
                }
            }
        }
        nn
    }


    pub fn process_input(&self, input: &Vec<f32>) -> Vec<f32> {
        // assert_eq!(if !USE_65_NEURONS {64} else {65}, input.len());
        match COMPUTING_UNIT {
            ComputingUnit::CPU => {
                self.process_input_cpu(input)
            }
            ComputingUnit::GPU => {
                self.process_input_gpu(input)
            }
        }
    }


    fn process_input_cpu(&self, input: &Vec<f32>) -> Vec<f32> {
        let layers = self.weight.len();
        let mut input: Vec<f32> = input.to_vec();
        for l in 1..layers {
            let mut res: Vec<f32> = Vec::with_capacity(self.weight[l].len());
            for h in 0..self.weight[l].len() {
                let mut sum: f32 = 0.0;
                // assert_eq!(self.weight[l-1].len(), self.weight[l][h].len());
                // for c in 0..self.weight[l][h].len() {
                for c in 0..self.weight[l-1].len() {
                    sum += self.weight[l][h][c] * input[c];
                }
                sum += self.consts[l][h];
                if l != layers - 1 {
                    sum = calc_activation_function(sum, self.activation_function);
                }
                // sum = 1.0 / (1.0 + (-sum).exp());
                // sum = if sum >= 0.0 { 1.0 } else { 0.0 };
                // sum = if sum >= 0.0 { sum } else { 0.0 };
                res.push(sum);
            }
            // assert_eq!(input.len(), self.weight[l-1].len());
            input = res;
        }
        return input;
    }

 
    fn process_input_gpu(&self, input: &Vec<f32>) -> Vec<f32> {
        let layers = self.weight.len();
        // this data is on gpu
        let mut input: Array<f32> = Array::new(input, Dim4::new(&[NEURONS_IN_FIRST_LAYER as u64, 1, 1, 1]));
        for l in 1..layers {
            let weights_dims: Dim4 = Dim4::new(&[self.weight[l].len() as u64, self.weight[l-1].len() as u64, 1, 1]);
            let weights_flat: Vec<f32> = self.weight[l]
                .iter()
                .flatten()
                .map(|&x| x)
                .collect();
            let weights_flat_slice: &[f32] = weights_flat.as_slice();
            // this data is on gpu
            let weights: Array<f32>/* 2D */ = Array::new(weights_flat_slice, weights_dims);
            // assert_eq!(
            //     input.dims().get()[0],
            //     weights.dims().get()[1]
            // );
            // println!("{:?}", input.dims().get());
            // println!("{:?}", weights.dims().get());
            // prepare consts on gpu:
            let consts_dims: Dim4 = Dim4::new(&[self.weight[l].len() as u64, 1, 1, 1]);
            let consts_slice: &[f32] = self.consts[l].as_slice();
            let consts: Array<f32> = Array::new(consts_slice, consts_dims);
            // calc result
            input = matmul(&weights, &input, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);
            input += consts;
            // todo!("use consts and correct activation func");
            //input += consts;
            // apply activation function: x -> 1 / (1 + e^-x)
            // input = constant(0.0_f32, input.dims()) - input;
            match self.activation_function {
                ActivationFunction::Sigmoid => {
                    input = -input;
                    input /= constant(10.0_f32, input.dims());
                    input = exp(&input);
                    input += constant(1.0_f32, input.dims());
                    input = div(&1.0_f32, &input, false);
                    //input = div(&1.0_f32, &(constant(1.0_f32, input.dims()) + exp(&(-input/constant(10.0_f32, &input.dims())))), false);
                }
                ActivationFunction::Arctan => {
                    todo!()
                }
                ActivationFunction::SignSqrtAbs => {
                    input = sign(&input) * sqrt(&abs(&input));
                }
                // ActivationFunction::SignLnAbs => {
                //     input = sign(&input) * log(&abs(&input));
                // }
                // ActivationFunction::Linear => {
                //     // input = input;
                // }
            }
            // assert_eq!(
            //     self.weight[l].len(),
            //     input.elements()
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


    fn get_total_neurons(&self) -> usize {
        let layers = self.weight.len();
        let mut res: usize = 0;
        // from 1, because 0's layer doesnt calculate anything
        for l in 1..layers {
            res += self.weight[l].len();
        }
        res
    }

    fn choose_random_neuron(&self, rng: &mut ThreadRng) -> (usize, usize) {
        let neuron_id_to_evolve: usize = rng.gen_range(0..self.get_total_neurons());
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

        let total_neurons: u32 = self.get_total_neurons() as u32;
        let neurons_to_evolve: u32 = ((total_neurons as f32) * evolution_factor) as u32;
        let neurons_to_evolve: u32 = rng.gen_range(1..=neurons_to_evolve.max(1));
        // println!("total_neurons = {}", total_neurons);
        // println!("neurons_to_evolve = {}", neurons_to_evolve);

        if rng.gen_bool(0.1) {
            self.activation_function = get_random_activation_function(&mut rng);
        }

        for _ in 0..neurons_to_evolve {
            let (l, h) = self.choose_random_neuron(&mut rng);

            let total_weights: u32 = self.weight[l][h].len() as u32;
            let weights_to_evolve: u32 = ((total_weights as f32) * evolution_factor) as u32;
            let weights_to_evolve: u32 = rng.gen_range(1..=weights_to_evolve.max(1));
            // println!("total_weights = {}", total_weights);
            // println!("weights_to_evolve = {}", weights_to_evolve);

            let sign: f32 = if rng.gen_bool((evolution_factor/2.0) as f64) { -1.0 } else { 1.0 };
            self.consts[l][h] *= sign;

            if rng.gen_bool(0.5) {
                self.consts[l][h] *= (1.0 + evolution_factor).powi(3);
            } else {
                self.consts[l][h] /= (1.0 + evolution_factor).powi(3);
            }
            if rng.gen_bool(0.5) {
                self.consts[l][h] += evolution_factor / 100.0;
            } else {
                self.consts[l][h] -= evolution_factor / 100.0;
            }

            for _ in 0..weights_to_evolve {
                // println!("old value: {}", self.weights[h]);
                let c: usize = rng.gen_range(0..total_weights) as usize;

                let sign: f32 = if rng.gen_bool((evolution_factor/2.0) as f64) { -1.0 } else { 1.0 };
                self.weight[l][h][c] *= sign;

                if rng.gen_bool(0.5) {
                    self.weight[l][h][c] *= (1.0 + evolution_factor).powi(3);
                } else {
                    self.weight[l][h][c] /= (1.0 + evolution_factor).powi(3);
                }
                if rng.gen_bool(0.5) {
                    self.weight[l][h][c] += evolution_factor / 100.0;
                } else {
                    self.weight[l][h][c] -= evolution_factor / 100.0;
                }
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

