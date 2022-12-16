/// This file contains all NN functions

use std::fmt;

use rand::{Rng, prelude::ThreadRng};
// use arrayfire::{Array, Dim4, matmul, div, constant, exp, sign, abs, sqrt};

use crate::{
    activation_functions::*,
    ComputingUnit,
    COMPUTING_UNIT,
    // NEURONS_IN_FIRST_LAYER,
    // simple_rng::SimpleRng,
};



#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    // TODO optimization?: rewrite to flat `Vec<f32>` -> `[f32; N]`
    input_len: usize,
    weight: Vec<Vec<Vec<f32>>>,
    consts: Vec<Vec<f32>>,
    activation_function: ActivationFunction,
}

impl NeuralNetwork {
    pub fn get_activation_function(&self) -> ActivationFunction {
        self.activation_function
    }

    pub fn new(input_len: usize, heights: &Vec<usize>) -> Self {
        let heights: Vec<usize> = {
            let mut heights: Vec<usize> = heights.to_vec();
            heights.push(1); // assuming that output size is 1
            heights
        };
        let layers = heights.len();
        // set layers amount:
        let mut nn: NeuralNetwork = NeuralNetwork {
            input_len,
            weight: vec![vec![]; layers],
            consts: vec![vec![]; layers],
            activation_function: ActivationFunction::Sigmoid,
        };
        // set up weights for every layer:
        for l in 0..layers {
            let height: usize = heights[l];
            nn.consts[l] = vec![0.0; height];
            nn.weight[l] = vec![
                vec![
                    1.0;
                    match l {
                        // set zero's layer connections amount
                        0 => { input_len }
                        // layers-1 => {  }
                        // set weights size for neurons in all other layers
                        _ => { heights[l-1] }
                    }
                ];
                height
            ];
        }
        nn
    }

    pub fn with_consts(
        input_len: usize,
        heights: &Vec<usize>,
        weight: f32,
        consts: f32,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut nn = NeuralNetwork::new(input_len, heights);
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
        input_len: usize,
        heights: &Vec<usize>,
        weight_min: f32, weight_max: f32,
        consts_min: f32, consts_max: f32,
        rng: &mut ThreadRng,
    ) -> Self {
        let mut nn = NeuralNetwork::new(input_len, heights);
        nn.activation_function = get_random_activation_function(rng);
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
    pub fn with_smart_random(
        input_len: usize,
        heights: &Vec<usize>,
        rng: &mut ThreadRng
    ) -> Self {
        let mut nn = NeuralNetwork::new(input_len, heights);
        nn.activation_function = get_random_activation_function(rng);
        for l in 1..nn.weight.len() {
            for h in 0..nn.weight[l].len() {
                nn.consts[l][h] = rng.gen_range(-2.0..=2.0_f32).powi(2);
                let amplitude: f32 = 1.0 / (nn.weight[l-1].len() as f32);
                for c in 0..nn.weight[l][h].len() {
                    nn.weight[l][h][c] = rng.gen_range(-amplitude..=amplitude);
                }
            }
        }
        nn
    }


    pub fn process_input<const N: usize>(&self, input: &[f32; N]) -> f32 {
        // debug_assert_eq!(NEURONS_IN_FIRST_LAYER, input.len());
        match COMPUTING_UNIT {
            ComputingUnit::CPU => {
                self.process_input_cpu(input)
            }
            ComputingUnit::GPU => {
                // self.process_input_gpu(input)
                todo!("plz enable `GPU`")
            }
        }
    }


    fn process_input_cpu<const N: usize>(&self, input: &[f32; N]) -> f32 {
        let mut input_for_next_layer: Vec<f32> = input.to_vec();
        let layers = self.weight.len();
        for l in 0..layers {
            let height: usize = self.weight[l].len();
            let mut res: Vec<f32> = Vec::with_capacity(height);
            for h in 0..height {
                let mut sum: f32 = 0.0;
                debug_assert_eq!(if l > 0 { self.weight[l-1].len() } else { input.len() }, self.weight[l][h].len());
                // let connections: usize = if l > 0 { self.weight[l-1].len() } else { input.len() };
                let connections: usize = input_for_next_layer.len();
                // for c in 0..self.weight[l][h].len() {
                for c in 0..connections {
                    sum += self.weight[l][h][c] * input_for_next_layer[c];
                }
                // or maybe try `+= const` after activation_function?
                sum += self.consts[l][h];
                sum = calc_activation_function(sum, self.activation_function);
                res.push(sum);
            }
            input_for_next_layer = res;
            debug_assert_eq!(input_for_next_layer.len(), self.weight[l].len());
        }
        // for optimization purposes result is `f32` instead of `Vec<f32>`
        return input_for_next_layer[0];
    }


    //fn process_input_gpu(&self, input: &Vec<f32>) -> Vec<f32> {
    //    let layers = self.weight.len();
    //    // this data is on gpu
    //    let mut input: Array<f32> = Array::new(input, Dim4::new(&[NEURONS_IN_FIRST_LAYER as u64, 1, 1, 1]));
    //    for l in 1..layers {
    //        let weights_dims: Dim4 = Dim4::new(&[self.weight[l].len() as u64, self.weight[l-1].len() as u64, 1, 1]);
    //        let weights_flat: Vec<f32> = self.weight[l]
    //            .iter()
    //            .flatten()
    //            .map(|&x| x)
    //            .collect();
    //        let weights_flat_slice: &[f32] = weights_flat.as_slice();
    //        // this data is on gpu
    //        let weights: Array<f32>/* 2D */ = Array::new(weights_flat_slice, weights_dims);
    //        // assert_eq!(
    //        //     input.dims().get()[0],
    //        //     weights.dims().get()[1]
    //        // );
    //        // println!("{:?}", input.dims().get());
    //        // println!("{:?}", weights.dims().get());
    //        // prepare consts on gpu:
    //        let consts_dims: Dim4 = Dim4::new(&[self.weight[l].len() as u64, 1, 1, 1]);
    //        let consts_slice: &[f32] = self.consts[l].as_slice();
    //        let consts: Array<f32> = Array::new(consts_slice, consts_dims);
    //        // calc result
    //        input = matmul(&weights, &input, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);
    //        input += consts;
    //        // todo!("use consts and correct activation func");
    //        //input += consts;
    //        // apply activation function: x -> 1 / (1 + e^-x)
    //        // input = constant(0.0_f32, input.dims()) - input;
    //        match self.activation_function {
    //            ActivationFunction::Sigmoid => {
    //                input = -input;
    //                input /= constant(10.0_f32, input.dims());
    //                input = exp(&input);
    //                input += constant(1.0_f32, input.dims());
    //                input = div(&1.0_f32, &input, false);
    //                //input = div(&1.0_f32, &(constant(1.0_f32, input.dims()) + exp(&(-input/constant(10.0_f32, &input.dims())))), false);
    //            }
    //            ActivationFunction::Arctan => {
    //                todo!()
    //            }
    //            ActivationFunction::SignSqrtAbs => {
    //                input = sign(&input) * sqrt(&abs(&input));
    //            }
    //            // ActivationFunction::SignLnAbs => {
    //            //     input = sign(&input) * log(&abs(&input));
    //            // }
    //            // ActivationFunction::Linear => {
    //            //     // input = input;
    //            // }
    //        }
    //        // assert_eq!(
    //        //     self.weight[l].len(),
    //        //     input.elements()
    //        // );
    //    }
    //    // assert_eq!(
    //    //     1,
    //    //     input.elements()
    //    // );
    //    let mut result: Vec<f32> = vec![0.0; self.weight.last().unwrap().len()];
    //    input.host(&mut result);
    //    return result;
    //}


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

    pub fn evolve(&mut self, evolution_factor: f32, rng: &mut ThreadRng) {
        assert!(0.0 <= evolution_factor && evolution_factor <= 1.0);

        let total_neurons: u32 = self.get_total_neurons() as u32;
        let neurons_to_evolve: u32 = ((total_neurons as f32) * evolution_factor) as u32;
        let neurons_to_evolve: u32 = 1 + rng.gen_range(0..=neurons_to_evolve);
        // println!("total_neurons = {}", total_neurons);
        // println!("neurons_to_evolve = {}", neurons_to_evolve);

        if rng.gen_bool(0.1) {
            self.activation_function = get_random_activation_function(rng);
        }

        for _ in 0..neurons_to_evolve {
            let (l, h) = self.choose_random_neuron(rng);

            let total_weights: usize = self.weight[l][h].len();
            let weights_to_evolve: u32 = ((total_weights as f32) * evolution_factor) as u32;
            let weights_to_evolve: u32 = 1 + rng.gen_range(0..=weights_to_evolve);
            // println!("total_weights = {}", total_weights);
            // println!("weights_to_evolve = {}", weights_to_evolve);

            let sign: f32 = if rng.gen_bool((evolution_factor/2.0) as f64) { -1.0 } else { 1.0 };
            self.consts[l][h] *= sign;

            if rng.gen_bool(0.5) {
                self.consts[l][h] *= (1.0 + evolution_factor).powi(2);
            } else {
                self.consts[l][h] /= (1.0 + evolution_factor).powi(2);
            }
            if rng.gen_bool(0.5) {
                self.consts[l][h] += evolution_factor / 10.0;
            } else {
                self.consts[l][h] -= evolution_factor / 10.0;
            }

            for _ in 0..weights_to_evolve {
                // println!("old value: {}", self.weights[h]);
                let c: usize = rng.gen_range(0..total_weights);

                let sign: f32 = if rng.gen_bool((evolution_factor/2.0) as f64) { -1.0 } else { 1.0 };
                self.weight[l][h][c] *= sign;

                if rng.gen_bool(0.5) {
                    self.weight[l][h][c] *= (1.0 + evolution_factor).powi(2);
                } else {
                    self.weight[l][h][c] /= (1.0 + evolution_factor).powi(2);
                }
                if rng.gen_bool(0.5) {
                    self.weight[l][h][c] += evolution_factor / 10.0;
                } else {
                    self.weight[l][h][c] -= evolution_factor / 10.0;
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





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manual() {
        assert_eq!(
            NeuralNetwork::new(4, &vec![3, 2]),
            NeuralNetwork {
                input_len: 4,
                weight: vec![
                    vec![
                        vec![1.0, 1.0, 1.0, 1.0],
                        vec![1.0, 1.0, 1.0, 1.0],
                        vec![1.0, 1.0, 1.0, 1.0],
                    ],
                    vec![
                        vec![1.0, 1.0, 1.0],
                        vec![1.0, 1.0, 1.0],
                    ],
                    vec![
                        vec![1.0, 1.0],
                    ],
                ],
                consts: vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0],
                    vec![0.0],
                ],
                activation_function: ActivationFunction::Sigmoid
            }
        );
    }

    #[test]
    fn new_auto() {
        let test_cases: Vec<(usize, Vec<usize>)> = vec![
            (4, vec![3, 2]),
            (10, vec![8, 6, 4, 2]),
            (64, vec![300, 70, 20, 5]),
            (1000, vec![800, 600, 400, 200]),
            (1000, vec![800, 900, 600, 700, 400, 500, 200, 300]),
            (1000, vec![900, 800, 700, 600, 500, 400, 300, 200, 100]),
            (1111, vec![999, 888, 777, 666, 555, 444, 333, 222, 111]),
        ];
        for test_case in test_cases {
            let (input_len, heights) = test_case;
            let nn: NeuralNetwork = NeuralNetwork::new(input_len, &heights);
            let layers: usize = heights.len() + 1;
            assert_eq!(layers, nn.weight.len());
            assert_eq!(layers, nn.consts.len());
            for l in 0..layers {
                let height: usize = if l != layers-1 { heights[l] } else { 1 };
                assert_eq!(height, nn.consts[l].len());
                assert_eq!(height, nn.weight[l].len());
                let connections: usize = match l {
                    0 => { input_len }
                    l if l == layers => { 1 }
                    _ => { heights[l-1] }
                };
                for h in 0..height {
                    assert_eq!(connections, nn.weight[l][h].len());
                }
            }
        }
    }

    #[test]
    fn with_consts() {
        let input_len: usize = 5;
        let heights: Vec<usize> = vec![3];
        let nn: NeuralNetwork = NeuralNetwork::with_consts(
            input_len,
            &heights,
            1.45,
            4.2,
            ActivationFunction::SignSqrtAbs,
        );
        assert_eq!(
            nn,
            NeuralNetwork {
                input_len,
                weight: vec![
                    vec![
                        vec![1.45, 1.45, 1.45, 1.45, 1.45],
                        vec![1.45, 1.45, 1.45, 1.45, 1.45],
                        vec![1.45, 1.45, 1.45, 1.45, 1.45],
                    ],
                    vec![
                        vec![1.45, 1.45, 1.45],
                    ],
                ],
                consts: vec![
                    vec![4.2, 4.2, 4.2],
                    vec![4.2],
                ],
                activation_function: ActivationFunction::SignSqrtAbs
            }
        );
    }

    #[test]
    fn process_input_cpu() {
        let input_len: usize = 5;
        let heights: Vec<usize> = vec![3]; // -> [5, 3, 1]
        let nn: NeuralNetwork = NeuralNetwork::with_consts(
            input_len,
            &heights,
            1.45,
            4.2,
            ActivationFunction::SignSqrtAbs,
        );
        // layer 0 (input): inp0, inp1, inp2, inp3, inp4
        //     res0 = ssa(4.2 + inp0*1.45 + inp1*1.45 + inp2*1.45 + inp3*1.45 + inp4*1.45)
        //     res1 = ssa(4.2 + inp0*1.45 + inp1*1.45 + inp2*1.45 + inp3*1.45 + inp4*1.45)
        //     res2 = ssa(4.2 + inp0*1.45 + inp1*1.45 + inp2*1.45 + inp3*1.45 + inp4*1.45)
        //     inp_for_next_layer = [res0, res1, res2]
        // layer 1 (inner): inp0, inp1, inp2
        //     res0 = ssa(4.2 + inp0*1.45 + inp1*1.45 + inp2*1.45)
        //     inp_for_next_layer = [res0]
        // layer 2 (output): inp0 -> result/output
        //
        // this results can be double checked using python3:
        // ```python3
        // >>> sign = lambda x: 0.0 if x==0 else (-1.0 if x<0 else 1.0)
        // >>> from math import sqrt
        // >>> ssa = lambda x: sign(x) * sqrt(abs(x))
        // >>> 0.0
        // 0.0
        // >>> ssa(4.2  +  1.45 * _ * 5)
        // 2.04939015319192
        // >>> ssa(4.2  +  1.45 * _ * 3)
        // 3.6214426913020246 # this is expected result
        // >>>
        // >>> 1.0
        // 1.0
        // >>> ssa(4.2  +  1.45 * _ * 5)
        // 3.383784863137726
        // >>> ssa(4.2  +  1.45 * _ * 3)
        // 4.349651038261473 # this is expected result
        // ```
        assert_eq!(3.6214426913020246, nn.process_input_cpu(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(4.349651038261473 , nn.process_input_cpu(&[1.0, 1.0, 1.0, 1.0, 1.0]));
        assert_eq!(4.79696998427992  , nn.process_input_cpu(&[2.0, 2.0, 2.0, 2.0, 2.0]));
        assert_eq!(5.645009901861524 , nn.process_input_cpu(&[5.0, 5.0, 5.0, 5.0, 5.0]));

        assert_eq!(3.973780813934329 , nn.process_input_cpu(&[2.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(3.973780813934329 , nn.process_input_cpu(&[0.0, 2.0, 0.0, 0.0, 0.0]));
        assert_eq!(3.973780813934329 , nn.process_input_cpu(&[0.0, 0.0, 2.0, 0.0, 0.0]));
        assert_eq!(3.973780813934329 , nn.process_input_cpu(&[0.0, 0.0, 0.0, 2.0, 0.0]));
        assert_eq!(3.973780813934329 , nn.process_input_cpu(&[0.0, 0.0, 0.0, 0.0, 2.0]));
    }

    #[ignore]
    #[test]
    fn process_input_gpu() {
        panic!("plz enable `GPU`");
    }

}

