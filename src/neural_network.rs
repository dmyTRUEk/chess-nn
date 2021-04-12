/// This file contains all NN functions

use std::fmt;

use crate::random::*;

use crate::activation_functions::activation_function;



#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    pub neurons: Vec<Vec<Neuron>>,
}

pub fn create_nn (heights: &Vec<usize>) -> NeuralNetwork {
    let mut nn = NeuralNetwork{
        neurons: vec![
            vec![
                Neuron{value: 0.0, weights: vec![]};
                0
            ];
            0
        ]
    };
    nn.set_nn_heights(heights);
    nn
}

pub fn create_nn_with_const_weights (heights: &Vec<usize>, weight_const: f32) -> NeuralNetwork {
    let mut nn = create_nn(heights);
    let layers = heights.len();
    for l in 1..layers {
        for h in 0..nn.neurons[l].len() {
            nn.neurons[l][h].init_weights_by_value(weight_const);
        }
    }
    nn
}

pub fn create_nn_with_random_weights (heights: &Vec<usize>, weight_min: f32, weight_max: f32) -> NeuralNetwork {
    let mut nn = create_nn(heights);
    nn.init_weights_by_random(weight_min, weight_max);
    nn
}

impl NeuralNetwork {
    pub fn set_nn_heights (&mut self, heights: &Vec<usize>) {
        let layers = heights.len();

        // create nn
        self.neurons = vec![
            vec![
                Neuron{value: 0.0, weights: vec![]};
                heights[0]
            ];
            layers
        ];

        // set right heights for nn
        for l in 0..layers {
            self.neurons[l] = vec![
                Neuron{value: 0.0, weights: vec![]};
                heights[l]
            ]
        }

        // set weights size = 0 for every neuron in first (0) layer
        // let height: usize = heights[0];
        for h in 0..heights[0] {
            self.neurons[0][h].set_size(0);
        }

        // set weights size for neurons in all other layers
        for l in 1..layers {
            let height = heights[l];
            for h in 0..height {
                self.neurons[l][h].set_size(heights[l-1]);
            }
        }
    }

    pub fn process_input (&mut self, input: &Vec<f32>) -> Vec<f32> {
        let layers = self.neurons.len();
        // let mut height_this: usize;
        // let mut height_next: usize;
        
        // set results of neurons in first layer (input for nn)
        for h in 0..self.neurons[0].len() {
            self.neurons[0][h].value = input[h];
        }

        let mut input_for_this_layer = input.clone();
        let mut input_for_next_layer: Vec<f32>;

        for l in 1..layers {
            let height_this = self.neurons[l].len();
            // height_next = self.neurons[l+1].len();
            input_for_next_layer = vec![0.0; height_this];

            // println!("\nnn = {}\n", self);

            for h in 0..height_this {
                input_for_next_layer[h] = self.neurons[l][h].process_input(&input_for_this_layer);
                // println!("    input_for_next_layer[{}] = {}", h, input_for_next_layer[h]);
            }
            input_for_this_layer = input_for_next_layer.clone();
            // println!("input_for_this_layer = {:?}", input_for_this_layer);
        }

        input_for_this_layer
    }

    pub fn init_weights_by_random (&mut self, weight_min: f32, weight_max: f32) {
        let layers = self.neurons.len();
        // let mut height_this: usize;

        for l in 0..layers {
            let height_this = self.neurons[l].len();
            for h in 0..height_this {
                self.neurons[l][h].init_weights_by_random(weight_min, weight_max);
            }
        }
    }

    pub fn get_total_neurons (&self) -> u32 {
        let layers = self.neurons.len();
        let mut res: u32 = 0;
        // from 1, because 0 layer doesnt calculate anything
        for l in 1..layers {
            res += self.neurons[l].len() as u32;
        }
        res
    }

    pub fn evolve (&mut self, evolution_factor: f32) {
        assert!(0.0 <= evolution_factor && evolution_factor <= 1.0);

        let total_neurons: u32 = self.get_total_neurons();
        let neurons_to_evolve: u32 = ((total_neurons as f32) * evolution_factor) as u32;
        let neurons_to_evolve: u32 = random_u32(1, neurons_to_evolve.max(1));
        // println!("total_neurons = {}", total_neurons);
        // println!("neurons_to_evolve = {}", neurons_to_evolve);

        for _i in 0..neurons_to_evolve {
            // let l: usize = random_u32(1, self.neurons.len() as u32 - 1) as usize;
            // let h: usize = random_u32(0, self.neurons[l].len() as u32 - 1) as usize;
            let (l, h): (usize, usize) = {
                let neuron_id_to_evolve: usize = random_u32(0, total_neurons-1) as usize;
                let mut l: usize = 1;
                let mut h: usize = 0;
                for _j in 0..neuron_id_to_evolve {
                    if h < self.neurons[l].len()-1 {
                        h += 1;
                    }
                    else {
                        l += 1;
                        h = 0;
                    }
                }
                (l, h)
            };
            self.neurons[l][h].evolve(evolution_factor);
        }
    }
}



impl fmt::Display for NeuralNetwork {
    fn fmt (&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = String::from("{");

        let layers = self.neurons.len();
        let mut height: usize;
        for l in 0..layers {
            height = self.neurons[l].len();
            for h in 0..height {
                res += &format!("\n    neuron[{}][{}] = {},", l, h, self.neurons[l][h]);
            }
            res += "\n";
        }
        res += "}\n";

        write!(f, "{}", res)
    }
}





#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub value: f32,
}

impl Neuron {
    pub fn set_size (&mut self, size: usize) {
        self.weights = vec![1.0; size];
    }

    pub fn init_weights_by_value (&mut self, v: f32) {
        for weight in &mut self.weights {
            *weight = v;
        }
    }

    pub fn init_weights_by_random (&mut self, min: f32, max: f32) {
        for weight in &mut self.weights {
            *weight = random_f32(min, max);
        }
    } 

    pub fn process_input (&mut self, input: &Vec<f32>) -> f32 {
        let mut sum: f32 = 0.0;

        for h in 0..self.weights.len() {
            sum += input[h] * self.weights[h];
        }
        self.value = activation_function(sum);
        self.value
    }

    pub fn evolve (&mut self, evolution_factor: f32) {
        assert!(0.0 <= evolution_factor && evolution_factor <= 1.0);

        let total_weights: u32 = self.weights.len() as u32;
        let weights_to_evolve: u32 = ((total_weights as f32) * evolution_factor) as u32;
        let weights_to_evolve: u32 = random_u32(1, weights_to_evolve.max(1));
        // println!("total_weights = {}", total_weights);
        // println!("weights_to_evolve = {}", weights_to_evolve);

        for _i in 0..weights_to_evolve {
            let h: usize = random_u32(0, total_weights-1) as usize;

            let sign: f32 = if random_f32_0_p1() < 0.3 { -1.0 } else { 1.0 };

            // println!("old value: {}", self.weights[h]);
            self.weights[h] *= sign *
                if random_f32_m1_p1() > 0.0 {
                    // 1.0 * (1.0 + evolution_factor)
                    1.0 * (1.1)
                } else {
                    // 1.0 / (1.0 + evolution_factor)
                    1.0 / (1.1)
                };
            // println!("new value: {}", self.weights[h]);
        }
    }
}



impl fmt::Display for Neuron {
    fn fmt (&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{value = {},   weights = {:?}}}", self.value, self.weights)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_process_input_size1_input_0 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(1);
        neuron.init_weights_by_value(1.0);
        let result: f32 = neuron.process_input(&vec![0.0]);
        assert_eq!(
            result, 
            activation_function(0.0)
        );
    }
    
    #[test]
    fn test_neuron_process_input_size2_input_0_0 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(2);
        neuron.init_weights_by_value(1.0);
        let result: f32 = neuron.process_input(&vec![0.0, 0.0]);
        assert_eq!(
            result, 
            activation_function(0.0)
        );
    }
    
    #[test]
    fn test_neuron_process_input_size2_input_m1_1 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(2);
        neuron.init_weights_by_value(1.0);
        let result: f32 = neuron.process_input(&vec![-1.0, 1.0]);
        assert_eq!(
            result, 
            activation_function(0.0)
        );
    }
    
    #[test]
    fn test_neuron_process_input_size2_input_1_1 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(2);
        neuron.init_weights_by_value(1.0);
        let result: f32 = neuron.process_input(&vec![1.0, 1.0]);
        assert_eq!(
            result, 
            activation_function(2.0)
        );
    }
    
    #[test]
    fn test_neuron_process_input_size3_input_1_2_3 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(3);
        neuron.init_weights_by_value(2.0);
        let result: f32 = neuron.process_input(&vec![1.0, 2.0, 3.0]);
        assert_eq!(
            result, 
            activation_function(12.0)
        );
    }
    
    
    
    #[test]
    fn test_neuron_set_size_0 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(0);
        assert_eq!(
            neuron.weights.len(),
            0
        );
    }
    
    #[test]
    fn test_neuron_set_size_1 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(1);
        assert_eq!(
            neuron.weights.len(),
            1
        );
    }
    
    #[test]
    fn test_neuron_set_size_2 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(2);
        assert_eq!(
            neuron.weights.len(),
            2
        );
    }
    
    #[test]
    fn test_neuron_set_size_5 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(5);
        assert_eq!(
            neuron.weights.len(),
            5
        );
    }
    
    #[test]
    fn test_neuron_set_size_100 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(100);
        assert_eq!(
            neuron.weights.len(),
            100
        );
    }
    
    #[test]
    fn test_neuron_set_size_1000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(1000);
        assert_eq!(
            neuron.weights.len(),
            1000
        );
    }
    
    #[test]
    fn test_neuron_set_size_10000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(10000);
        assert_eq!(
            neuron.weights.len(),
            10000
        );
    }
    
    #[test]
    fn test_neuron_set_size_100000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(100000);
        assert_eq!(
            neuron.weights.len(),
            100000
        );
    }

    #[ignore]
    #[test]
    fn test_process_nn_heights_1_1_weights_1_1_input_6 () {
        println!("[WARNING]: maybe it dont work, because of wrong activation function used");
        let heights: Vec<usize> = vec![1, 1];
        let input: Vec<f32> = vec![6.0];
        let mut nn = create_nn(&heights);
        assert_eq!(
            nn.process_input(&input),
            vec![6.0]
        );
    }

    #[ignore]
    #[test]
    fn test_process_nn_heights_3_2_weights_1_1_1_1_1_1_input_1_2_3 () {
        println!("[WARNING]: maybe it dont work, because of wrong activation function used");
        let heights: Vec<usize> = vec![3, 2];
        let input: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut nn = create_nn(&heights);
        assert_eq!(
            nn.process_input(&input),
            vec![6.0, 6.0]
        );
    }

    #[test]
    fn test_create_nn_heights_0 () {
        let heights: Vec<usize> = vec![0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1 () {
        let heights: Vec<usize> = vec![1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_9 () {
        let heights: Vec<usize> = vec![9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_0 () {
        let heights: Vec<usize> = vec![0, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_1 () {
        let heights: Vec<usize> = vec![0, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_0 () {
        let heights: Vec<usize> = vec![1, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_1 () {
        let heights: Vec<usize> = vec![1, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_9_1 () {
        let heights: Vec<usize> = vec![9, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_9 () {
        let heights: Vec<usize> = vec![1, 9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_8_9 () {
        let heights: Vec<usize> = vec![8, 9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_88_99 () {
        let heights: Vec<usize> = vec![88, 99];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[ignore]
    #[test]
    fn test_create_nn_heights_888_999 () {
        let heights: Vec<usize> = vec![888, 999];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[ignore]
    #[test]
    fn test_create_nn_heights_8888_9999 () {
        // takes about 280Mb
        let heights: Vec<usize> = vec![8888, 9999];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_0_0 () {
        let heights: Vec<usize> = vec![0, 0, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_0_1 () {
        let heights: Vec<usize> = vec![0, 0, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_1_0 () {
        let heights: Vec<usize> = vec![0, 1, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_0_1_1 () {
        let heights: Vec<usize> = vec![0, 1, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_0_0 () {
        let heights: Vec<usize> = vec![1, 0, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_0_1 () {
        let heights: Vec<usize> = vec![1, 0, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_1_0 () {
        let heights: Vec<usize> = vec![1, 1, 0];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_1_1_1 () {
        let heights: Vec<usize> = vec![1, 1, 1];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_7_8_9 () {
        let heights: Vec<usize> = vec![7, 8, 9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_6_7_8_9 () {
        let heights: Vec<usize> = vec![6, 7, 8, 9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_create_nn_heights_5_6_7_8_9 () {
        let heights: Vec<usize> = vec![5, 6, 7, 8, 9];
        let nn = create_nn(&heights);
        assert_eq!(
            nn.neurons.len(),
            heights.len()
        );
        for i in 0..heights.len() {
            let h = heights[i];
            assert_eq!(
                nn.neurons[i].len(),
                h
            );
        }
    }

    #[test]
    fn test_evolution () {
        let nn_heights: Vec<usize> = vec![100, 200, 50];
        for _i in 0..100 {
            // println!("i = {}/1000", i);
            let nn1 = create_nn_with_random_weights(&nn_heights.clone(), -1.0, 1.0);
            let mut nn2 = nn1.clone();
            nn2.evolve(0.1);
            assert_ne!(nn1, nn2);
        }
    }

}
