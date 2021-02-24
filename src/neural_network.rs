/// This file contains all NN functions

use std::fmt;

use crate::neuron::*;
use crate::random::*;



#[derive(Debug, Clone)]
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

        // let layers = self.neurons.len();

        let total_neurons: u32 = self.get_total_neurons();
        let neurons_to_evolve: u32 = ((total_neurons as f32) * evolution_factor) as u32;
        // println!("neurons_to_evolve = {}", neurons_to_evolve);

        for _i in 0..neurons_to_evolve {
            let l: usize = random_u32(0, self.neurons.len() as u32 - 1) as usize;
            let h: usize = random_u32(0, self.neurons[l].len() as u32 - 1) as usize;
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



#[cfg(test)]
mod tests {
    use super::*;

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

}
