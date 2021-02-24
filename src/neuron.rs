/// This file contains all Neuron functions

use std::fmt;

use crate::activation_functions::activation_function;
use crate::random::*;



#[derive(Debug, Clone)]
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

        for _i in 0..weights_to_evolve {
            let h: usize = random_u32(0, total_weights-1) as usize;

            self.weights[h] *= (random_m1_p1() as f32) *
                if random_in_m1_p1() > 0.0 {
                    1.0 * (1.0 + evolution_factor)
                } else {
                    1.0 / (1.0 + evolution_factor)
                };

            // if random_in_m1_p1() > 0.0 {
            //     self.weights[h] *= (random_m1_p1() as f32) * (1.0 + evolution_factor);
            //     // self.weights[h] *= (random_in_m1_p1() as f32) * (1.0 + evolution_factor);   // THIS LINE IS WRONG!
            // }
            // else {
            //     self.weights[h] *= (random_m1_p1() as f32) / (1.0 + evolution_factor);
            //     // self.weights[h] *= (random_in_m1_p1() as f32) / (1.0 + evolution_factor);   // THIS LINE IS WRONG!
            // }
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
    fn test_process_input_size1_input_0 () {
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
    fn test_process_input_size2_input_0_0 () {
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
    fn test_process_input_size2_input_m1_1 () {
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
    fn test_process_input_size2_input_1_1 () {
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
    fn test_process_input_size3_input_1_2_3 () {
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
    fn test_set_size_0 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(0);
        assert_eq!(
            neuron.weights.len(),
            0
        );
    }
    
    #[test]
    fn test_set_size_1 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(1);
        assert_eq!(
            neuron.weights.len(),
            1
        );
    }
    
    #[test]
    fn test_set_size_2 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(2);
        assert_eq!(
            neuron.weights.len(),
            2
        );
    }
    
    #[test]
    fn test_set_size_5 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(5);
        assert_eq!(
            neuron.weights.len(),
            5
        );
    }
    
    #[test]
    fn test_set_size_100 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(100);
        assert_eq!(
            neuron.weights.len(),
            100
        );
    }
    
    #[test]
    fn test_set_size_1000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(1000);
        assert_eq!(
            neuron.weights.len(),
            1000
        );
    }
    
    #[test]
    fn test_set_size_10000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(10000);
        assert_eq!(
            neuron.weights.len(),
            10000
        );
    }
    
    #[test]
    fn test_set_size_100000 () {
        let mut neuron = Neuron{value: 0.0, weights: vec![]};
        neuron.set_size(100000);
        assert_eq!(
            neuron.weights.len(),
            100000
        );
    }

}



