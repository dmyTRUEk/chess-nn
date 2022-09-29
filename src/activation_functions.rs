/// Activation Functions

use rand::{Rng, prelude::ThreadRng};



#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    Sigmoid,        //   1 / (1 + e^-x)
    Arctan,         //   arctan(x)
    SignSqrtAbs,    //   sign(x) * sqrt(abs(x))
    // SignLnAbs,      //   sign(x) * ln(abs(x))
    // Linear,         //   x
    // Step,           //   if x > 0 { 1 } else { 0 }
    // StepWithLin,    //   x>1 => 1, x<0 => 0, else => x
    // ReLU,           //   if x > 0 { x } else { 0 }
    // LeakyReLU,      //   if x > 0 { x } else { 0.1 * x }
    // IMPORTANT: when adding new activation functions dont forget to add them to random
}

pub fn calc_activation_function(x: f32, activation_function: ActivationFunction) -> f32 {
    match activation_function {
        ActivationFunction::Sigmoid => {
            1.0 / (1.0 + (-x/10.0).exp())
        }
        ActivationFunction::Arctan => {
            x.atan()
        }
        ActivationFunction::SignSqrtAbs => {
            x.signum() * x.abs().sqrt()
        }
        // ActivationFunction::SignLnAbs => {
        //     x.signum() * x.abs().ln()
        // }
        // ActivationFunction::Linear => {
        //     x
        // }
        // ActivationFunction::Step => {
        //     if x < 0.0 { 0.0 } else { 1.0 }
        // }
        // ActivationFunction::StepWithLin => {
        //     if x < 0.0 { 0.0 } else if x < 1.0 { x } else { 1.0 }
        // }
        // ActivationFunction::ReLU => {
        //     if x > 0.0 { x } else { 0.0 }
        // }
        // ActivationFunction::LeakyReLU => {
        //     if x > 0.0 { x } else { 0.1 * x }
        // }
    }
}

// TODO: pass &mut rng
pub fn get_random_activation_function(rng: &mut ThreadRng) -> ActivationFunction {
    match rng.gen_range(0..3) {
        0 => ActivationFunction::Sigmoid,
        1 => ActivationFunction::Arctan,
        2 => ActivationFunction::SignSqrtAbs,
        // 0 => ActivationFunction::SignLnAbs,
        // 0 => ActivationFunction::Linear,
        // 0 => ActivationFunction::Step,
        // 0 => ActivationFunction::StepWithLin,
        // 0 => ActivationFunction::ReLU,
        // 0 => ActivationFunction::LeakyReLU,
        _ => panic!()
    }
}

