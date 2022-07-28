/// Activation Functions

use rand::{Rng, thread_rng, prelude::ThreadRng};



#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    Sigmoid,        //   1 / (1 + e^-x)
    Step,           //   if x > 0 { 1 } else { 0 }
    StepWithLin,    //   x>1 => 1, x<0 => 0, else => x
    ReLU,           //   if x > 0 { x } else { 0 }
    LeakyReLU,      //   if x > 0 { x } else { 0.1 * x }
    SignSqrtAbs,    //   sign(x) * sqrt(abs(x))
    SignLnAbs,      //   sign(x) * ln(abs(x))
    // IMPORTANT: when adding new activation functions dont forget to add them to random
}

pub fn calc_activation_function(x: f32, activation_function: ActivationFunction) -> f32 {
    match activation_function {
        ActivationFunction::Sigmoid => {
            1.0 / (1.0 + (-x).exp())
        }
        ActivationFunction::Step => {
            if x < 0.0 { 0.0 } else { 1.0 }
        }
        ActivationFunction::StepWithLin => {
            if x < 0.0 { 0.0 } else if x < 1.0 { x } else { 1.0 }
        }
        ActivationFunction::ReLU => {
            if x > 0.0 { x } else { 0.0 }
        }
        ActivationFunction::LeakyReLU => {
            if x > 0.0 { x } else { 0.1 * x }
        }
        ActivationFunction::SignSqrtAbs => {
            x.signum() * x.abs().sqrt()
        }
        ActivationFunction::SignLnAbs => {
            x.signum() * x.abs().ln()
        }
    }
}

pub fn get_random_activation_function() -> ActivationFunction {
    let mut rng: ThreadRng = thread_rng();
    match rng.gen_range(0..7) {
        0 => ActivationFunction::Sigmoid,
        1 => ActivationFunction::Step,
        2 => ActivationFunction::StepWithLin,
        3 => ActivationFunction::ReLU,
        4 => ActivationFunction::LeakyReLU,
        5 => ActivationFunction::SignSqrtAbs,
        6 => ActivationFunction::SignLnAbs,
        _ => panic!()
    }
}

