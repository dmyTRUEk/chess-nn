/// This file contains all activation functions



pub fn activation_function (input: f32) -> f32 {
    // linear(input)
    // sigmoid(input)
    // sigmoid_m05(input)
    sign_x_sqrt_abs_x(input)
}



#[allow(dead_code)]
#[allow(non_snake_case)]
fn linear (x: f32) -> f32 {
    x
}

#[allow(dead_code)]
#[allow(non_snake_case)]
fn exp (x: f32) -> f32 {
    x.exp()
}

#[allow(dead_code)]
#[allow(non_snake_case)]
fn sigmoid (x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[allow(dead_code)]
#[allow(non_snake_case)]
fn sigmoid_m05 (x: f32) -> f32 {
    1.0 / (1.0 + exp(-x)) - 0.5
}

#[allow(dead_code)]
#[allow(non_snake_case)]
fn ReLU (x: f32) -> f32 {
    if x > 0.0 {
        x
    }
    else {
        0.0
    }
}

#[allow(dead_code)]
#[allow(non_snake_case)]
fn sign_x_sqrt_abs_x (x: f32) -> f32 {
    x.signum() * x.abs().sqrt()
}


