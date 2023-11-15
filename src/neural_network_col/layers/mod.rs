//! Layer trait

pub mod activation_functions;
pub mod fully_connected;


use crate::float_type::float;

use super::vector_type::Vector;

use self::{
    activation_functions::*,
    fully_connected::FullyConnected,
};


#[allow(dead_code, non_camel_case_types)]
pub enum LayerSpecs {
    /// Contains `output_size`.
    FullyConnected(usize),
    AF_Abs,
    AF_BinaryStep,
    AF_Elu,
    AF_Gaussian,
    AF_LeakyRelu,
    AF_MaxOut,
    AF_Relu,
    AF_Sigmoid,
    AF_Signum,
    AF_SignSqrtAbs,
    AF_Silu,
    AF_SoftMax,
    AF_SoftPlus,
    AF_Tanh,
}

impl LayerSpecs {
    pub fn to_layer(&self, input_size: &mut usize) -> BoxDynLayer {
        type LS = LayerSpecs;
        match *self {
            LS::FullyConnected(output_size) => {
                let fc = Box::new(FullyConnected::new(*input_size, output_size));
                *input_size = output_size;
                fc
            },
            LS::AF_Abs => Box::new(AF_Abs::new(*input_size)),
            LS::AF_BinaryStep => Box::new(AF_BinaryStep::new(*input_size)),
            LS::AF_Elu => Box::new(AF_Elu::new(*input_size)),
            LS::AF_Gaussian => Box::new(AF_Gaussian::new(*input_size)),
            LS::AF_LeakyRelu => Box::new(AF_LeakyRelu::new(*input_size)),
            LS::AF_MaxOut => Box::new(AF_MaxOut::new(*input_size)),
            LS::AF_Relu => Box::new(AF_Relu::new(*input_size)),
            LS::AF_Sigmoid => Box::new(AF_Sigmoid::new(*input_size)),
            LS::AF_Signum => Box::new(AF_Signum::new(*input_size)),
            LS::AF_SignSqrtAbs => Box::new(AF_SignSqrtAbs::new(*input_size)),
            LS::AF_Silu => Box::new(AF_Silu::new(*input_size)),
            LS::AF_SoftMax => Box::new(AF_SoftMax::new(*input_size)),
            LS::AF_SoftPlus => Box::new(AF_SoftPlus::new(*input_size)),
            LS::AF_Tanh => Box::new(AF_Tanh::new(*input_size)),
        }
    }
}


pub type BoxDynLayer = Box<dyn Layer + Send + Sync>;


pub trait Layer: CloneLayer + ToString {
    /// Returns layer `output` for given [`input`].
    fn forward_propagation(&self, input: Vector) -> Vector;

    /// Returns layer `output` for given [`input`].
    fn forward_propagation_for_training(&mut self, input: Vector) -> Vector;

    /// Update self params and returns layer `input_error` for given [`output_error`].
    fn backward_propagation(&mut self, output_error: Vector, learning_rate: float) -> Vector;
}


pub trait CloneLayer {
    fn clone_box(&self) -> Box<dyn Layer + Send + Sync>;
}
impl<T> CloneLayer for T
where T: 'static + Layer + Clone + Send + Sync
{
    fn clone_box(&self) -> Box<dyn Layer + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Layer + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

