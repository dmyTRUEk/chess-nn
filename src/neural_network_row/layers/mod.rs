//! Layer trait

pub mod activation_functions;
pub mod fully_connected;


use crate::{float_type::float, linalg_types::Matrix};

use super::vector_type::Vector;

use {activation_functions::*, fully_connected::FullyConnected};


#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
#[expect(dead_code)]
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
    AF_Silu,
    AF_SoftMax,
    AF_SoftPlus,
    AF_SymLn,
    AF_SymSqrt,
    AF_Tanh,
}

impl LayerSpecs {
    pub fn to_layer(&self, input_size: &mut usize) -> BoxDynLayer {
        match *self {
            Self::FullyConnected(output_size) => {
                let fc = Box::new(FullyConnected::new(*input_size, output_size));
                *input_size = output_size;
                fc
            },
            Self::AF_Abs => Box::new(AF_Abs::new(*input_size)),
            Self::AF_BinaryStep => Box::new(AF_BinaryStep::new(*input_size)),
            Self::AF_Elu => Box::new(AF_Elu::new(*input_size)),
            Self::AF_Gaussian => Box::new(AF_Gaussian::new(*input_size)),
            Self::AF_LeakyRelu => Box::new(AF_LeakyRelu::new(*input_size)),
            Self::AF_MaxOut => Box::new(AF_MaxOut::new(*input_size)),
            Self::AF_Relu => Box::new(AF_Relu::new(*input_size)),
            Self::AF_Sigmoid => Box::new(AF_Sigmoid::new(*input_size)),
            Self::AF_Signum => Box::new(AF_Signum::new(*input_size)),
            Self::AF_Silu => Box::new(AF_Silu::new(*input_size)),
            Self::AF_SoftMax => Box::new(AF_SoftMax::new(*input_size)),
            Self::AF_SoftPlus => Box::new(AF_SoftPlus::new(*input_size)),
            Self::AF_SymLn => Box::new(AF_SymLn::new(*input_size)),
            Self::AF_SymSqrt => Box::new(AF_SymSqrt::new(*input_size)),
            Self::AF_Tanh => Box::new(AF_Tanh::new(*input_size)),
        }
    }

    pub fn is_activation_function(&self) -> bool {
        match self {
            Self::FullyConnected(..) => false,
            Self::AF_Abs
            | Self::AF_BinaryStep
            | Self::AF_Elu
            | Self::AF_Gaussian
            | Self::AF_LeakyRelu
            | Self::AF_MaxOut
            | Self::AF_Relu
            | Self::AF_Sigmoid
            | Self::AF_Signum
            | Self::AF_Silu
            | Self::AF_SoftMax
            | Self::AF_SoftPlus
            | Self::AF_SymLn
            | Self::AF_SymSqrt
            | Self::AF_Tanh
            => true
        }
    }
}


pub type BoxDynLayer = Box<dyn Layer + Send + Sync>;


pub trait Layer: CloneLayer {
    /// Returns layer `output` for given [`input`].
    fn forward_propagation(&self, input: Vector) -> Vector;

    /// Returns layer `output` for given [`input`].
    fn forward_propagation_for_training(&mut self, input: Vector) -> Vector;

    /// Update self params and returns layer `input_error` for given [`output_error`].
    fn backward_propagation(&mut self, output_error: Vector, learning_rate: float) -> Vector;

    /// Get Weights and Shifts of FullyConnected layer.
    fn get_fc_weights_shifts(&self) -> Option<(&Matrix, &Vector)> { None }
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

