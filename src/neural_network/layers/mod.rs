//! Layer trait

pub mod activation_functions;
pub mod fc_layer;

use crate::{
    float_type::float,
    linalg_types::RowVector,
};


pub trait Layer: CloneLayer + ToString {
    /// Creates a layer.
    fn new() -> Self where Self: Sized;

    /// Returns layer `output` for given [`input`].
    fn forward_propagation(&self, input: RowVector) -> RowVector;

    /// Returns layer `output` for given [`input`].
    fn forward_propagation_for_training(&mut self, input: RowVector) -> RowVector;

    /// Update self params and returns layer `input_error` for given [`output_error`].
    fn backward_propagation(&mut self, output_error: RowVector, learning_rate: float) -> RowVector;
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

