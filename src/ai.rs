//! AI = ChessNN + name

use crate::{
    ChessNeuralNetwork, // from main
};



#[derive(Clone)]
pub struct AI {
    pub name: String,
    pub nn: ChessNeuralNetwork,
}

impl AI {
    pub const fn new(name: String, nn: ChessNeuralNetwork) -> Self {
        Self {
            name,
            nn,
        }
    }
}

