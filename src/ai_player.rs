//! AI Player = AI + rating

use crate::{
    ChessNeuralNetwork,
    DEFAULT_RATING,
    ai::AI,
    float_type::float,
};


#[allow(non_camel_case_types)]
#[derive(Clone)]
pub struct AI_Player {
    pub ai: AI,
    pub rating: float,
}

impl AI_Player {
    pub const fn new(name: String, nn: ChessNeuralNetwork) -> Self {
        Self {
            ai: AI::new(name, nn),
            rating: DEFAULT_RATING,
        }
    }
}

impl From<AI> for AI_Player {
    fn from(ai: AI) -> Self {
        Self::new(ai.name, ai.nn)
    }
}

