//! AI

pub mod ai;
pub mod generator;


use super::rating::Rating;

use ai::AI;


#[derive(Clone)]
pub struct AIwithRating {
    rating: Rating,
    ai: AI,
}

impl AIwithRating {
    pub fn new(ai: AI) -> Self {
        Self {
            rating: Rating::default(),
            ai,
        }
    }
    pub fn get_rating(&self) -> Rating {
        self.rating
    }
    pub fn get_rating_mut(&mut self) -> &mut Rating {
        &mut self.rating
    }
    pub fn get_ai(&self) -> &AI {
        &self.ai
    }
    pub fn get_ai_mut(&mut self) -> &mut AI {
        &mut self.ai
    }
}

