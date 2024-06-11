//! Player with Rating.

use std::{
	cmp::Ordering,
	ops::{AddAssign, Div, SubAssign},
};

use crate::{DEFAULT_RATING, Winner, float_type::float};


#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Rating(float);

impl Rating {
	pub fn get(&self) -> float {
		self.0
	}
}

impl Default for Rating {
	fn default() -> Self {
		Self(DEFAULT_RATING)
	}
}

// yes, it's needed, but if you implement it then it leaks a private type `RatingDelta`
// impl Add<RatingDelta> for Rating {
//     type Output = Self;
//     fn add(self, rhs: RatingDelta) -> Self::Output {
//         Self(self.0 + rhs.0)
//     }
// }

impl AddAssign<RatingDelta> for Rating {
	fn add_assign(&mut self, rhs: RatingDelta) {
		self.0 += rhs.0;
	}
}

// impl Sub for Rating {
//     type Output = RatingDelta;
//     fn sub(self, rhs: Self) -> Self::Output {
//         RatingDelta::new(self, rhs)
//     }
// }

impl SubAssign<RatingDelta> for Rating {
	fn sub_assign(&mut self, rhs: RatingDelta) {
		self.0 -= rhs.0;
	}
}


#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct RatingDelta(float);

impl RatingDelta {
	fn new(rating_1: Rating, rating_2: Rating) -> Self {
		Self(rating_1.0 - rating_2.0)
	}

	fn max(self, other: Self) -> Self {
		match self.0.partial_cmp(&other.0).expect("`rating::RatingDelta::max`: can't compare") {
			Ordering::Greater => self,
			Ordering::Less => other,
			Ordering::Equal => self, // or `other`, dont matter, they are the same
		}
	}
}

impl Div<float> for RatingDelta {
	type Output = Self;
	fn div(self, rhs: float) -> Self::Output {
		Self(self.0 / rhs)
	}
}

// TODO(refactor): type RatingDelta(float)

fn elo_rating_delta(rating_delta: RatingDelta) -> RatingDelta {
	let x: float = rating_delta.0;
	// TODO(refactor): extract into consts?
	let delta: float = 100. / ( 1. + (10. as float).powf(x / 400.) );
	RatingDelta(delta)
}

pub fn update_ratings(white_rating: &mut Rating, black_rating: &mut Rating, winner: Winner) {
	// const WINNER_SCALE: float = 1.;
	const LOSE_SCALE: float = 1.;
	const DRAW_SCALE_STRONGER: float = 5.;
	const DRAW_SCALE_WEAKER  : float = 5.;
	let delta_rating_w = elo_rating_delta(RatingDelta::new(*white_rating, *black_rating));
	let delta_rating_b = elo_rating_delta(RatingDelta::new(*black_rating, *white_rating));
	// TODO(refactor): how?
	match winner {
		Winner::White => {
			*white_rating += delta_rating_w;
			*black_rating -= delta_rating_w / LOSE_SCALE;
		}
		Winner::Black => {
			*white_rating -= delta_rating_b / LOSE_SCALE;
			*black_rating += delta_rating_b;
		}
		Winner::Draw => {
			// let delta_rating_min: float = delta_rating_w.min(delta_rating_b);
			let delta_rating_max: RatingDelta = delta_rating_w.max(delta_rating_b);
			match white_rating.partial_cmp(&black_rating).unwrap() {
				Ordering::Greater => {
					*white_rating -= delta_rating_max / DRAW_SCALE_STRONGER;
					*black_rating += delta_rating_max / DRAW_SCALE_WEAKER;
				}
				Ordering::Less => {
					*white_rating += delta_rating_max / DRAW_SCALE_WEAKER;
					*black_rating -= delta_rating_max / DRAW_SCALE_STRONGER;
				}
				_ => {}
			}
		}
	}
}

