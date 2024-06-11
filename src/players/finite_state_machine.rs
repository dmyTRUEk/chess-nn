//! Finite State Machine.

use std::collections::HashMap;

use chess::{Board, ChessMove};

use crate::select_random_move;

use super::{MaybeChessMove, Player};


pub struct FiniteStateMachine {
	decisions: HashMap<Board, ChessMove>,
}

impl FiniteStateMachine {
	#[expect(dead_code)]
	pub fn new(decisions: HashMap<Board, ChessMove>) -> Self {
		Self { decisions }
	}

	fn select(&self, board: &Board) -> Option<ChessMove> {
		self.decisions.get(board).copied()
	}
}

impl Player for FiniteStateMachine {
	fn select_move(&self, board: Board) -> Option<MaybeChessMove> {
		Some(MaybeChessMove::Move(
				self.select(&board)
				.unwrap_or_else(|| select_random_move(&board))
		))
	}
}

