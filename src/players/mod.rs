//! Players: AI(NN), Human, Finite State Machine, etc.

pub mod ai;
pub mod finite_state_machine;
pub mod human;
pub mod rating;


use chess::{Board, ChessMove};


pub type BoxDynPlayer<'a> = Box<&'a (dyn Player + Send + Sync)>;

pub trait Player {
    // TODO(optimization?): pass by ref.
    fn select_move(&self, board: Board) -> Option<MaybeChessMove>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaybeChessMove {
    Move(ChessMove),
    Surrender,
    Quit,
}

