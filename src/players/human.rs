//! Human.

use std::str::FromStr;

use chess::{Board, ChessMove, File, Piece, Rank, Square};

use crate::utils_io::prompt;

use super::{MaybeChessMove, Player};


pub struct Human;

impl Human {
    fn select_move(&self, board: Board) -> MaybeChessMove {
        const CMD_QUIT: &str = "q";
        const CMD_SURRENDER: &str = "s";
        loop {
            let line = prompt("Your move: ");
            match line.as_str() {
                CMD_QUIT => { return MaybeChessMove::Quit }
                CMD_SURRENDER => { return MaybeChessMove::Surrender }
                _ => {}
            }
            let move_ = string_to_chess_move(&line);
            if let Some(move_) = move_ && board.legal(move_) {
                return MaybeChessMove::Move(move_)
            }
            println!("Illegal move.");
        }
    }
}

impl Player for Human {
    fn select_move(&self, board: Board) -> Option<MaybeChessMove> {
        Some(self.select_move(board))
    }
}


fn string_to_chess_move(line: &str) -> Option<ChessMove> {
    if !(4..=5).contains(&line.len()) { return None }
    let chars: Vec<char> = line.chars().collect();
    let (from_file, from_rank, to_file, to_rank, promote_to) = (chars[0], chars[1], chars[2], chars[3], chars.get(4));
    // TODO(refactor): redo without intermediate `String`. How? Is it even possible?
    let from_file = File::from_str(&from_file.to_string()).ok()?;
    let from_rank = Rank::from_str(&from_rank.to_string()).ok()?;
    let to_file = File::from_str(&to_file.to_string()).ok()?;
    let to_rank = Rank::from_str(&to_rank.to_string()).ok()?;
    mod promotes {
        pub const ALL: [char; 4] = [KNIGHT, BISHOP, ROOK, QUEEN];
        pub const KNIGHT: char = 'n';
        pub const BISHOP: char = 'b';
        pub const ROOK  : char = 'r';
        pub const QUEEN : char = 'q';
    }
    if let Some(promote_to) = promote_to && !promotes::ALL.contains(promote_to) { return None }
    let promote_to: Option<Piece> = promote_to.map(|&ch| match ch {
        promotes::KNIGHT => Piece::Knight,
        promotes::BISHOP => Piece::Bishop,
        promotes::ROOK   => Piece::Rook,
        promotes::QUEEN  => Piece::Queen,
        _ => unreachable!()
    });
    Some(ChessMove::new(
        Square::make_square(from_rank, from_file),
        Square::make_square(to_rank, to_file),
        promote_to,
    ))
}

