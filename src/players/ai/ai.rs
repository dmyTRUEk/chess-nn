//! AI.

use chess::{Board, ChessMove};

use crate::{
    CHESS_NN_THINK_DEPTH_FOR_TRAINING,
    CHESS_NN_THINK_DEPTH_IN_TOURNAMENT,
    CHESS_NN_THINK_DEPTH_VS_HUMAN,
    ChessNeuralNetwork, // from main
    players::MaybeChessMove,
};

use super::super::Player;


#[derive(Clone)]
pub struct AI {
    name: String,
    nn: ChessNeuralNetwork,
    thinking_depth: AI_ThinkingDepth,
}

impl AI {
    pub const fn new_for_training(name: String, nn: ChessNeuralNetwork) -> Self {
        Self { name, nn, thinking_depth: AI_ThinkingDepth::Training }
    }
    #[expect(dead_code)]
    pub const fn new(name: String, nn: ChessNeuralNetwork, thinking_depth: AI_ThinkingDepth) -> Self {
        Self { name, nn, thinking_depth }
    }
    pub fn get_name(&self) -> String {
        self.name.clone()
    }
    pub fn get_nn(&self) -> &ChessNeuralNetwork {
        &self.nn
    }
    pub fn get_nn_mut(&mut self) -> &mut ChessNeuralNetwork {
        &mut self.nn
    }
    pub fn set_mode(&mut self, thinking_depth: AI_ThinkingDepth) {
        self.thinking_depth = thinking_depth;
    }
    #[expect(dead_code)]
    pub fn set_training_mode(&mut self) {
        self.set_mode(AI_ThinkingDepth::Training);
    }
    #[expect(dead_code)]
    pub fn set_tournament_mode(&mut self) {
        self.set_mode(AI_ThinkingDepth::Tournament);
    }
    #[expect(dead_code)]
    pub fn set_vs_human_mode(&mut self) {
        self.set_mode(AI_ThinkingDepth::VsHuman);
    }
}

impl Player for AI {
    fn select_move(&self, board: Board) -> Option<MaybeChessMove> {
        let best_move: ChessMove = move_score_weight::choose_best_move(
            board,
            self.get_nn(),
            self.thinking_depth.get().get(),
        )?;
        Some(MaybeChessMove::Move(best_move))
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)] // TODO(later, when fixed): use `expect`.
pub enum AI_ThinkingDepth {
    // Default,
    Training,
    Tournament,
    VsHuman,
    #[expect(dead_code, private_interfaces)]
    Other(ThinkingDepth),
}
impl AI_ThinkingDepth {
    // const THINKING_DEPTH_DEFAULT       : ThinkingDepth = ThinkingDepth(1);
    const THINKING_DEPTH_FOR_TRAINING  : ThinkingDepth = ThinkingDepth(CHESS_NN_THINK_DEPTH_FOR_TRAINING);
    const THINKING_DEPTH_FOR_TOURNAMENT: ThinkingDepth = ThinkingDepth(CHESS_NN_THINK_DEPTH_IN_TOURNAMENT);
    const THINKING_DEPTH_FOR_VS_HMAN   : ThinkingDepth = ThinkingDepth(CHESS_NN_THINK_DEPTH_VS_HUMAN);
    const fn get(&self) -> ThinkingDepth {
        match self {
            // Self::Default => unreachable!("should set mode"),
            Self::Training => Self::THINKING_DEPTH_FOR_TRAINING,
            Self::Tournament => Self::THINKING_DEPTH_FOR_TOURNAMENT,
            Self::VsHuman => Self::THINKING_DEPTH_FOR_VS_HMAN,
            Self::Other(thinking_depth) => *thinking_depth,
        }
    }
}

// impl Default for AI_ThinkingDepth {
//     fn default() -> Self {
//         Self::Default
//     }
// }

#[derive(Debug, Clone, Copy)]
struct ThinkingDepth(u8);

impl ThinkingDepth {
    const fn get(&self) -> u8 {
        self.0
    }
}


// TODO(refactor): rename!!! and other
mod move_score_weight {
    use std::cmp::Ordering;
    use chess::{Board, ChessMove as Move, Color, MoveGen};
    use rand::{Rng, thread_rng};
    use crate::{
        NN_RESULT_RANDOM_CHOICE,
        board_to_vector_for_nn,
        float_type::float,
        neural_network_row::ChessNeuralNetwork,
    };

    pub fn choose_best_move(
        board: Board,
        nn: &ChessNeuralNetwork,
        depth: u8,
    ) -> Option<Move> {
        gen_variants(board, nn, depth)
            .choose_best_move_by_minimax(board.side_to_move())
    }

    // TODO(when done): remove redundant derives.

    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Score(float);
    // impl Hash for Score {
    //     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    //         self.0.to_bits().hash(state)
    //     }
    // }

    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Weight(float);
    impl Weight {
        const DEFAULT_VALUE: float = 1.;
    }
    // impl Hash for ChessMoveWeight {
    //     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    //         self.0.to_bits().hash(state)
    //     }
    // }
    impl Default for Weight {
        fn default() -> Self {
            Self(Self::DEFAULT_VALUE)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    struct ScoreWeight {
        score: Score,
        weight: Weight,
    }
    impl ScoreWeight {
        fn get_weighted(&self) -> ScoreWeighted {
            ScoreWeighted(self.score.0 * self.weight.0)
        }
    }
    // impl PartialOrd for ChessMoveScoreAndWeight {
    //     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    //         todo!()
    //     }
    // }

    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
    struct ScoreWeighted(float);

    #[derive(Debug, Copy, Clone, PartialEq)]
    struct MoveScoreWeight {
        move_: Move,
        score: Score,
        weight: Weight,
    }
    impl MoveScoreWeight {
        fn new(move_: Move, score: Score) -> Self {
            Self { move_, score, weight: Weight::default() }
        }
    }
    impl From<(Move, Score)> for MoveScoreWeight {
        fn from((move_, score): (Move, Score)) -> Self {
            Self::new(move_, score)
        }
    }

    impl MoveScoreWeight {
        fn get_weighted_score(&self) -> ScoreWeighted {
            ScoreWeight { score: self.score, weight: self.weight }.get_weighted()
        }
    }

    fn gen_variants(board: Board, nn: &ChessNeuralNetwork, depth: u8) -> Variants {
        let legal_moves = MoveGen::new_legal(&board);
        match depth {
            0 => unreachable!(),
            1 => {
                // TODO(optimization): parallel processing
                fn analyze(board: Board, nn: &ChessNeuralNetwork) -> float {
                    let input_for_nn = board_to_vector_for_nn(board);
                    // println!("input_for_nn = {:?}", array_board);
                    nn.process_input(input_for_nn)
                }
                let mut rng = thread_rng();
                let mut leaves: Vec<MoveScoreWeight> = Vec::with_capacity(legal_moves.len());
                for move_ in legal_moves {
                    let board_possible: Board = board.make_move_new(move_);
                    let score: float = analyze(board_possible, nn);
                    let weight = if let Some((w_min, w_max)) = NN_RESULT_RANDOM_CHOICE { rng.gen_range(w_min..w_max) } else { 1. };
                    let msw_possible = MoveScoreWeight { move_, score: Score(score), weight: Weight(weight) };
                    // TODO:
                    // if config.show_logs {
                    //     msws.push(msw_possible);
                    // }
                    leaves.push(msw_possible);
                }
                leaves.shrink_to_fit();

                // TODO:
                // if config.show_logs {
                //     msws.sort_by(|msw1, msw2| msw1.get_weighted_score().total_cmp(&msw2.get_weighted_score()));
                //     for msw in msws {
                //         let MoveScoreWeight { move_, score, weight } = msw;
                //         let weighted_score = msw.get_weighted_score();
                //         let ChessMoveScore(score) = score;
                //         // TODO: better formatting
                //         println!("{move_:<5} -> score = {score:<20} weight = {weight:<20} -> weighted_score = {weighted_score}");
                //     }
                // }

                Variants::Leaves(leaves)
            }
            _ => {
                let mut branches: Vec<(Move, Variants)> = Vec::with_capacity(legal_moves.len());
                for move_ in legal_moves {
                    let board_possible: Board = board.make_move_new(move_);
                    branches.push((
                        move_,
                        gen_variants(board_possible, nn, depth-1),
                    ));
                }
                branches.shrink_to_fit();
                Variants::Branches(branches)
            }
        }
    }

    // TODO?
    // #[test]
    // fn gen_variants_depth_0() {
    //     assert_eq!(
    //         Variants::Leaves(vec![]),
    //         gen_variants(Board::default())
    //     );
    // }

    fn ordering_from_color(color: Color) -> Ordering {
        match color {
            Color::White => Ordering::Greater,
            Color::Black => Ordering::Less,
        }
    }

    #[derive(Debug, Clone)]
    enum Variants {
        // TODO: try with HashSet/HashMap
        Branches(Vec<(Move, Variants)>),
        Leaves(Vec<MoveScoreWeight>),
    }
    impl Variants {
        pub(self) fn choose_best_move_by_minimax(self, color: Color) -> Option<Move> {
            self.choose_best_msw_by_minimax(color)
                .map(|msw| msw.move_)
        }
        fn choose_best_msw_by_minimax(self, color: Color) -> Option<MoveScoreWeight> {
            match self {
                Variants::Leaves(msws) => {
                    let best_is = ordering_from_color(color);
                    let mut omsw_best: Option<MoveScoreWeight> = None;
                    for msw_possible in msws {
                        omsw_best = match omsw_best {
                            None => { // executes at first cycle of the loop, when `omsw_best` isn't set yet
                                Some(msw_possible)
                            }
                            Some(/* msw_best @ */ MoveScoreWeight { score, .. }) if score.0.is_nan() => { // executes if best is NaN
                                Some(msw_possible)
                            }
                            Some(msw_best) => {
                                // assert!(msw_possible.score.is_finite()); // allowed to be ±infinite?
                                match msw_possible.get_weighted_score().partial_cmp(&msw_best.get_weighted_score()) {
                                    Some(ordering) if ordering == best_is => { // executes if new is better
                                        Some(msw_possible)
                                    }
                                    _ => omsw_best, // don't change best
                                }
                            }
                        };
                    }
                    omsw_best
                }
                Variants::Branches(branches) => {
                    branches
                        .into_iter()
                        .map(|(move_, branch)| (move_, branch.choose_best_msw_by_minimax(!color)))
                        .filter(|(_move, omsw)| omsw.is_some())
                        .map(|(move_, omsw)| (move_, omsw.unwrap()))
                        .max_by(|(_move1, msw1), (_move2, msw2)| {
                            let (msw1, msw2) = match ordering_from_color(color) {
                                Ordering::Greater => (msw1, msw2),
                                Ordering::Less => (msw2, msw1),
                                Ordering::Equal => unreachable!()
                            };
                            msw1.get_weighted_score()
                                .partial_cmp(&msw2.get_weighted_score())
                                .unwrap()
                        })
                        .map(|(move_, msw)| MoveScoreWeight { move_, ..msw }) // TODO: or maybe return `move_`?
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use chess::Square;
        use lazy_static::lazy_static;
        use super::*;

        // TODO(when https://github.com/jordanbray/chess/pull/84 gets accepted): remove `lazy_static`, rewrite to consts.
        lazy_static!{
            static ref E2E4: Move = Move::new(Square::E2, Square::E4, None);
            static ref E2E3: Move = Move::new(Square::E2, Square::E3, None);
            static ref E7E5: Move = Move::new(Square::E7, Square::E5, None);
            static ref E7E6: Move = Move::new(Square::E7, Square::E6, None);
            static ref G1F3: Move = Move::new(Square::G1, Square::F3, None);
            static ref D2D4: Move = Move::new(Square::D2, Square::D4, None);
        }

        #[test]
        fn choose_best_move_by_minimax_depth1_len1() {
            let msw_e2e4 = (*E2E4, Score(0.5)).into();
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Leaves(vec![
                    msw_e2e4,
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
        #[test]
        fn choose_best_move_by_minimax_depth1_len2_white() {
            let msw_e2e4 = (*E2E4, Score(0.5)).into();
            let msw_e2e3 = (*E2E3, Score(-0.5)).into();
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Leaves(vec![
                    msw_e2e4,
                    msw_e2e3,
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
        #[test]
        fn choose_best_move_by_minimax_depth1_len2_black() {
            let msw_e2e4 = (*E2E4, Score(0.5)).into();
            let msw_e2e3 = (*E2E3, Score(-0.5)).into();
            assert_eq!(
                Some(E2E3.to_string()),
                Variants::Leaves(vec![
                    msw_e2e4,
                    msw_e2e3,
                ]).choose_best_move_by_minimax(Color::Black).map(|move_| move_.to_string())
            );
        }

        #[test]
        fn choose_best_move_by_minimax_depth2_dduu() {
            let msw_e7e5 = (*E7E5, Score(0.)).into();
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Branches(vec![
                    (*E2E4, Variants::Leaves(vec![
                        msw_e7e5,
                    ])),
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
        #[test]
        fn choose_best_move_by_minimax_depth2_dduduu() {
            let msw_e7e5 = (*E7E5, Score(0.)).into();
            let msw_e7e6 = (*E7E6, Score(0.5)).into();
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Branches(vec![
                    (*E2E4, Variants::Leaves(vec![
                        msw_e7e5,
                        msw_e7e6,
                    ])),
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
        #[test]
        fn choose_best_move_by_minimax_depth2_dduudduu() {
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Branches(vec![
                    (*E2E4, Variants::Leaves(vec![
                        (*E7E5, Score(0.)).into(),
                    ])),
                    (*E2E3, Variants::Leaves(vec![
                        (*E7E5, Score(-1.)).into(),
                    ])),
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
        #[test]
        fn choose_best_move_by_minimax_depth2_dduduudduduu() {
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Branches(vec![
                    (*E2E4, Variants::Leaves(vec![
                        (*E7E5, Score(0.)).into(),
                        (*E7E6, Score(0.5)).into(),
                    ])),
                    (*E2E3, Variants::Leaves(vec![
                        (*E7E5, Score(-1.)).into(),
                        (*E7E6, Score(0.)).into(),
                    ])),
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }

        #[test]
        fn choose_best_move_by_minimax_depth3() {
            assert_eq!(
                Some(E2E4.to_string()),
                Variants::Branches(vec![
                    (*E2E4, Variants::Branches(vec![
                        (*E7E5, Variants::Leaves(vec![
                            (*G1F3, Score(0.2)).into(),
                            (*D2D4, Score(-0.2)).into(),
                        ])),
                        (*E7E6, Variants::Leaves(vec![
                            (*G1F3, Score(0.1)).into(),
                            (*D2D4, Score(0.3)).into(),
                        ])),
                    ])),
                    (*E2E3, Variants::Branches(vec![
                        (*E7E5, Variants::Leaves(vec![
                            (*G1F3, Score(-0.4)).into(),
                            (*D2D4, Score(0.1)).into(),
                        ])),
                        (*E7E6, Variants::Leaves(vec![
                            (*G1F3, Score(0.1)).into(),
                            (*D2D4, Score(0.)).into(),
                        ])),
                    ])),
                ]).choose_best_move_by_minimax(Color::White).map(|move_| move_.to_string())
            );
        }
    }
}

