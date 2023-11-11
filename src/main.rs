//! Main file

#![feature(
    // array_chunks,
    // array_windows,
    // adt_const_params,
    const_trait_impl,
    file_create_new,
    iter_map_windows,
    let_chains,
    slice_group_by,
)]


use std::{
    cmp::Ordering,
    collections::HashMap,
    str::FromStr,
};

use chess::{
    Action,
    Board,
    BoardBuilder,
    ChessMove,
    Color,
    File,
    Game,
    GameResult,
    MoveGen,
    Piece,
    Rank,
    Square,
};
use math_functions::exp;
use rand::{Rng, prelude::ThreadRng, seq::SliceRandom, thread_rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

mod extensions;
mod float_type;
mod linalg_types;
mod math_functions;
mod neural_network;
mod utils_io;

use crate::{
    float_type::float,
    linalg_types::RowVector,
    neural_network::{ChessNeuralNetwork, layers::*},
    utils_io::*,
};



const FILENAME_ALL_DATA: &str = "positions_evaluated_2023-11-11_00-18-58";
const FILENAME_ONLY_POSITIONS: &str = "positions";

const TRAIN_TO_TEST_RATIO: float = 0.9;
/// Starting learning rate, it will gradually decrease with epochs.
const LEARNING_RATE_0: float = 0.01;
const TRAINING_EPOCHS: u64 = 1_000;

const DEFAULT_RATING: float = 1000.0;
const TOURNAMENTS_NUMBER: u32 = 100;

// const EVO_GENERATIONS: u32 = 100;
const EVO_GEN_TO_START_WATCHING: u32 = 300;

const ALLOW_WIN_BY_POINTS: bool = true;
const MOVES_LIMIT: u32 = 200;

/// additional neuron for choosing move a bit random
const USE_NOISE: bool = true;
const NOISE_RANGE: (float, float) = (-1., 1.);
const NN_INPUT_SIZE: usize = if !USE_NOISE { 64 } else { 65 };

const SHOW_EVO_LOGS: bool = false;
const PLAY_WITH_NN_AFTER_TRAINING: bool = true;



fn main() {
    // let fens = generate_random_positions(10_000);
    // save_random_positions(fens);
    // return;

    let mut ais: Vec<AI> = vec![
        AI::new(
            "weighted sum",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 1>::new()),
            ]),
        ),

        AI::new(
            "FC-Tanh 100-50-20-10-5-3",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFTanh::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 50>::new()),
                Box::new(activation_functions::AFTanh::<50>::new()),
                Box::new(fc_layer::FCLayer::<50, 20>::new()),
                Box::new(activation_functions::AFTanh::<20>::new()),
                Box::new(fc_layer::FCLayer::<20, 10>::new()),
                Box::new(activation_functions::AFTanh::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(activation_functions::AFTanh::<5>::new()),
                Box::new(fc_layer::FCLayer::<5, 3>::new()),
                Box::new(activation_functions::AFTanh::<3>::new()),
                Box::new(fc_layer::FCLayer::<3, 1>::new()),
            ]),
        ),
        AI::new( // CRAZY BUG??: breaks on `LEARNING_RATE_0` = 0.1?!?
            "FC-Tanh 100-10-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFTanh::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(activation_functions::AFTanh::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 1>::new()),
            ]),
        ),
        AI::new(
            "FC-Tanh 100-10-5-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFTanh::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(activation_functions::AFTanh::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(activation_functions::AFTanh::<5>::new()),
                Box::new(fc_layer::FCLayer::<5, 1>::new()),
            ]),
        ),

        AI::new(
            "FC-Sigmoid 100-50-20-10-5-3",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFSigmoid::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 50>::new()),
                Box::new(activation_functions::AFSigmoid::<50>::new()),
                Box::new(fc_layer::FCLayer::<50, 20>::new()),
                Box::new(activation_functions::AFSigmoid::<20>::new()),
                Box::new(fc_layer::FCLayer::<20, 10>::new()),
                Box::new(activation_functions::AFSigmoid::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(activation_functions::AFSigmoid::<5>::new()),
                Box::new(fc_layer::FCLayer::<5, 3>::new()),
                Box::new(activation_functions::AFSigmoid::<3>::new()),
                Box::new(fc_layer::FCLayer::<3, 1>::new()),
            ]),
        ),
        AI::new(
            "FC-Sigmoid 100-10-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFSigmoid::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(activation_functions::AFSigmoid::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 1>::new()),
            ]),
        ),
        AI::new(
            "FC-Sigmoid 100-10-5-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(activation_functions::AFSigmoid::<100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(activation_functions::AFSigmoid::<10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(activation_functions::AFSigmoid::<5>::new()),
                Box::new(fc_layer::FCLayer::<5, 1>::new()),
            ]),
        ),

        AI::new(
            "FC 100-50-20-10-5-3",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(fc_layer::FCLayer::<100, 50>::new()),
                Box::new(fc_layer::FCLayer::<50, 20>::new()),
                Box::new(fc_layer::FCLayer::<20, 10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(fc_layer::FCLayer::<5, 3>::new()),
                Box::new(fc_layer::FCLayer::<3, 1>::new()),
            ]),
        ),
        AI::new(
            "FC 100-10-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(fc_layer::FCLayer::<10, 1>::new()),
            ]),
        ),
        AI::new(
            "FC 100-10-5-1",
            ChessNeuralNetwork::new(vec![
                Box::new(fc_layer::FCLayer::<NN_INPUT_SIZE, 100>::new()),
                Box::new(fc_layer::FCLayer::<100, 10>::new()),
                Box::new(fc_layer::FCLayer::<10, 5>::new()),
                Box::new(fc_layer::FCLayer::<5, 1>::new()),
            ]),
        ),
    ];
    // assert!(ais.len() > 1, "ais.len should be > 1, else its useless, but it is {}", ais.len());

    let mut rng = thread_rng();
    ais.shuffle(&mut rng);

    let all_data = load_all_data_str();
    let all_data = AnyData::from(all_data);
    let train_and_test_data = TrainAndTestData::from(all_data, TRAIN_TO_TEST_RATIO);

    // let ais_len = ais.len();
    pretrain_nns(
        &mut ais[..],
        &train_and_test_data,
    );

    // let players_number: usize = players.len();

    // let mut rng: ThreadRng = thread_rng();
    // let mut players_old: Vec<Player>;
    // let mut new_best_same_counter: u32 = 0;

    // for generation in 0..=EVO_GENERATIONS {
    //     println!("generation: {generation} / {EVO_GENERATIONS}");

    //     players_old = players.clone();

    //     play_tournament(&mut players, generation);

    //     if generation < EVO_GENERATIONS {
    //         fn generation_to_evolve_factor(gen: u32, gens: u32) -> float {
    //             // ( -(gen as float) / (gens as float).sqrt() ).exp()
    //             // ( -(gen as float) / (gens as float).powf(0.8) ).exp()
    //             // ( -(gen as float) / (gens as float) ).exp()
    //             // ( - 3.0 * (gen as float) / (gens as float) ).exp()
    //             // 0.3 * ( -(gen as float) / (gens as float) ).exp()
    //             0.999 * ( -(gen as float) / (gens as float) ).exp()
    //             // 0.8 * ( - 5.0 * (gen as float) / (gens as float) ).exp()
    //             // 0.1 * ( -(gen as float) / (gens as float) ).exp()
    //             // 0.1 * ( - 3.0 * (gen as float) / (gens as float) ).exp()
    //         }
    //         let evolution_factor: float = generation_to_evolve_factor(generation, EVO_GENERATIONS);
    //         if SHOW_TRAINING_LOGS {
    //             println!("evolving with evolution_factor = {}%", 100.0*evolution_factor);
    //             // let total_neurons: u64 =
    //             //     NN_INPUT_SIZE as u64
    //             //     + nn_heights.iter().map(|&h| h as u64).sum::<u64>()
    //             //     + 1;
    //             // let approx_neurons_to_evolve: float = evolution_factor * (total_neurons as float);
    //             // println!("approx neurons_to_evolve = {approx_neurons_to_evolve}");
    //         }

    //         // first part is best nns so dont evolve them, but second part will be evolved
    //         let save_best_n: usize = 1 + players_number / 3;
    //         for i in save_best_n..players_number {
    //             nns[i] = nns[i % save_best_n].clone();
    //             nns[i].evolve(evolution_factor, &mut rng);
    //         }
    //         let nns_len = nns.len();
    //         // nns[len-2] = NeuralNetwork::with_consts(&nn_heights, 0.01, 0.0, get_random_activation_function());
    //         // nns[len-1] = NeuralNetwork::with_smart_random(&nn_heights);
    //         nns[nns_len-1] = ChessNeuralNetwork::new();
    //     }

    //     if nns_old[0] == nns[0] && generation > 0 {
    //         new_best_same_counter += 1;
    //         if SHOW_TRAINING_LOGS {
    //             println!("WARNING: new best is same {new_best_same_counter} times!!!");
    //         }
    //     }
    //     else {
    //         new_best_same_counter = 0;
    //     }

    //     if SHOW_TRAINING_LOGS {
    //         println!("\n");
    //     }
    // }
    // println!("Evolution finished successfuly!");
    // println!("best_nn = {}\n\n", nns[0]);

    println!("Playing {TOURNAMENTS_NUMBER} tournaments to set ratings...");
    let mut ais = ais;
    for i in 1..TOURNAMENTS_NUMBER {
        print!("{i} "); flush();
        play_tournament(&mut ais, 0, false);
    }
    println!("\nPlaying last tournament to show some games...");
    play_tournament(&mut ais, 0, true);
    let ais = ais;
    println!("AIs' ratings after {TOURNAMENTS_NUMBER} tournaments:");
    for (i, ai) in ais.iter().enumerate() {
        println!("#{i}: {r:.2} - {n}", i=i+1, r=ai.rating, n=ai.name);
    }

    if !PLAY_WITH_NN_AFTER_TRAINING { return }

    loop {
        print_and_flush("\nChoose NeuralNetwork to play with (`best`, `worst`, index or name): ");
        let line = read_line();
        enum NeuralNetworkToPlayWith {
            Best,
            Worst,
            Index(usize),
            Name(String),
        }
        type NNTPW = NeuralNetworkToPlayWith;
        let nn_to_play_with: NNTPW = match line.trim() {
            "q" => { break }
            "best" => NNTPW::Best,
            "worst" => NNTPW::Worst,
            text => if let Ok(n) = text.parse::<usize>() {
                NNTPW::Index(n-1)
            } else {
                let name = text.to_string();
                NNTPW::Name(name)
            }
        };
        let ai_to_play_with: &AI = match nn_to_play_with {
            NNTPW::Best => &ais.first().unwrap(),
            NNTPW::Worst => &ais.last().unwrap(),
            NNTPW::Index(index) => if let Some(ai) = ais.get(index) { &ai } else { continue }
            NNTPW::Name(name) => if let Some(ai) = ais.iter().find(|ai| ai.name == name) { &ai } else { continue }
        };
        let nn_to_play_with: &ChessNeuralNetwork = &ai_to_play_with.nn;
        print_and_flush("Choose side to play (`w` or `b`, `q` to quit): ");
        let line = read_line();
        let side_to_play: Color = match line.chars().nth(0) {
            Some('w') => Color::White,
            Some('b') => Color::Black,
            Some('q') => { break }
            _ => { continue } // ask everything from start again
        };
        let (winner, game_moves): (Option<WhoWon>, Option<String>) = play_game(
            &nn_to_play_with,
            &nn_to_play_with,
            PlayGameConfig {
                get_game_moves: true,
                show_logs: true,
                wait_for_enter_after_every_move: false,
                human_color: Some(side_to_play),
            }
        );
        let winner = if let Some(winner) = winner { winner } else { continue };
        println!(
            "{who_vs_who}: winner={winner:?}, moves: ' {moves} '\n",
            who_vs_who = match side_to_play {
                Color::White => "HUMAN vs NN_BEST",
                Color::Black => "NN_BEST vs HUMAN",
            },
            moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
        );
    }
}


fn pretrain_nns(nns: &mut [AI], train_and_test_data: &TrainAndTestData) {
    let TrainAndTestData { train_data, test_data } = train_and_test_data;
    let mut rng = thread_rng();
    for epoch in 0..TRAINING_EPOCHS {
        let learning_rate = learning_rate_from_epoch(epoch);
        let mut train_data_shuffled: AnyData = train_data.clone();
        train_data_shuffled.xy.shuffle(&mut rng);
        let mut msg = format!("epoch {ep}/{TRAINING_EPOCHS}:\n", ep=epoch+1);
        msg += &nns
            // .iter_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(i, ai)| {
                let avg_train_error = pretrain_step(&mut ai.nn, &train_data_shuffled, learning_rate);
                let avg_test_error = calc_avg_test_error(&ai.nn, test_data);
                format!("NN#{i}\tavg_train_error = {avg_train_error}\tavg_test_error = {avg_test_error}\t{name}", name=ai.name)
            })
            .collect::<Vec<String>>()
            .join("\n");
        println!("{msg}\n");
    }
}

/// Returns `total_error`.
fn pretrain_step(nn: &mut ChessNeuralNetwork, train_data_shuffled: &AnyData, learning_rate: float) -> float {
    let mut total_error = 0.;
    for (x, y) in train_data_shuffled.xy.iter() {
        // println!("{}", "\n".repeat(10));
        // println!("x: {x:?}");
        // println!("y: {y:?}");
        let output = nn.process_input_for_training(x.clone());
        // println!("output: {output:?}");
        total_error += nn.loss(output, *y);
        let error = nn.loss_prime(output, *y);
        let mut error = RowVector::from_element(1, error);
        for layer in nn.layers.iter_mut().rev() {
            // println!("error: {error}");
            error = layer.backward_propagation(error, learning_rate);
        }
        // panic!();
    }
    let avg_error = total_error / (train_data_shuffled.xy.len() as float);
    avg_error
}

fn calc_avg_test_error(nn: &ChessNeuralNetwork, test_data: &AnyData) -> float {
    let mut total_error = 0.;
    for (x, y) in test_data.xy.iter() {
        let output = nn.process_input(x.clone());
        total_error += nn.loss(output, *y);
    }
    let avg_error = total_error / (test_data.xy.len() as float);
    avg_error
}

fn learning_rate_from_epoch(epoch: u64) -> float {
    const E: float = TRAINING_EPOCHS as float;
    let e = epoch as float;
    // exp( -e / sqrt(E) )
    // exp( -e / E.powf(0.8) )
    // exp( -e / E )
    // exp( - 3.0 * e / E )
    // 0.3 * exp( -e / E )
    // 0.999 * exp( -e / E )
    // 0.8 * exp( - 5.0 * e / E )
    // 0.1 * exp( -e / E )
    // 0.1 * exp( - 3.0 * e / E )
    LEARNING_RATE_0 * exp( -5.*e/E )
}
#[test]
fn learning_rate_from_epoch_0() {
    assert_eq!(LEARNING_RATE_0, learning_rate_from_epoch(0));
}
#[test]
fn learning_rate_from_epoch_last() {
    assert!(learning_rate_from_epoch(TRAINING_EPOCHS) < LEARNING_RATE_0);
}
#[test]
fn learning_rate_from_epoch_all() {
    (0..TRAINING_EPOCHS)
        .into_iter()
        .map(learning_rate_from_epoch)
        .map_windows(|&[lr_prev, lr_this]| [lr_prev, lr_this])
        // .map_windows(identity::<&[float; 2]>)
        .for_each(|[lr_prev, lr_this]| {
            assert!(lr_this < lr_prev);
        });
}


#[derive(Debug, Clone, PartialEq)]
struct TrainAndTestData {
    train_data: AnyData,
    test_data: AnyData,
}

impl TrainAndTestData {
    fn from(all_data: AnyData, split_k: float) -> Self {
        assert!(0. <= split_k && split_k <= 1.);
        let split_at = (split_k * (all_data.xy.len() as float)) as usize;
        let (train_data_vec, test_data_vec) = all_data.xy.split_at(split_at);
        Self {
            train_data: AnyData { xy: train_data_vec.to_vec() },
            test_data : AnyData { xy: test_data_vec .to_vec() },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AnyData {
    xy: Vec<(RowVector, float)>,
}

impl AnyData {
    fn from(all_data_str: Vec<String>) -> Self {
        let mut rng = thread_rng();
        Self {
            xy: all_data_str
                .into_iter()
                .map(|line| {
                    let (score_str, position_str) = line.trim().split_once(' ').unwrap();
                    (score_str.to_string(), position_str.to_string())
                })
                .map(|(score_str, position_str)| (position_vec_from_string(&position_str, &mut rng), score_from_string(&score_str)))
                .collect()
        }
    }
}

fn score_from_string(score_str: &str) -> float {
    score_str
        .parse::<float>()
        .unwrap_or_else(|_e| {
            assert_eq!('#', score_str.chars().next().unwrap(), "score_str = {score_str}");
            let mate_in_str = &score_str[1..];
            let mate_in = mate_in_str.parse::<i8>().unwrap();
            let score = (mate_in.signum() as float) * match mate_in.abs() {
                0 => unreachable!(),
                1 => 200.,
                2 => 190.,
                3 => 180.,
                4 => 170.,
                5 => 160.,
                6 => 150.,
                7 => 140.,
                8 => 130.,
                9 => 120.,
                10=> 110.,
                _ => 100.
            };
            score
        })
}

fn position_vec_from_string(position_str: &str, rng: &mut ThreadRng) -> RowVector {
    board_to_vector_for_nn(&Board::from_str(position_str).unwrap(), rng)
}


fn generate_random_positions(fens_number: usize) -> Vec<String> {
    let moves_limit = 100;
    let mut fens = Vec::with_capacity(fens_number);
    let mut rng = thread_rng();
    while fens.len() < fens_number {
        let mut game = Game::new();
        for _move_number in 0..2*moves_limit {
            let possible_moves = MoveGen::new_legal(&game.current_position());
            let possible_moves = possible_moves.into_iter().collect::<Vec<_>>();
            let random_move = possible_moves[rng.gen_range(0..possible_moves.len())];
            game.make_move(random_move);
            if game.result().is_some() { break }
            fens.push(game.current_position().to_string());
            // if game.can_declare_draw() { break }
        }
    }
    fens
}

fn save_random_positions(fens: Vec<String>) {
    use std::io::Write;
    let mut file = ::std::fs::File::create_new(FILENAME_ONLY_POSITIONS).unwrap();
    for fen in fens {
        writeln!(file, "{fen}").unwrap();
    }
}

fn load_all_data_str() -> Vec<String> {
    use ::std::io::BufRead;
    let file = ::std::fs::File::open(FILENAME_ALL_DATA).unwrap();
    ::std::io::BufReader::new(file)
        .lines()
        .map(|line| line.unwrap())
        .collect()
}



const MOVES_NOT_PROVIDED: &str = "moves not provided";

mod chess_pieces {
    use chess::{Color, Piece};

    pub const NONE: char = '.';

    pub const PAWN_WHITE  : char = 'P';
    pub const KNIGHT_WHITE: char = 'N';
    pub const BISHOP_WHITE: char = 'B';
    pub const ROOK_WHITE  : char = 'R';
    pub const QUEEN_WHITE : char = 'Q';
    pub const KING_WHITE  : char = 'K';

    pub const PAWN_BLACK  : char = 'p';
    pub const KNIGHT_BLACK: char = 'n';
    pub const BISHOP_BLACK: char = 'b';
    pub const ROOK_BLACK  : char = 'r';
    pub const QUEEN_BLACK : char = 'q';
    pub const KING_BLACK  : char = 'k';

    pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> char {
        let piece_and_color = if let Some(piece_and_color) = option_piece_and_color { piece_and_color } else { return NONE };
        match piece_and_color {
            (Piece::Pawn  , Color::White) => PAWN_WHITE,
            (Piece::Knight, Color::White) => KNIGHT_WHITE,
            (Piece::Bishop, Color::White) => BISHOP_WHITE,
            (Piece::Rook  , Color::White) => ROOK_WHITE,
            (Piece::Queen , Color::White) => QUEEN_WHITE,
            (Piece::King  , Color::White) => KING_WHITE,

            (Piece::Pawn  , Color::Black) => PAWN_BLACK,
            (Piece::Knight, Color::Black) => KNIGHT_BLACK,
            (Piece::Bishop, Color::Black) => BISHOP_BLACK,
            (Piece::Rook  , Color::Black) => ROOK_BLACK,
            (Piece::Queen , Color::Black) => QUEEN_BLACK,
            (Piece::King  , Color::Black) => KING_BLACK,
        }
    }
}

mod chess_pieces_beautiful {
    use chess::{Color, Piece};

    pub const NONE: char = '.';

    /*
        ♖♘♗♕♔♗♘♖
        ♙♙♙♙♙♙♙♙
        ♟♟♟♟♟♟♟♟
        ♜♞♝♛♚♝♞♜
    */

    pub const PAWN_WHITE  : char = '♟';
    pub const KNIGHT_WHITE: char = '♞';
    pub const BISHOP_WHITE: char = '♝';
    pub const ROOK_WHITE  : char = '♜';
    pub const QUEEN_WHITE : char = '♛';
    pub const KING_WHITE  : char = '♚';

    pub const PAWN_BLACK  : char = '♙';
    pub const KNIGHT_BLACK: char = '♘';
    pub const BISHOP_BLACK: char = '♗';
    pub const ROOK_BLACK  : char = '♖';
    pub const QUEEN_BLACK : char = '♕';
    pub const KING_BLACK  : char = '♔';

    pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> char {
        let piece_and_color = if let Some(piece_and_color) = option_piece_and_color { piece_and_color } else { return NONE };
        match piece_and_color {
            (Piece::Pawn  , Color::White) => PAWN_WHITE,
            (Piece::Knight, Color::White) => KNIGHT_WHITE,
            (Piece::Bishop, Color::White) => BISHOP_WHITE,
            (Piece::Rook  , Color::White) => ROOK_WHITE,
            (Piece::Queen , Color::White) => QUEEN_WHITE,
            (Piece::King  , Color::White) => KING_WHITE,

            (Piece::Pawn  , Color::Black) => PAWN_BLACK,
            (Piece::Knight, Color::Black) => KNIGHT_BLACK,
            (Piece::Bishop, Color::Black) => BISHOP_BLACK,
            (Piece::Rook  , Color::Black) => ROOK_BLACK,
            (Piece::Queen , Color::Black) => QUEEN_BLACK,
            (Piece::King  , Color::Black) => KING_BLACK,
        }
    }
}

fn get_all_pieces(board: &Board) -> Vec<(Piece, Color)> {
    let board_builder: BoardBuilder = board.into();
    (0..64)
        .filter_map(|i| {
            let square = unsafe { Square::new(i) }; // SAFETY: this is safe bc `i` is from 0 to 64 (not including)
            board_builder[square]
        })
        .collect()
}

/// Returns unsorted `(white_pieces, black_pieces)`
struct PiecesByColor { white_pieces: Vec<Piece>, black_pieces: Vec<Piece> }
fn get_pieces_by_color(board: &Board) -> PiecesByColor {
    let board_builder: BoardBuilder = board.into();
    let mut white_pieces = Vec::new();
    let mut black_pieces = Vec::new();
    for i in 0..64 {
        let square = unsafe { Square::new(i) }; // SAFETY: this is safe bc `i` is from 0 to 64 (not including)
        let option_piece_and_color = board_builder[square];
        match option_piece_and_color {
            Some((piece, Color::White)) => { white_pieces.push(piece) }
            Some((piece, Color::Black)) => { black_pieces.push(piece) }
            _ => {}
        }
    }
    PiecesByColor { white_pieces, black_pieces }
}

fn get_pieces_diff(board: &Board) -> PiecesByColor {
    let PiecesByColor { mut white_pieces, mut black_pieces } = get_pieces_by_color(board);
    white_pieces.sort();
    black_pieces.sort();
    fn pieces_to_some_pieces(pieces: Vec<Piece>) -> Vec<Option<Piece>> {
        pieces
            .into_iter()
            .map(|piece| Some(piece))
            .collect()
    }
    let mut white_pieces: Vec<Option<Piece>> = pieces_to_some_pieces(white_pieces);
    let mut black_pieces: Vec<Option<Piece>> = pieces_to_some_pieces(black_pieces);
    let mut i = 0;
    let mut j = 0;
    while i < white_pieces.len() || j < black_pieces.len() {
        // println!("white_pieces = {white_pieces:?}");
        // println!("black_pieces = {black_pieces:?}");
        // dbg!(i, j);
        let white_piece = &mut white_pieces[i];
        let black_piece = &mut black_pieces[j];
        match white_piece.cmp(&black_piece) {
            Ordering::Equal => {
                *white_piece = None;       
                *black_piece = None;       
                i += 1;
                j += 1;
            }
            Ordering::Greater => { j += 1 }
            Ordering::Less => { i += 1 }
        }
    }
    // println!("white_pieces = {white_pieces:?}");
    // println!("black_pieces = {black_pieces:?}");
    let mut white_pieces: Vec<Piece> = white_pieces.into_iter().flatten().collect();
    let mut black_pieces: Vec<Piece> = black_pieces.into_iter().flatten().collect();
    // println!("white_pieces = {white_pieces:?}");
    // println!("black_pieces = {black_pieces:?}");
    white_pieces.shrink_to_fit();
    black_pieces.shrink_to_fit();
    PiecesByColor { white_pieces, black_pieces }
}

fn get_pieces_diff_str(board: &Board) -> String {
    let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff(board);
    // let white_pieces: Vec<Piece> = white_pieces.into_iter().map(|(piece, color)| piece).collect();
    // let black_pieces: Vec<Piece> = black_pieces.into_iter().map(|(piece, color)| piece).collect();
    fn get_pieces_str(pieces: Vec<Piece>, color: Color) -> String {
        pieces
            .into_iter()
            .map(|piece| chess_pieces_beautiful::get(Some((piece, color))).to_string())
            .collect::<Vec<String>>()
            .join(" ")
    }
    let white_pieces_str: String = get_pieces_str(white_pieces, Color::White);
    let black_pieces_str: String = get_pieces_str(black_pieces, Color::Black);
    // let white_pieces_str = "abcef";
    // let black_pieces_str = "def";
    format_left_right::<19, 2>(&white_pieces_str, &black_pieces_str)
}

fn format_left_right<const LEN: usize, const MIN_SPACES_IN_BETWEEN: usize>(left: &str, right: &str) -> String {
    let spaces = " ".repeat(MIN_SPACES_IN_BETWEEN);
    let shift = LEN.saturating_sub(MIN_SPACES_IN_BETWEEN).saturating_sub(left.len());
    format!("{left}{spaces}{right:>shift$}")
}

struct BoardToHumanViewableConfig { beautiful_output: bool, show_files_ranks: bool, show_pieces_diff: bool }
fn board_to_human_viewable(board: &Board, config: BoardToHumanViewableConfig) -> String {
    const FILES: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const RANKS: [&str; 8] = ["1", "2", "3", "4", "5", "6", "7", "8"];
    let x_line: String = format!("  {}", FILES.join(" "));
    let approx_capacity: usize = if config.show_files_ranks { 250 } else { 200 }; // 64*2 +? 16*4
    let mut res: String = String::with_capacity(approx_capacity);
    if config.show_pieces_diff {
        res += &get_pieces_diff_str(board);
        res += "\n";
    }
    if config.show_files_ranks {
        res += &x_line;
        res += "\n";
    }
    let board_builder: BoardBuilder = board.into();
    for y in (0..8).rev() {
        if y != 7 {
            res += "\n";
        }
        for x in 0..8 {
            let index = y*8 + x;
            let square = unsafe { Square::new(index) }; // SAFETY: this is safe bc `index` is from 0 to 64 (not including)
            let option_piece_and_color = board_builder[square];
            if x == 0 {
                res += &format!("{} ", RANKS[y as usize]);
            }
            res += &if config.beautiful_output {
                chess_pieces_beautiful::get(option_piece_and_color)
            } else {
                chess_pieces::get(option_piece_and_color)
            }.to_string();
            res += &" ".to_string();
            if x == 7 {
                res += &RANKS[y as usize];
            }
        }
    }
    if config.show_files_ranks {
        res += "\n";
        res += &x_line;
    }
    res.shrink_to_fit();
    res
}



#[allow(non_snake_case)]
mod pieces_values {
    use chess::{Color, Piece};

    use crate::float_type::float;

    pub const NONE: float = 0.0;

    // pub const PAWN  : float = 1.0;
    // pub const KNIGHT: float = 2.7;
    // pub const BISHOP: float = 3.0;
    // pub const ROOK  : float = 5.0;
    // pub const QUEEN : float = 7.0;
    // pub const KING  : float = 15.0;
    pub const PAWN  : float = 0.1;
    pub const KNIGHT: float = 0.2;
    pub const BISHOP: float = 0.4;
    pub const ROOK  : float = 0.6;
    pub const QUEEN : float = 0.8;
    pub const KING  : float = 1.0;

    pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> float {
        let (piece, color): (Piece, Color) = if let Some(piece_and_color) = option_piece_and_color { piece_and_color } else { return NONE };
        let value = match piece {
            Piece::Pawn   => PAWN,
            Piece::Knight => KNIGHT,
            Piece::Bishop => BISHOP,
            Piece::Rook   => ROOK,
            Piece::Queen  => QUEEN,
            Piece::King   => KING,
        };
        match color {
            Color::White => value,
            Color::Black => -value,
        }
    }
}


fn board_to_vector_for_nn(board: &Board, rng: &mut ThreadRng) -> RowVector {
    fn board_to_vector(board: &Board, rng: &mut ThreadRng) -> RowVector {
        let mut vector: RowVector = RowVector::zeros(NN_INPUT_SIZE);
        let board_builder: BoardBuilder = board.into();
        for i in 0..64 {
            let square: Square = unsafe { Square::new(i) }; // SAFETY: this is safe bc `i` is from 0 to 64 (not including)
            let option_piece_and_color: Option<(Piece, Color)> = board_builder[square];
            vector[i as usize] = pieces_values::get(option_piece_and_color);
        }
        if USE_NOISE {
            vector[64] = rng.gen_range(NOISE_RANGE.0 .. NOISE_RANGE.1);
        }
        vector
    }
    let mut input_for_nn = board_to_vector(board, rng);
    // input_for_nn[64] = match board.side_to_move() {
    //     Color::White => { 1.0 }
    //     Color::Black => { -1.0 }
    // };
    if board.side_to_move() == Color::Black {
        // TODO
        input_for_nn = RowVector::from_iterator(NN_INPUT_SIZE, input_for_nn.into_iter().rev().map(|&x| x));
        input_for_nn = -input_for_nn;
    }
    // println!("{input_for_nn:?}");
    input_for_nn
}

fn analyze(board: &Board, nn: &ChessNeuralNetwork, rng: &mut ThreadRng) -> float {
    let input_for_nn = board_to_vector_for_nn(board, rng);
    // println!("input_for_nn = {:?}", array_board);
    nn.process_input(input_for_nn)
}


/// Return `pieces_sum_white - pieces_sum_black`
fn board_to_pieces_sum(board: &Board) -> float {
    let board_builder: BoardBuilder = board.into();
    let mut pieces_sum = 0.;
    for i in 0..64 {
        let square = unsafe { Square::new(i) }; // SAFETY: this is safe bc `i` is from 0 to 64 (not including)
        let option_piece_and_color = board_builder[square];
        pieces_sum += pieces_values::get(option_piece_and_color);
    }
    pieces_sum
}




#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
enum WhoWon {
    White,
    Black,
    Draw,
    WhiteByPoints,
    BlackByPoints,
    DrawByPoints,
}

fn actions_to_string(actions: Vec<Action>) -> String {
    const ACCEPT_DRAW: &str = "accept_draw";
    const DECLARE_DRAW: &str = "declare_draw";
    actions
        .into_iter()
        .map(|action| match action {
            Action::MakeMove(chess_move) => chess_move.to_string(),
            Action::OfferDraw(color) => format!("offer_draw_{color:?}"),
            Action::AcceptDraw => ACCEPT_DRAW.to_string(),
            Action::DeclareDraw => DECLARE_DRAW.to_string(),
            Action::Resign(color) => format!("resign_{color:?}"),
        })
        // .collect::<Vec<String>>()
        // .join(" ")
        // .reduce(|acc, el| acc + " " + &el)
        // .unwrap()
        .fold(String::new(), |acc, el| acc + " " + &el)
}



fn string_to_chess_move(line: &str) -> Option<ChessMove> {
    if !(4..=5).contains(&line.len()) { return None; }
    let chars: Vec<char> = line.chars().collect();
    let (from_file, from_rank, to_file, to_rank, promote_to) = (chars[0], chars[1], chars[2], chars[3], chars.get(4));
    let from_file = File::from_str(&from_file.to_string()).ok()?;
    let from_rank = Rank::from_str(&from_rank.to_string()).ok()?;
    let to_file = File::from_str(&to_file.to_string()).ok()?;
    let to_rank = Rank::from_str(&to_rank.to_string()).ok()?;
    if let Some(&promote_to) = promote_to && !"qrbn".contains(promote_to) { return None; }
    let promote_to: Option<Piece> = promote_to.map(|ch| match ch {
        'q' => { Piece::Queen }
        'r' => { Piece::Rook }
        'b' => { Piece::Bishop }
        'n' => { Piece::Knight }
        _ => unreachable!()
    });
    Some(ChessMove::new(
        Square::make_square(from_rank, from_file),
        Square::make_square(to_rank, to_file),
        promote_to,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveByHuman {
    Move(ChessMove),
    Surrender,
    Exit,
}
fn get_move_from_human(game: &Game) -> MoveByHuman {
    loop {
        print_and_flush("Your move: ");
        let line = read_line();
        let line = line.trim();
        match line {
            "q" => { return MoveByHuman::Exit }
            "s" => { return MoveByHuman::Surrender }
            _ => {}
        }
        let move_ = string_to_chess_move(line);
        if let Some(move_) = move_ && game.current_position().legal(move_) {
            return MoveByHuman::Move(move_);
        }
        println!("That move is illegal.");
    }
}


struct PlayGameConfig {
    pub get_game_moves: bool,
    pub show_logs: bool,
    pub wait_for_enter_after_every_move: bool,
    pub human_color: Option<Color>,
}

fn play_game(
    nn_white: &ChessNeuralNetwork,
    nn_black: &ChessNeuralNetwork,
    config: PlayGameConfig,
) -> (Option<WhoWon>, Option<String>) {
    #[derive(Debug, Copy, Clone)]
    struct MoveWithMark {
        pub chess_move: ChessMove,
        pub mark: float,
    }

    // let mut game = Game::new_with_board(Board::from_fen(FEN_INIT_POSITION.to_string()).unwrap());
    let mut game = Game::new();

    if config.show_logs {
        // println!("{}", game.current_position());
        println!(
            "{}",
            board_to_human_viewable(
                &game.current_position(),
                BoardToHumanViewableConfig {
                    beautiful_output: true,
                    show_files_ranks: true,
                    show_pieces_diff: true,
                }
            )
        );
    }

    let mut rng = thread_rng();

    let mut move_number: u32 = 0;
    while game.result() == None && move_number < 2*MOVES_LIMIT {
        move_number += 1;
        // if config.show_logs {
        //     println!("move_number = {move_number}");
        // }

        let side_to_move: Color = game.current_position().side_to_move();

        // if vs human
        if let Some(human_color) = config.human_color && human_color == side_to_move {
            let move_by_human = get_move_from_human(&game);
            let move_ = match move_by_human {
                MoveByHuman::Exit => { return (None, None) }
                MoveByHuman::Surrender => {
                    game.resign(human_color);
                    continue
                }
                MoveByHuman::Move(move_) => move_,
            };
            game.make_move(move_);
            continue // go to AI's move
        }

        let nn_to_make_move = match side_to_move {
            Color::White => nn_white,
            Color::Black => nn_black,
        };

        let mut omwm_best: Option<MoveWithMark> = None;
        let mut mwms: Vec<MoveWithMark> = vec![]; // used only if `config.show_logs`
        let legal_moves = MoveGen::new_legal(&game.current_position());
        for move_ in legal_moves {
            let board_possible: Board = game.current_position().make_move_new(move_);

            // println!("{}", board_to_string_with_fen(board_possible, false));

            let mark_possible: float = analyze(
                &board_possible,
                nn_to_make_move,
                &mut rng,
            );
            let mwm_possible = MoveWithMark{ chess_move: move_, mark: mark_possible };
            if config.show_logs {
                // println!("{} -> {}", mwm_possible.chess_move, mwm_possible.mark);
                mwms.push(mwm_possible);
            }

            match omwm_best {
                Some(mwm_best) => {
                    if mwm_possible.mark.total_cmp(&mwm_best.mark) == Ordering::Greater {
                        omwm_best = Some(mwm_possible);
                    }
                }
                None => { // executes only at first cycle of the loop
                    omwm_best = Some(mwm_possible);
                }
            }
        }

        if config.show_logs {
            // dbg!(&mwms);
            mwms.sort_by(|mwm1, mwm2| mwm1.mark.total_cmp(&mwm2.mark));
            for mwm in mwms {
                println!("{move_} -> {mark:.4}", move_=mwm.chess_move.to_string(), mark=mwm.mark);
            }
        }

        match omwm_best {
            Some(mwm_best) => {
                // board_now = make_move(board_now, mwm_best.chess_move);
                // println!("{}", game.current_position());

                if config.show_logs {
                    println!("making move: {}", mwm_best.chess_move);
                }
                game.make_move(mwm_best.chess_move);

                // println!("{}", game.current_position());
            }
            None => {
                if config.show_logs {
                    println!("game have ended, because no move can be made, i suppose");
                }
                // println!("{game:?}");
                break
            }
        }

        if game.can_declare_draw() {
            if ALLOW_WIN_BY_POINTS {
                // move_number = 2*MOVES_LIMIT;
                break
            }
            else {
                game.declare_draw();
            }
        }

        if config.show_logs {
            println!(
                "{}",
                board_to_human_viewable(
                    &game.current_position(),
                    BoardToHumanViewableConfig {
                        beautiful_output: true,
                        show_files_ranks: true,
                        show_pieces_diff: true,
                    }
                )
            );
            // println!("{}", game.current_position());
            // println!("{:?}", game);
            if config.wait_for_enter_after_every_move {
                wait_for_enter();
            }
        }
    }

    let create_game_str_if_needed = || {
        if config.get_game_moves {
            Some(actions_to_string(game.actions().to_vec()))
        } else {
            None
        }
    };

    // dbg!(move_number, MOVES_LIMIT);

    // TODO: refactor/fix `if`
    if move_number <= 2*MOVES_LIMIT && !ALLOW_WIN_BY_POINTS { // true victory/lose:
        let game_res: GameResult = game.result().unwrap();
        if config.show_logs {
            println!("game result: {:?}", game_res);
        }
        type GR = GameResult;
        let winner = match game_res {
            GR::WhiteCheckmates | GR::BlackResigns => WhoWon::White,
            GR::WhiteResigns | GR::BlackCheckmates => WhoWon::Black,
            GR::Stalemate | GR::DrawAccepted | GR::DrawDeclared => WhoWon::Draw,
        };
        (Some(winner), create_game_str_if_needed())
    }
    else { // by points:
        if config.show_logs {
            println!("game result: true draw or {}+ moves", 2*MOVES_LIMIT);
            println!("so winner will be calculated by pieces");
        }
        let pieces_sum = board_to_pieces_sum(&game.current_position());
        let winner = pieces_sum.partial_cmp(&0.)
            .map(|cmp| match cmp {
                Ordering::Greater => WhoWon::WhiteByPoints,
                Ordering::Less    => WhoWon::BlackByPoints,
                Ordering::Equal   => WhoWon::DrawByPoints,
            });
        (winner, create_game_str_if_needed())
    }
}



#[derive(Clone)]
struct AI {
    pub name: &'static str,
    pub nn: ChessNeuralNetwork,
    pub rating: float,
}

impl AI {
    fn new(name: &'static str, nn: ChessNeuralNetwork) -> Self {
        Self {
            name,
            nn,
            rating: DEFAULT_RATING,
        }
    }
}

fn logistic(x: float) -> float {
    100. / ( 1. + (10. as float).powf(x / 400.) )
}

/// Plays tournament and sorts AIs.
fn play_tournament(ais: &mut Vec<AI>, gen: u32, print_games_at_tournament: bool) {
    let ais_number = ais.len();

    let mut tournament_statistics: HashMap<WhoWon, u32> = HashMap::new();

    // can be used, to verify that indices are correct
    // #[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
    // struct NNGameResult {
    //     i: usize,
    //     j: usize,
    //     winner: WhoWon,
    // }

    let nn_game_results: Vec<Option<WhoWon>> = (0..ais_number.pow(2)) // ij
        .collect::<Vec<usize>>()
        .into_par_iter()
        .map(|ij| (ij / ais_number, ij % ais_number))
        .map(|(i, j)| if i != j { Some((ais[i].nn.clone(), ais[j].nn.clone())) } else { None })
        .map(|ai_ij| {
            ai_ij.map(|(ai_i, ai_j)| {
                let game_res: (Option<WhoWon>, Option<String>) = play_game(
                    &ai_i,
                    &ai_j,
                    PlayGameConfig {
                        get_game_moves: false,
                        show_logs: false,
                        wait_for_enter_after_every_move: false,
                        human_color: None,
                    }
                );
                game_res.0.unwrap()
            })
        })
        .collect();

    for i in 0..ais_number {
        for j in 0..ais_number {
            if i == j { continue }
            // i->w->white, j->b->black
            //let mut player_w: Player = players[i].clone();
            //let mut player_b: Player = players[j].clone();

            // can be used, to verify that indices are correct
            //let nn_game_result: NNGameResult = nn_game_results[i*player_number+j].unwrap();
            //let game_res_winner: WhoWon = nn_game_result.winner;
            let game_res_winner: WhoWon = nn_game_results[i*ais_number+j].unwrap();

            // if white wins
            let delta_rating_w: float = logistic(ais[i].rating - ais[j].rating);
            // if black wins
            let delta_rating_b: float = logistic(ais[j].rating - ais[i].rating);

            let counter = tournament_statistics.entry(game_res_winner).or_insert(0);
            *counter += 1;

            match game_res_winner {
                WhoWon::White => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("W");
                    }
                    ais[i].rating += delta_rating_w;
                    // players[j].rating -= delta_rating_w;
                    ais[j].rating -= delta_rating_w / 5.0;
                }
                WhoWon::Black => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("B");
                    }
                    // players[i].rating -= delta_rating_b;
                    ais[i].rating -= delta_rating_b / 5.0;
                    ais[j].rating += delta_rating_b;
                }
                WhoWon::Draw => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("D");
                    }
                    let delta_rating_min: float = delta_rating_w.min(delta_rating_b);
                    if ais[i].rating > ais[j].rating {
                        ais[i].rating -= delta_rating_min / 3.0;
                        ais[j].rating += delta_rating_min / 3.0;
                        // player_i.rating -= 1.0;
                        // player_j.rating += 1.0;
                    }
                    else if ais[j].rating > ais[i].rating {
                        ais[i].rating += delta_rating_min / 3.0;
                        ais[j].rating -= delta_rating_min / 3.0;
                        // player_i.rating += 1.0;
                        // player_j.rating -= 1.0;
                    }
                    else { // equal
                        // nothing
                    }
                }
                WhoWon::WhiteByPoints => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("w");
                    }
                    ais[i].rating += delta_rating_w / 20.0;
                    ais[j].rating -= delta_rating_w / 20.0;
                    // player_i.rating += 3.0;
                    // player_j.rating -= 3.0;
                }
                WhoWon::BlackByPoints => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("b");
                    }
                    ais[i].rating -= delta_rating_b / 20.0;
                    ais[j].rating += delta_rating_b / 20.0;
                    // player_i.rating -= 3.0;
                    // player_j.rating += 3.0;
                }
                WhoWon::DrawByPoints => {
                    if SHOW_EVO_LOGS {
                        print_and_flush("d");
                    }
                    if ais[i].rating > ais[j].rating {
                        ais[i].rating -= delta_rating_w / 20.0;
                        ais[j].rating += delta_rating_w / 20.0;
                        // player_i.rating -= 0.3;
                        // player_j.rating += 0.3;
                    }
                    else if ais[j].rating > ais[i].rating {
                        ais[i].rating += delta_rating_w / 20.0;
                        ais[j].rating -= delta_rating_w / 20.0;
                        // player_i.rating += 0.3;
                        // player_j.rating -= 0.3;
                    }
                    else { // equal
                        // nothing
                    }
                }
            }
            // if SHOW_TRAINING_LOGS {
            //     println!("new ratings: i={}, j={}", player_i.rating, player_j.rating);
            //     println!();
            // }

            // players[i].rating = player_w.rating;
            // players[j].rating = player_b.rating;
        }
        if SHOW_EVO_LOGS {
            print!(" ");
        }
    }
    if SHOW_EVO_LOGS {
        println!();
    }

    // sort AIs:
    ais.sort_by(|ai1, ai2| ai2.rating.partial_cmp(&ai1.rating).unwrap());

    if print_games_at_tournament {
        println!("\nstats: {:?}", tournament_statistics);

        let ratings_sorted: Vec<float> = ais.iter().map(|p| p.rating).collect();
        print!("final ratings (sorted): [");
        for i in 0..ratings_sorted.len() {
            print!("{r:.0}", r=ratings_sorted[i]);
            if i != ratings_sorted.len()-1 {
                print!(", ");
            }
        }
        println!("]\n");

        {
            let (winner, game_moves) = play_game(
                &ais[0].nn,
                &ais[0].nn,
                PlayGameConfig {
                    get_game_moves: true,
                    show_logs: gen >= EVO_GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                }
            );
            println!(
                "BEST vs SELF: winner={winner:?}, moves: ' {moves} '\n",
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }

        {
            let (winner, game_moves) = play_game(
                &ais[0].nn,
                &ais[1].nn,
                PlayGameConfig {
                    get_game_moves: true,
                    show_logs: gen >= EVO_GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                }
            );
            println!(
                "BEST vs BEST2: winner={winner:?}, moves: ' {moves} '\n",
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }

        {
            let (winner, game_moves) = play_game(
                &ais[0].nn,
                &ais.last().unwrap().nn,
                PlayGameConfig {
                    get_game_moves: true,
                    show_logs: gen >= EVO_GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                }
            );
            println!(
                "BEST vs WORST: winner={winner:?}, moves: ' {moves} '\n",
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }

        {
            let (winner, game_moves) = play_game(
                &ais.last().unwrap().nn,
                &ais.last().unwrap().nn,
                PlayGameConfig {
                    get_game_moves: true,
                    show_logs: gen >= EVO_GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                }
            );
            println!(
                "WORST vs WORST: winner={winner:?}, moves: ' {moves} '\n",
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }
    }
}

