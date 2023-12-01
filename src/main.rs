//! Main file

#![feature(
    // array_chunks,
    // array_windows,
    // adt_const_params,
    // const_trait_impl,
    file_create_new,
    get_many_mut,
    iter_map_windows,
    let_chains,
    lint_reasons,
    // negative_bounds,
    // negative_impls,
    never_type,
    // slice_group_by,
    test,
)]

#![deny(
    dead_code,
    unreachable_patterns,
    // unused, // TODO: enable
)]


use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::Hash,
    str::FromStr,
};

use chess::{ALL_SQUARES, Action, Board, BoardBuilder, ChessMove, Color, Game, GameResult, MoveGen, Piece};
use image::{ImageBuffer, RgbImage, ImageResult, Rgb};
use linalg_types::Matrix;
use rand::{Rng, seq::SliceRandom, thread_rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

// mod either;
mod float_type;
mod linalg_types;
mod math_aliases;
mod math_functions;
// mod math_functions_pade_approx;
mod neural_network_row;
mod players;
mod utils_io;

use crate::{
    float_type::float,
    math_aliases::exp,
    neural_network_row::{ChessNeuralNetwork, layers::LayerSpecs as LS, vector_type::Vector},
    players::{
        BoxDynPlayer,
        MaybeChessMove,
        Player,
        ai::{
            AIwithRating,
            ai::{AI, AI_ThinkingDepth},
            generator::AIsGenerator
        },
        human::Human,
        rating::Rating,
        rating::update_ratings,
    },
    utils_io::{print_and_flush, prompt, wait_for_enter},
};



// TODO(refactor): separate consts into local(here) modules

const FILENAMES_ALL_DATA: &[&str] = &[
    // "positions/lt_part1_evaluated_2023-11-11_00-18-58",
    // "positions/lt_part2_evaluated_2023-11-12_12-48-37",
    // "positions/lt_part3_evaluated_2023-11-13_09-46-52",
    // "positions/lt_part4_evaluated_2023-11-13_19-26-50",
    // "positions/lt_part5_evaluated_2023-11-16_21-16-25",
    // "positions/pc_part1_evaluated_2023-11-12_15-17-11",
    // "positions/pc_part2_evaluated_2023-11-13_09-35-55",
    // "positions/pc_part3_evaluated_2023-11-13_14-34-22",
    // "positions/pc_part4_evaluated_2023-11-13_18-45-15",
    // "positions/pc_part5_evaluated_2023-11-13_19-27-46",
    "positions/pc_part6_evaluated_2023-11-14_09-01-36",
    // "positions/pc_part7_evaluated_2023-11-14_13-26-40",
    // "positions/pc_part8_evaluated_2023-11-15_09-36-39",
    // "positions/pc_part9_evaluated_2023-11-16_15-59-12",
    // "positions/pc_part10_evaluated_2023-11-16_18-54-26",
    // "positions/pc_part11_evaluated_2023-11-17_07-51-10",
];
const FILENAME_TO_SAVE_POSITIONS: &str = "positions/lt_or_pc_partN";

mod fully_connected_layer_initial_values {
    use crate::float_type::float;
    pub const W_MIN: float = -1.;
    pub const W_MAX: float =  1.;
    pub const S_MIN: float = -1.;
    pub const S_MAX: float =  1.;
}

// const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Two;
// const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Three { use_opposite_signs: false };
const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Four;
const NUMBER_OF_DIFFERENT_CHESS_PIECES: usize = chess::NUM_PIECES; // 6
const NUMBER_OF_SQUARES_ON_CHESS_BOARD: usize = chess::NUM_SQUARES; // 64
const NN_INPUT_SIZE_PER_COLOR_CHANNEL: usize = NUMBER_OF_DIFFERENT_CHESS_PIECES * NUMBER_OF_SQUARES_ON_CHESS_BOARD; // 384
const NN_INPUT_SIZE: usize = NUMBER_OF_DEPTH_CHANNELS.discriminant() * NN_INPUT_SIZE_PER_COLOR_CHANNEL; // 768 or 1152 or 1536
const NN_OUTPUT_SIZE: usize = 1;

mod ais_generator_consts {
    use crate::{
        players::ai::generator::{ActivationFunctions, LayersNumber},
        float_type::float,
    };
    pub const MULTI_AF_PROB: float = 0.5;
    pub const ACTIVATION_FUNCTIONS: ActivationFunctions = ActivationFunctions::All;
    pub const LAYERS_NUMBER: LayersNumber = LayersNumber::Range { min: 2, max: 7 };
    pub const LAYERS_SIZES: &[usize] = &[200, 150, 100, 80, 60, 50, 35, 20, 10, 5, 3, 2];
}

const NUMBER_OF_NNS: usize = 1 * 12; // it's better be multiple of number of cores/threads on your machine, or else...

const TRAIN_TO_TEST_RATIO: float = 0.9;

/// Starting learning rate, gradually decreases with epochs.
const LEARNING_RATE_0: float = 0.1;
const LEARNING_RATE_EXP_K: float = 2.;
const TRAINING_EPOCHS: usize = 2;
// TODO(feat): const for use depth analysis when training?
const CHESS_NN_THINK_DEPTH_FOR_TRAINING: u8 = 1;

const TOURNAMENTS_NUMBER: usize = 1;
const DEFAULT_RATING: float = 1_000.;
const CHESS_NN_THINK_DEPTH_IN_TOURNAMENT: u8 = 1;
const NN_RESULT_RANDOM_CHOICE: Option<(float, float)> = Some((0.9, 1.1));
const PLAY_GAME_MOVES_LIMIT: usize = 500;

const CHESS_NN_THINK_DEPTH_VS_HUMAN: u8 = 3; // 4 if parallel



fn main() {
    // generate_and_save_random_positions(10_000, 200);
    // return;

    print_and_flush("Creating Neural Networks... ");
    let mut ai_players: Vec<AIwithRating> = vec![
        AIwithRating::new(AI::new_for_training(
            "Weighted Sum".to_string(),
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(NN_OUTPUT_SIZE),
            ]),
        )),
    ];
    ai_players.extend(
        AIsGenerator::default()
            .generate(NUMBER_OF_NNS)
            .into_iter()
            .map(|ai| AIwithRating::new(ai))
            .collect::<Vec<AIwithRating>>()
    );
    println!("{} created.", ai_players.len());
    assert!(ai_players.len() > 1, "number of AIs should be > 1, else its not interesting, but it is {}", ai_players.len());

    // let mut rng = thread_rng();
    // ais.shuffle(&mut rng);

    print_and_flush(format!("Loading data from {} files... ", FILENAMES_ALL_DATA.len()));
    let all_data = load_all_data_str(FILENAMES_ALL_DATA);
    println!("{} lines loaded.", all_data.len());

    print_and_flush("Loading data... ");
    let all_data = AnyData::from(all_data);
    println!("{} samples loaded.", all_data.xy.len());

    print_and_flush(format!("Splitting data to train and test datasets with `train/test ratio`={TRAIN_TO_TEST_RATIO}... "));
    let train_and_test_data = TrainAndTestData::from(all_data, TRAIN_TO_TEST_RATIO);
    println!("Done.");

    fn set_ais_mode(ai_players: &mut Vec<AIwithRating>, mode: AI_ThinkingDepth) {
        for ai_player in ai_players.iter_mut() {
            ai_player.get_ai_mut().set_mode(mode);
        }
    }

    println!("Starting training Neural Networks...\n");
    set_ais_mode(&mut ai_players, AI_ThinkingDepth::Training);
    // TODO?: if NN returns NaN => recreate it few times, else delete it
    train_nns(
        &mut ai_players,
        train_and_test_data,
        TrainNNsConfig {
            remove_ai_if_it_gives_nan: true,
        }
    );

    println!("Playing {TOURNAMENTS_NUMBER} tournaments to set ratings...");
    set_ais_mode(&mut ai_players, AI_ThinkingDepth::Tournament);
    for i in 0..TOURNAMENTS_NUMBER {
        let n = index_to_number(i);
        print_and_flush(format!("#{n}/{TOURNAMENTS_NUMBER}:\t"));
        play_tournament(
            &mut ai_players,
            PlayTournametConfig {
                sort: n == TOURNAMENTS_NUMBER,
                print_games_results: true,
                print_games: n == TOURNAMENTS_NUMBER,
            }
        );
    }
    fn print_ais_ratings(ai_players: &Vec<AIwithRating>, is_first_time: bool) {
        let maybe_after_n_tournaments_str = if is_first_time { format!(" after {TOURNAMENTS_NUMBER} tournaments") } else { "".to_string() };
        println!();
        println!("AIs' ratings{maybe_after_n_tournaments_str}:");
        for (i, ai_player) in ai_players.iter().enumerate() {
            let n = index_to_number(i);
            let rating = ai_player.get_rating().get();
            let name = ai_player.get_ai().get_name();
            println!("#{n}: {rating:.2} - {name}");
        }
    }
    print_ais_ratings(&ai_players, true);

    // return;

    set_ais_mode(&mut ai_players, AI_ThinkingDepth::VsHuman);
    let ai_players = ai_players;

    // TODO?: assert AIs is sorted by rating.

    loop {
        println!("\nWhat do you want to do?");
        const CMD_BEST_STR : &str = "best";
        const CMD_WORST_STR: &str = "worst";
        println!("- play with Neural Network: `{CMD_BEST_STR}`, `{CMD_WORST_STR}`, index or name");
        const CMD_LIST_SHORT: &str = "l";
        const CMD_LIST_FULL : &str = "list";
        println!("- list neutal networks: `{CMD_LIST_SHORT}` or `{CMD_LIST_FULL}`");
        const CMD_QUIT_SHORT: &str = "q";
        const CMD_QUIT_FULL : &str = "quit";
        println!("- quit: `{CMD_QUIT_SHORT}` or `{CMD_QUIT_FULL}`");
        const CMD_EXPORT_NN_AS_IMAGES_SHORT: &str = "e";
        const CMD_EXPORT_NN_AS_IMAGES_FULL : &str = "export";
        println!("- export NN as images: `{CMD_EXPORT_NN_AS_IMAGES_SHORT}` or `{CMD_EXPORT_NN_AS_IMAGES_FULL}`");
        // TODO(feat): make AI #N play vs AI #M
        let line = prompt("So what's it gonna be, huuh? ");
        enum NeuralNetworkToPlayWith {
            Best,
            Worst,
            Index(usize),
            Name(String),
        }
        type NNTPW = NeuralNetworkToPlayWith;
        let ai_to_play_with: NNTPW = match line.as_str() {
            CMD_QUIT_SHORT | CMD_QUIT_FULL => { break }
            CMD_LIST_SHORT | CMD_LIST_FULL => { print_ais_ratings(&ai_players, false); continue }
            CMD_EXPORT_NN_AS_IMAGES_SHORT | CMD_EXPORT_NN_AS_IMAGES_FULL => { process_export_nn_as_images(&ai_players); continue }
            CMD_BEST_STR => NNTPW::Best,
            CMD_WORST_STR => NNTPW::Worst,
            text => if let Ok(n) = text.parse::<usize>() {
                let Some(i) = number_to_index_checked(n) else { continue };
                NNTPW::Index(i)
            } else {
                let name = text.to_string();
                NNTPW::Name(name)
            }
        };
        let ai_to_play_with: &AIwithRating = match ai_to_play_with {
            NNTPW::Best => ai_players.first().unwrap(),
            NNTPW::Worst => ai_players.last().unwrap(),
            NNTPW::Index(index) => if let Some(ai) = ai_players.get(index) { ai } else { continue }
            NNTPW::Name(name) => if let Some(ai) = ai_players.iter().find(|aip| aip.get_ai().get_name() == name) { ai } else { continue }
        };
        let ai_to_play_with: BoxDynPlayer = Box::new(ai_to_play_with.get_ai());
        println!("Choose side to play:");
        const CMD_SIDE_TO_PLAY_WHITE_SHORT: &str = "w";
        const CMD_SIDE_TO_PLAY_WHITE_FULL : &str = "white";
        println!("- white: `{CMD_SIDE_TO_PLAY_WHITE_SHORT}` or `{CMD_SIDE_TO_PLAY_WHITE_FULL}`");
        const CMD_SIDE_TO_PLAY_BLACK_SHORT: &str = "b";
        const CMD_SIDE_TO_PLAY_BLACK_FULL : &str = "black";
        println!("- black: `{CMD_SIDE_TO_PLAY_BLACK_SHORT}` or `{CMD_SIDE_TO_PLAY_BLACK_FULL}`");
        println!("- return back: anything else");
        let line = prompt("Choose wisely. ");
        let human_side_to_play: Color = match line.as_str() {
            CMD_SIDE_TO_PLAY_WHITE_SHORT | CMD_SIDE_TO_PLAY_WHITE_FULL => Color::White,
            CMD_SIDE_TO_PLAY_BLACK_SHORT | CMD_SIDE_TO_PLAY_BLACK_FULL => Color::Black,
            _ => { continue } // ask everything from start again
        };
        println!("Good luck! In any unclear situation use `s` to surrender or `q` to quit.");
        let (player_white, player_black): (BoxDynPlayer, BoxDynPlayer) = match human_side_to_play {
            Color::White => (Box::new(&Human), ai_to_play_with),
            Color::Black => (ai_to_play_with, Box::new(&Human)),
        };
        let config = PlayGameConfig {
            wait_for_enter_after_every_move: false,
            ..PlayGameConfig::all()
        };
        // TODO(enhancement): add "Thinking..." when AI is thinking.
        let (winner, game_moves) = play_game(player_white, player_black, config);
        let Ok(winner) = winner else { continue };
        println!(
            "{who_vs_who}: winner={winner:?}, moves: ' {moves} '\n",
            who_vs_who = match human_side_to_play {
                Color::White => "HUMAN vs NN_BEST",
                Color::Black => "NN_BEST vs HUMAN",
            },
            moves = game_moves.unwrap_or(MOVES_WASNT_PROVIDED.to_string()),
        );
    }
}


struct TrainNNsConfig { remove_ai_if_it_gives_nan: bool }
fn train_nns(ai_players: &mut Vec<AIwithRating>, train_and_test_data: TrainAndTestData, config: TrainNNsConfig) {
    let TrainAndTestData { mut train_data, mut test_data } = train_and_test_data;
    let mut rng = thread_rng();
    for epoch in 0..TRAINING_EPOCHS {
        let learning_rate = learning_rate_from_epoch(epoch);
        train_data.xy.shuffle(&mut rng);
        test_data.xy.shuffle(&mut rng);
        let (is_to_remove_vec, msg_parts): (Vec<bool>, Vec<String>) = ai_players
            // .iter_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(i, ai_player)| {
                let avg_train_error = train_step(ai_player.get_ai_mut().get_nn_mut(), &train_data, learning_rate);
                let avg_test_error = calc_avg_test_error(&ai_player.get_ai().get_nn(), &test_data);
                let is_to_remove = avg_train_error.is_nan() || avg_test_error.is_nan();
                let n = index_to_number(i);
                let name = ai_player.get_ai().get_name();
                (is_to_remove, format!("NN#{n}\tavg train error = {avg_train_error}\tavg test error = {avg_test_error}\t{name}"))
                // TODO(optimization): also return time spent training to then sort them by it for efficiency
            })
            .unzip();
        let mut msg = format!("Epoch {epoch_number}/{TRAINING_EPOCHS}:\n", epoch_number=index_to_number(epoch));
        msg += &msg_parts.join("\n");
        println!("{msg}\n");
        if config.remove_ai_if_it_gives_nan {
            assert_eq!(ai_players.len(), is_to_remove_vec.len());
            let mut is_to_remove_iter = is_to_remove_vec.iter();
            ai_players.retain(|_| !*is_to_remove_iter.next().unwrap());
        }
    }
}

/// Makes one train step & returns `total_error`.
fn train_step(nn: &mut ChessNeuralNetwork, train_data_shuffled: &AnyData, learning_rate: float) -> float {
    let mut total_error = 0.;
    for (x, y) in train_data_shuffled.xy.iter() {
        // println!("{}", "\n".repeat(10));
        // println!("x: {x:?}");
        // println!("y: {y:?}");
        let output = nn.process_input_for_training(x.clone());
        // println!("output: {output:?}");
        total_error += nn.loss(output, *y);
        if total_error.is_nan() { return float::NAN }
        let error = nn.loss_prime(output, *y);
        let mut error = Vector::from_element(1, error);
        for layer in nn.layers.iter_mut().rev() {
            // println!("error: {error}");
            error = layer.backward_propagation(error, learning_rate);
        }
        // panic!();
    }
    let avg_error = total_error / (train_data_shuffled.xy.len() as float);
    avg_error.sqrt()
}

fn calc_avg_test_error(nn: &ChessNeuralNetwork, test_data: &AnyData) -> float {
    let mut total_error = 0.;
    for (x, y) in test_data.xy.iter() {
        let output = nn.process_input(x.clone());
        total_error += nn.loss(output, *y);
        if total_error.is_nan() { return float::NAN }
    }
    let avg_error = total_error / (test_data.xy.len() as float);
    avg_error.sqrt()
}

fn learning_rate_from_epoch(epoch: usize) -> float {
    const E: float = TRAINING_EPOCHS as float;
    let e = epoch as float;
    // exp( -e / sqrt(E) )
    // exp( -e / E.powf(0.8) )
    LEARNING_RATE_0 * exp( -LEARNING_RATE_EXP_K * e / E )
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


#[derive(Debug)]
struct TrainAndTestData {
    train_data: AnyData,
    test_data: AnyData,
}

impl TrainAndTestData {
    fn from(mut all_data: AnyData, split_k: float) -> Self {
        assert!(0. <= split_k && split_k <= 1.);
        let mut rng = thread_rng();
        all_data.xy.shuffle(&mut rng);
        let split_at = (split_k * (all_data.xy.len() as float)) as usize;
        let (train_data, test_data) = all_data.xy.split_at(split_at);
        let mut train_data = train_data.to_vec();
        let mut test_data = test_data.to_vec();
        train_data.shrink_to_fit();
        test_data.shrink_to_fit();
        Self {
            train_data: AnyData { xy: train_data },
            test_data : AnyData { xy: test_data  },
        }
    }
}

#[derive(Debug, Clone)]
struct AnyData {
    xy: Vec<(Vector, float)>,
}

impl AnyData {
    fn from(all_data_str: Vec<String>) -> Self {
        let mut xy: Vec<(Vector, float)> = all_data_str
            .into_iter()
            // .into_par_iter() // no need, it's fast enough
            .map(|line| {
                let (score_str, position_str) = line.trim().split_once(' ').unwrap();
                (position_vec_from_string(position_str), score_from_string(score_str))
            })
            .collect();
        xy.shrink_to_fit();
        Self { xy }
    }
}

fn score_from_string(score_str: &str) -> float {
    score_str
        .parse::<float>()
        .unwrap_or_else(|_e| {
            assert_eq!('#', score_str.chars().next().unwrap(), "score_str = {score_str}");
            let mate_in_str = &score_str[1..];
            let mate_in = mate_in_str.parse::<i8>().unwrap();
            // TODO(think): what is the best way to convert mates in N moves into score?
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

fn position_vec_from_string(position_str: &str) -> Vector {
    board_to_vector_for_nn(Board::from_str(position_str).unwrap())
}


#[expect(dead_code)]
fn generate_and_save_random_positions(fens_number: usize, moves_limit: usize) {
    let fens = generate_random_positions(fens_number, moves_limit);
    save_random_positions(fens);
}

fn generate_random_positions(fens_number: usize, moves_limit: usize) -> Vec<String> {
    let mut fens = Vec::with_capacity(fens_number);
    while fens.len() < fens_number {
        let mut game = Game::new();
        for _move_number in 0..moves_limit {
            let random_move = select_random_move(&game.current_position());
            game.make_move(random_move);
            if game.result().is_some() || game.can_declare_draw() { break }
            fens.push(game.current_position().to_string());
        }
    }
    fens
}

pub fn select_random_move(board: &Board) -> ChessMove {
    let moves = MoveGen::new_legal(board);
    let moves = moves.into_iter().collect::<Vec<_>>();
    let random_move_index = thread_rng().gen_range(0..moves.len());
    let random_move = moves[random_move_index];
    random_move
}

fn save_random_positions(fens: Vec<String>) {
    use std::{fs::File, io::Write};
    let mut file = File::create_new(FILENAME_TO_SAVE_POSITIONS).unwrap();
    for fen in fens {
        writeln!(file, "{fen}").unwrap();
    }
}

fn load_all_data_str(filenames: &[&str]) -> Vec<String> {
    use std::{fs::File, io::BufRead};
    let mut all_data_str = filenames
        .into_iter()
        .flat_map(|filename| {
            let file = File::open(filename).unwrap();
            ::std::io::BufReader::new(file)
                .lines()
                .map(|line| line.unwrap())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<String>>();
    all_data_str.shrink_to_fit();
    all_data_str
}



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
        let Some(piece_and_color) = option_piece_and_color else { return NONE };
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

    /* ♚♛♜♝♞♟ ♔♕♖♗♘♙ */

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
        let Some(piece_and_color) = option_piece_and_color else { return NONE };
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

#[derive(Debug, PartialEq)]
struct PiecesByColor<T> { white_pieces: T, black_pieces: T }
/// Returns unsorted `(white_pieces, black_pieces)`
fn get_pieces_by_color(board: Board) -> PiecesByColor<Vec<Piece>> {
    let board_builder: BoardBuilder = board.into();
    let mut white_pieces = Vec::new();
    let mut black_pieces = Vec::new();
    for square in ALL_SQUARES {
        let option_piece_and_color = board_builder[square];
        match option_piece_and_color {
            Some((piece, Color::White)) => { white_pieces.push(piece) }
            Some((piece, Color::Black)) => { black_pieces.push(piece) }
            _ => {}
        }
    }
    PiecesByColor { white_pieces, black_pieces }
}

/// Returns `{white_pieces: Vec<Piece>, black_pieces: Vec<Piece>}`,
/// where `white_pieces` - pieces that white have,
/// and black dont, and `black_pieces` - vice versa.
fn get_pieces_diff_have(board: Board) -> PiecesByColor<Vec<Piece>> {
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
    let mut white_pieces: Vec<Piece> = white_pieces.into_iter().flatten().collect();
    let mut black_pieces: Vec<Piece> = black_pieces.into_iter().flatten().collect();
    white_pieces.reverse();
    black_pieces.reverse();
    white_pieces.shrink_to_fit();
    black_pieces.shrink_to_fit();
    PiecesByColor { white_pieces, black_pieces }
}

/// Returns `{white_pieces: Vec<Piece>, black_pieces: Vec<Piece>}`,
/// inversed to [`get_pieces_diff_have`].
fn get_pieces_diff_lost(board: Board) -> PiecesByColor<Vec<Piece>> {
    let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff_have(board);
    PiecesByColor { white_pieces: black_pieces, black_pieces: white_pieces }
}

// /// Returns `{white_pieces: String, black_pieces: String}`,
// /// where `white_pieces` - pieces that white have,
// /// and black dont, and `black_pieces` - vice versa.
// fn get_pieces_diff_have_str(board: Board) -> PiecesByColor<String> {
//     let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff_have(board);
//     PiecesByColor {
//         white_pieces: vec_piece_to_string(white_pieces, Color::White),
//         black_pieces: vec_piece_to_string(black_pieces, Color::Black),
//     }
// }

#[derive(Clone, Copy)]
struct VecPieceToStringConfig { separator: Option<&'static str>, is_beautiful: bool }
fn vec_piece_to_string(pieces: Vec<Piece>, color: Color, config: VecPieceToStringConfig) -> String {
    pieces
        .into_iter()
        .map(|piece| {
            let option_piece_and_color = Some((piece, color));
            let get_chess_piece = if config.is_beautiful { chess_pieces_beautiful::get } else { chess_pieces::get };
            get_chess_piece(option_piece_and_color).to_string()
        })
        .reduce(|acc, el| acc + config.separator.unwrap_or_default() + &el)
        .unwrap_or_default()
}

fn get_pieces_diff_lost_str(board: Board, vec_piece_to_string_config: VecPieceToStringConfig) -> PiecesByColor<String> {
    let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff_lost(board);
    PiecesByColor {
        white_pieces: vec_piece_to_string(white_pieces, Color::White, vec_piece_to_string_config),
        black_pieces: vec_piece_to_string(black_pieces, Color::Black, vec_piece_to_string_config),
    }
}
#[test]
fn get_pieces_diff_lost_str_() {
    /* ♚♛♜♝♞♟ ♔♕♖♗♘♙ */
    let expected = PiecesByColor {
        white_pieces: "♟ ♟ ♟ ♟".to_string(),
        black_pieces: "♕ ♖ ♗ ♘".to_string()
    };
    let actual = get_pieces_diff_lost_str(
        Board::from_str("r1b1k1n1/pppppppp/8/8/8/8/P1P1P1P1/RNBQKBNR w KQq - 0 1").unwrap(),
        VecPieceToStringConfig { separator: Some(" "), is_beautiful: true },
    );
    // let expected_chars: Vec<char> = expected.chars().collect();
    // let actual_chars: Vec<char> = actual.chars().collect();
    assert_eq!(expected, actual);
}

// fn format_left_right<const LEN: usize, const MIN_SPACES_IN_BETWEEN: usize>(left: &str, right: &str) -> String {
//     let spaces = " ".repeat(MIN_SPACES_IN_BETWEEN);
//     let shift = LEN.saturating_sub(MIN_SPACES_IN_BETWEEN).saturating_sub(left.chars().count()); // `.chars().count()` used instead of `.len()`, bc `len` returns bytes, not chars.
//     format!("{left}{spaces}{right:>shift$}")
// }

struct BoardToHumanViewableConfig { beautiful_output: bool, show_files_ranks: bool, show_pieces_diff: bool }
impl BoardToHumanViewableConfig {
    fn all() -> Self {
        Self { beautiful_output: true, show_files_ranks: true, show_pieces_diff: true }
    }
}
fn board_to_human_viewable(board: Board, config: BoardToHumanViewableConfig) -> String {
    const FILES: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const RANKS: [&str; 8] = ["1", "2", "3", "4", "5", "6", "7", "8"];
    let x_line: String = format!("  {}", FILES.join(" "));
    let options_pieces_diff_lost_str = if config.show_pieces_diff {
        Some(get_pieces_diff_lost_str(board, VecPieceToStringConfig { separator: None, is_beautiful: true }))
    } else {
        None
    };
    let approx_capacity: usize = if config.show_files_ranks { 250 } else { 200 }; // 64*2 +? 16*4
    let mut res: String = String::with_capacity(approx_capacity);
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
            let square = ALL_SQUARES[index];
            let option_piece_and_color = board_builder[square];
            if x == 0 {
                res += RANKS[y as usize];
                res += " ";
            }
            res += &if config.beautiful_output {
                chess_pieces_beautiful::get(option_piece_and_color)
            } else {
                chess_pieces::get(option_piece_and_color)
            }.to_string();
            res += &" ".to_string();
            if x == 7 {
                res += &RANKS[y as usize];
                if config.show_pieces_diff {
                    match y {
                        0 => { res += " "; res += &options_pieces_diff_lost_str.as_ref().unwrap().white_pieces }
                        7 => { res += " "; res += &options_pieces_diff_lost_str.as_ref().unwrap().black_pieces }
                        _ => {}
                    }
                }
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



pub fn board_to_vector_for_nn(board: Board) -> Vector {
    let mut input_for_nn: Vector = Vector::zeros(NN_INPUT_SIZE);
    let board_builder: BoardBuilder = board.into();
    for (index_in_64, square) in ALL_SQUARES.into_iter().enumerate() {
        let option_piece_and_color: Option<(Piece, Color)> = board_builder[square];
        if let Some((piece, color)) = option_piece_and_color {
            // bow = white or black
            let index_of_64_wob = match (piece, color) {
                (Piece::Pawn  , Color::White) => 0,
                (Piece::Knight, Color::White) => 1,
                (Piece::Bishop, Color::White) => 2,
                (Piece::Rook  , Color::White) => 3,
                (Piece::Queen , Color::White) => 4,
                (Piece::King  , Color::White) => 5,
                (Piece::Pawn  , Color::Black) => 6,
                (Piece::Knight, Color::Black) => 7,
                (Piece::Bishop, Color::Black) => 8,
                (Piece::Rook  , Color::Black) => 9,
                (Piece::Queen , Color::Black) => 10,
                (Piece::King  , Color::Black) => 11,
            };
            // set white or black channel:
            input_for_nn[64*index_of_64_wob + index_in_64] = 1.;
            // TODO(refactor): extract this `1.` into const?

            fn get_index_of_64_wab(piece: Piece) -> usize {
                match piece {
                    Piece::Pawn   => 12,
                    Piece::Knight => 13,
                    Piece::Bishop => 14,
                    Piece::Rook   => 15,
                    Piece::Queen  => 16,
                    Piece::King   => 17,
                }
            }

            fn value_from_color(color: Color) -> float {
                match color {
                    Color::White => 1.,
                    Color::Black => -1.,
                }
            }

            match NUMBER_OF_DEPTH_CHANNELS {
                NumberOfDepthChannels::Two => {}
                NumberOfDepthChannels::Three { use_opposite_signs } => {
                    // wab = white and black
                    let index_of_64_wab = get_index_of_64_wab(piece);
                    let value = if !use_opposite_signs { 1. } else { value_from_color(color) };
                    // set white and black channel:
                    input_for_nn[64*index_of_64_wab + index_in_64] = value;
                }
                NumberOfDepthChannels::Four => {
                    // wab = white and black
                    let index_of_64_wab = get_index_of_64_wab(piece);
                    // set white and black channel:
                    input_for_nn[64*index_of_64_wab + index_in_64] = 1.;

                    // wanb = white and negative black
                    let index_of_64_wanb = match piece {
                        Piece::Pawn   => 18,
                        Piece::Knight => 19,
                        Piece::Bishop => 20,
                        Piece::Rook   => 21,
                        Piece::Queen  => 22,
                        Piece::King   => 23,
                    };
                    let value = value_from_color(color);
                    // set white and negative black channel:
                    input_for_nn[64*index_of_64_wanb + index_in_64] = value;
                }
            }
        }
    }
    input_for_nn
}


#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
enum Winner {
    White,
    Black,
    Draw,
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
        .reduce(|acc, el| acc + " " + &el)
        .unwrap_or_default()
        // .intersperse(" ".into())
        // .collect()
}



struct PlayGameConfig {
    pub get_game_moves: bool,
    pub show_logs: bool,
    pub wait_for_enter_after_every_move: bool,
}
impl PlayGameConfig {
    fn none() -> Self {
        Self {
            get_game_moves: false,
            show_logs: false,
            wait_for_enter_after_every_move: false,
        }
    }
    fn all() -> Self {
        Self {
            get_game_moves: true,
            show_logs: true,
            wait_for_enter_after_every_move: true,
        }
    }
}

#[derive(Debug)]
enum PlayGameError {
    Quit,
}

fn play_game(
    player_white: BoxDynPlayer,
    player_black: BoxDynPlayer,
    config: PlayGameConfig,
) -> (Result<Winner, PlayGameError>, Option<String>) {
    let maybe_print_position = |board: Board| {
        if config.show_logs {
            let bthv_config = BoardToHumanViewableConfig::all();
            println!("{}", board_to_human_viewable(board, bthv_config));
            if config.wait_for_enter_after_every_move {
                wait_for_enter();
            }
        }
    };

    let mut game = Game::new();

    let mut move_number: usize = 0;
    while game.result() == None && move_number < PLAY_GAME_MOVES_LIMIT {
        move_number += 1;
        // if config.show_logs {
        //     println!("move_number = {move_number}");
        // }

        let board = game.current_position();
        let side_to_move: Color = board.side_to_move();

        maybe_print_position(board);

        let player_to_make_move = match side_to_move {
            Color::White => &player_white,
            Color::Black => &player_black,
        };

        let Some(maybe_best_move) = player_to_make_move.select_move(board) else {
            println!("{game:?}");
            panic!()
        };
        let best_move = match maybe_best_move {
            MaybeChessMove::Move(move_) => move_,
            MaybeChessMove::Surrender => {
                game.resign(side_to_move);
                break
            }
            MaybeChessMove::Quit => {
                return (Err(PlayGameError::Quit), None);
            }
        };
        game.make_move(best_move);

        if game.can_declare_draw() {
            game.declare_draw();
        }
    }

    maybe_print_position(game.current_position());

    let game_res = game.result();
    if config.show_logs {
        println!("game result: {:?}", game_res);
    }
    type GR = GameResult;
    let winner = match game_res.unwrap_or(GR::Stalemate) {
        GR::WhiteCheckmates | GR::BlackResigns => Winner::White,
        GR::WhiteResigns | GR::BlackCheckmates => Winner::Black,
        GR::Stalemate | GR::DrawAccepted | GR::DrawDeclared => Winner::Draw,
    };
    let game_str = if config.get_game_moves { Some(actions_to_string(game.actions().to_vec())) } else { None };
    (Ok(winner), game_str)
}



const MOVES_WASNT_PROVIDED: &str = "moves wasnt provided";

struct PlayTournametConfig { sort: bool, print_games_results: bool, print_games: bool }
/// Plays tournament and sorts AIs.
fn play_tournament(ai_players: &mut Vec<AIwithRating>, config: PlayTournametConfig) {
    let ais_number = ai_players.len();

    let ij_from_i_j = |i: usize, j: usize| -> usize {
        i * ais_number + j
    };
    let ij_to_i_j = |ij: usize| -> (usize, usize) {
        ij.div_mod(ais_number)
    };

    let nn_game_results: Vec<Option<Winner>> = (0..ais_number.pow(2)) // ij
        .collect::<Vec<usize>>()
        .into_par_iter()
        .map(ij_to_i_j)
        .map(|(i, j)| {
            if i == j { return None }
            Some((
                Box::new(ai_players[i].get_ai() as &(dyn Player + Send + Sync)),
                Box::new(ai_players[j].get_ai() as &(dyn Player + Send + Sync)),
            ))
        })
        .map(|option_ai_ij| {
            option_ai_ij.map(|(ai_i, ai_j)| {
                let game_res = play_game(ai_i, ai_j, PlayGameConfig::none());
                game_res.0.unwrap()
            })
        })
        .collect();

    let mut tournament_statistics: HashMap<Winner, u32> = HashMap::new();

    for i in 0..ais_number {
        for j in 0..ais_number {
            if i == j { continue }
            // i->w->white, j->b->black

            let winner: Winner = nn_game_results[ij_from_i_j(i, j)].unwrap();
            if config.print_games_results {
                let winner_char: char = match winner {
                    Winner::White => 'W',
                    Winner::Black => 'B',
                    Winner::Draw  => '.',
                };
                print!("{winner_char}");
            }

            *tournament_statistics.entry(winner).or_insert(0) += 1;

            let [player_i, player_j]: [&mut AIwithRating; 2] = ai_players.get_many_mut([i, j]).unwrap();
            update_ratings(player_i.get_rating_mut(), player_j.get_rating_mut(), winner);
            // if SHOW_TRAINING_LOGS {
            //     println!("new ratings: i={}, j={}", player_i.rating, player_j.rating);
            //     println!();
            // }
        }
        if config.print_games_results {
            print!(" ");
        }
    }
    if config.print_games_results {
        println!();
    }

    if config.sort {
        ai_players.sort_by(|ai1, ai2| ai2.get_rating().partial_cmp(&ai1.get_rating()).unwrap());
    }

    if config.print_games {
        println!("\nstats: {:?}", tournament_statistics);

        let ratings: Vec<Rating> = ai_players
            .iter()
            .map(|p| p.get_rating())
            .collect();
        let ratings_str = ratings
            .iter()
            .map(|r| format!("{r:.2}", r=r.get()))
            .reduce(|acc, el| acc + ", " + &el)
            .unwrap_or_default();
        println!("final ratings (sorted): [{ratings_str}]");
        println!();

        assert!(config.sort);
        {
            let (winner, game_moves) = play_game(
                Box::new(ai_players[0].get_ai()),
                Box::new(ai_players[0].get_ai()),
                PlayGameConfig {
                    get_game_moves: true,
                    ..PlayGameConfig::none()
                }
            );
            println!(
                "BEST vs SELF: winner={winner:?}, moves: ' {moves} '",
                moves = game_moves.unwrap_or(MOVES_WASNT_PROVIDED.to_string()),
            );
            println!();
        }

        {
            let (winner, game_moves) = play_game(
                Box::new(ai_players[0].get_ai()),
                Box::new(ai_players[1].get_ai()),
                PlayGameConfig {
                    get_game_moves: true,
                    ..PlayGameConfig::none()
                }
            );
            println!(
                "BEST vs BEST2: winner={winner:?}, moves: ' {moves} '",
                moves = game_moves.unwrap_or(MOVES_WASNT_PROVIDED.to_string()),
            );
            println!();
        }

        {
            let (winner, game_moves) = play_game(
                Box::new(ai_players[0].get_ai()),
                Box::new(ai_players.last().unwrap().get_ai()),
                PlayGameConfig {
                    get_game_moves: true,
                    ..PlayGameConfig::none()
                }
            );
            println!(
                "BEST vs WORST: winner={winner:?}, moves: ' {moves} '",
                moves = game_moves.unwrap_or(MOVES_WASNT_PROVIDED.to_string()),
            );
            println!();
        }

        {
            let (winner, game_moves) = play_game(
                Box::new(ai_players.last().unwrap().get_ai()),
                Box::new(ai_players.last().unwrap().get_ai()),
                PlayGameConfig {
                    get_game_moves: true,
                    ..PlayGameConfig::none()
                }
            );
            println!(
                "WORST vs SELF: winner={winner:?}, moves: ' {moves} '",
                moves = game_moves.unwrap_or(MOVES_WASNT_PROVIDED.to_string()),
            );
            // println!();
        }
    }
}


fn get_datetime_now() -> String {
    let now = chrono::Local::now();
    let year   = now.format("%Y");
    let month  = now.format("%m");
    let day    = now.format("%d");
    let hour   = now.format("%H");
    let minute = now.format("%M");
    let second = now.format("%S");
    let milis  = now.format("%3f");
    // let micros = now.format("%6f");
    // let nanos  = now.format("%9f");
    format!("{year}-{month}-{day}_{hour}:{minute}:{second}.{milis}")
}


fn process_export_nn_as_images(ai_players: &Vec<AIwithRating>) {
    use std::{fs, io};
    // TODO(enhancement): export all
    let Ok(nn_number) = prompt("Choose NN number to export: ").parse::<usize>() else { println!("Can't parse input as integer."); return };
    let nn_index = number_to_index(nn_number);
    let Some(ai_player) = ai_players.get(nn_index) else { println!("Number out of bounds."); return };

    let ai = ai_player.get_ai();
    let ai_name = ai.get_name().replace(' ', "_");
    let nn = ai.get_nn();

    let export_dir_name = &format!("./exports/{datetime}__{ai_name}", datetime=get_datetime_now());
    let io::Result::Ok(()) = fs::create_dir(export_dir_name) else { println!("Can't create directory to export."); return };

    for (i, layer) in nn.layers.iter().enumerate() {
        #[expect(unused_variables)]
        if let Some((weights_matrix, shift_vector)) = layer.get_fc_weights_shifts() {
            let maybe_simple = if i == 0 { "_simple" } else { "" };
            let file_name_wm_simple = format!(
                "{export_dir_name}/layer_{i:02}{maybe_simple}.png"
            );
            let img = weights_matrix_to_image_simple(weights_matrix);
            let ImageResult::Ok(()) = img.save(file_name_wm_simple) else { println!("Can't save the simple image#{i}."); continue };

            // TODO(feat): also export `shift_vector`.

            if i == 0 {
                use weights_matrix_to_image_smart::{Arrangement, convert};
                const DEBUG: bool = true;
                for arrangement in Arrangement::ALL {
                    let file_name_wm_smart = format!("{export_dir_name}/layer_{i:02}_smart_{arrangement:?}.png");
                    let img = convert::<DEBUG>(weights_matrix, Some(arrangement));
                    let ImageResult::Ok(()) = img.save(file_name_wm_smart) else { println!("Can't save the smart image#{i}."); continue };
                }
                panic!();
            }
        }
    }
    println!("Export finished successfully.");
}

fn rgb_from_value(value_f: float) -> [u8; 3] {
    let g = 0;
    let value_f = 255. * value_f;
    let value_u8: u8 = value_f.abs() as u8;
    let (r, b) = if value_f > 0. { (value_u8, 0) } else { (0, value_u8) };
    [r, g, b]
}

fn weights_matrix_to_image_simple(weights_matrix: &Matrix) -> RgbImage {
    let (wm_rows, wm_cols) = weights_matrix.shape();
    let (w, h) = (wm_rows, wm_cols);
    let mut img: RgbImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            println!();
            println!("wm_rows = {wm_rows}, wm_cols = {wm_cols}");
            println!("w = {w}, h = {h}");
            println!("x = {x}, y = {y}");
            let color = rgb_from_value(weights_matrix[(x, y)]);
            img[(x as u32, y as u32)] = Rgb(color);
        }
    }
    img
}

mod weights_matrix_to_image_smart {
    use super::*; // TODO(refactor): write explicit `use`s.

    const SPACING_BETWEEN_COLORS : usize = 4;
    const SPACING_BETWEEN_OUTPUTS: usize = 2;
    const SPACING_BETWEEN_PIECES : usize = 1;
    // TODO(feat): remake to SPACING_MAJOR, SPACING_MINOR, SPACING_SUB_MINOR.

    const NUMBER_OF_FILES_RANKS: usize = 8;

    // TODO?: REMOVE
    const _64: usize = NUMBER_OF_SQUARES_ON_CHESS_BOARD;
    const _8 : usize = NUMBER_OF_FILES_RANKS;
    const _6 : usize = NUMBER_OF_DIFFERENT_CHESS_PIECES;

    pub fn convert<const DEBUG: bool>(weights_matrix: &Matrix, arrangement: Option<Arrangement>) -> RgbImage {
        let arrangement = arrangement.unwrap_or_default();
        let weights_matrix = weights_matrix.transpose();
        let (wm_rows, wm_cols) = weights_matrix.shape();
        let (w, h) = w_h_from_col_row::<DEBUG>(wm_cols, wm_rows, arrangement);
        let (w, h) = wh_max_to_image_size(w, h, arrangement);
        // assert_eq!(wm_rows * wm_cols, w * h);

        let mut img: RgbImage = ImageBuffer::new(w as u32, h as u32);
        let mut out_of_bounds: u64 = 0;
        let mut total: u64 = 0;
        for r in 0..wm_rows {
            for c in 0..wm_cols {
                let (w, h) = w_h_from_col_row::<DEBUG>(c, r, arrangement);
                // let chess_board_square_index = c % _64;
                // let (dw, dh) = chess_board_square_index.mod_div(_8);
                let (x, y) = (w, h);
                // let (x, y) = (w + dw, h + dh);
                let color = rgb_from_value(weights_matrix[(r, c)]);
                if let Some(pixel) = img.get_pixel_mut_checked(x as u32, y as u32) {
                    *pixel = Rgb(color);
                } else {
                    out_of_bounds += 1;
                    println!();
                    println!("XY OUT OF BOUNDS:");
                    println!("wm_rows = {wm_rows}, wm_cols = {wm_cols}");
                    println!("r = {r}, c = {c}");
                    println!("w = {w}, h = {h}");
                    // println!("chess_board_square_index = {chess_board_square_index}");
                    // println!("dw = {dw}, dh = {dh}");
                    println!("x = {x}, y = {y}");
                }
                total += 1;
            }
        }

        println!();
        let percentage: float = 100. * (out_of_bounds as float) / (total as float);
        println!("out_of_bounds = {out_of_bounds}, total = {total}, percentage = {percentage:.2}%");
        println!();
        img
    }

    /// bc we don't want empty pixels at right and bottom.
    const fn wh_max_to_image_size(w: usize, h: usize, arrangement: Arrangement) -> (usize, usize) {
        use Arrangement::*;
        use SPACING_BETWEEN_COLORS  as C;
        use SPACING_BETWEEN_OUTPUTS as O;
        use SPACING_BETWEEN_PIECES  as P;
        let (dh, dw): (usize, usize) = match arrangement {
            Y_Colors_X_OutputsPieces => (C, O * (P-1)),
            Y_Colors_X_PiecesOutputs => (C, P * (O-1)),
            Y_Outputs_X_ColorsPieces => (O, C * (P-1)),
            Y_Outputs_X_PiecesColors => (O, P * (C-1)),
            Y_Pieces_X_ColorsOutputs => (P, C * (O-1)),
            Y_Pieces_X_OutputsColors => (P, O * (C-1)),
        };
        (w - dw, h - dh)
    }

    /// DC = Depth Channel
    /// b = chess board (8 by x * 8 by y)
    #[allow(non_camel_case_types)] // TODO(when fixed): use `expect`.
    #[derive(Debug, Clone, Copy)]
    pub enum Arrangement {
        /// Firstly organize/split/sort in rows by COLORS, then in cols by OUTPUTS and then by PIECES.
        /// b b b b b b   b b b b b b   ...
        /// < out0 DC >   < out1 DC >   ...
        /// ^ ^ ^ ^ ^ ^
        /// | | | | | king
        /// | | | | queen
        /// | | | rook
        /// | | bishop
        /// | knight
        /// pawn
        Y_Colors_X_OutputsPieces,

        /// Firstly organize/split/sort in rows by COLORS, then in cols by PIECES and then by OUTPUTS.
        /// b b b ... b         b b b ... b b         ...
        /// < pawn DC >         < knight DC >         ...
        /// ^ ^ ^ ... ^
        /// | | |     outputLast
        /// | | output2
        /// | output1
        /// output0
        Y_Colors_X_PiecesOutputs,

        /// Firstly organize/split/sort in rows by OUTPUTS, then in cols by COLORS and then by PIECES.
        /// b b b  b b b   ...            ...                      ...
        /// < white DC >   < black DC >   < white and black DC >   < white and neg black DC >
        /// ^ ^ ^  ^ ^ ^
        /// | | |  | | king
        /// | | |  | queen
        /// | | |  rook
        /// | | bishop
        /// | knight
        /// pawn
        Y_Outputs_X_ColorsPieces,

        /// Firstly organize/split/sort in rows by OUTPUTS, then in cols by PIECES and then by COLORS.
        /// b  b   b  b   ...             ...             ...             ...            ...
        /// < pawn DC >   < knight DC >   < bishop DC >   < rook DC >   < queen DC >   < king DC >
        /// ^  ^   ^  ^
        /// |  |   |  white and black neg DC
        /// |  |   white and black DC
        /// |  black DC
        /// white DC
        Y_Outputs_X_PiecesColors,

        /// Firstly organize/split/sort in rows by PIECES, then in cols by COLORS and then by OUTPUTS.
        /// b b b ...  b   ...            ...                      ...
        /// < white DC >   < black DC >   < white and black DC >   < white and neg black DC >
        /// ^ ^ ^ ...  ^
        /// | | |      outputLast
        /// | | output2
        /// | output1
        /// output0
        Y_Pieces_X_ColorsOutputs,

        /// Firstly organize/split/sort in rows by PIECES, then in cols by OUTPUTS and then by COLORS.
        /// b b  b b   ...        ...        ...
        /// < out0 >   < out1 >   < out2 >   ...
        /// ^ ^  ^ ^
        /// | |  | white and neg black DC
        /// | |  white and black DC
        /// | black DC
        /// white DC
        Y_Pieces_X_OutputsColors,
    }
    impl Arrangement {
        pub const ALL: [Self; 6] = [
            Self::Y_Colors_X_OutputsPieces,
            Self::Y_Colors_X_PiecesOutputs,
            Self::Y_Outputs_X_ColorsPieces,
            Self::Y_Outputs_X_PiecesColors,
            Self::Y_Pieces_X_ColorsOutputs,
            Self::Y_Pieces_X_OutputsColors,
        ];
    }
    impl Default for Arrangement {
        fn default() -> Self {
            Self::Y_Colors_X_OutputsPieces
        }
    }

    fn w_h_from_col_row<const DEBUG: bool>(col: usize, row: usize, arrangement: Arrangement) -> (usize, usize) {
        use Arrangement::*;
        if DEBUG { println!("col = {col}, row = {row}") }
        (match arrangement {
            Y_Colors_X_OutputsPieces => w_h_from_col_row__y_colors_x_outputs_pieces::<DEBUG>,
            Y_Colors_X_PiecesOutputs => w_h_from_col_row__y_colors_x_pieces_outputs::<DEBUG>,
            Y_Outputs_X_ColorsPieces => w_h_from_col_row__y_outputs_x_colors_pieces::<DEBUG>,
            Y_Outputs_X_PiecesColors => w_h_from_col_row__y_outputs_x_pieces_colors::<DEBUG>,
            Y_Pieces_X_ColorsOutputs => w_h_from_col_row__y_pieces_x_colors_outputs::<DEBUG>,
            Y_Pieces_X_OutputsColors => w_h_from_col_row__y_pieces_x_outputs_colors::<DEBUG>,
        })(col, row)
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_colors_x_outputs_pieces<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        let (index_of_color_depchan, index_in_color_depchan) = col.div_mod(NN_INPUT_SIZE_PER_COLOR_CHANNEL);
        if DEBUG { println!("index_of_color_depchan = {index_of_color_depchan}, index_in_color_depchan = {index_in_color_depchan}") }
        let (index_of_piece_depchan, index_in_piece_depchan) = index_in_color_depchan.div_mod(_64);
        if DEBUG { println!("index_of_piece_depchan = {index_of_piece_depchan}, index_in_piece_depchan = {index_in_piece_depchan}") }
        let (w_in_piece_depchan, h_in_piece_depchan) = index_in_piece_depchan.mod_div(_8);
        if DEBUG { println!("w_in_piece_depchan = {w_in_piece_depchan}, h_in_piece_depchan = {h_in_piece_depchan}") }
        let w = {
            w_in_piece_depchan
            + index_of_piece_depchan * (_8 + SPACING_BETWEEN_PIECES)
            + row * _6 * (_8 + SPACING_BETWEEN_OUTPUTS)
        };
        let h = {
            h_in_piece_depchan
            + index_of_color_depchan * (_8 + SPACING_BETWEEN_COLORS)
        };
        (w, h)
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_colors_x_pieces_outputs<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        let (index_of_color_depchan, index_in_color_depchan) = col.div_mod(NN_INPUT_SIZE_PER_COLOR_CHANNEL);
        if DEBUG { println!("index_of_color_depchan = {index_of_color_depchan}, index_in_color_depchan = {index_in_color_depchan}") }
        // compile_error!();
        let (index_of_piece_depchan, index_in_piece_depchan) = index_in_color_depchan.div_mod(_64);
        if DEBUG { println!("index_of_piece_depchan = {index_of_piece_depchan}, index_in_piece_depchan = {index_in_piece_depchan}") }
        let (w_in_piece_depchan, h_in_piece_depchan) = index_in_piece_depchan.mod_div(_8);
        if DEBUG { println!("w_in_piece_depchan = {w_in_piece_depchan}, h_in_piece_depchan = {h_in_piece_depchan}") }
        let w = {
            w_in_piece_depchan
            + index_of_piece_depchan * (_8 + SPACING_BETWEEN_PIECES)
            + row * _6 * (_8 + SPACING_BETWEEN_OUTPUTS)
        };
        let h = {
            h_in_piece_depchan
            + index_of_color_depchan * (_8 + SPACING_BETWEEN_COLORS)
        };
        (w, h)
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_outputs_x_colors_pieces<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        todo!()
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_outputs_x_pieces_colors<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        todo!()
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_pieces_x_colors_outputs<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        todo!()
    }

    #[allow(non_snake_case)] // TODO(when fixed): change to `expect`.
    fn w_h_from_col_row__y_pieces_x_outputs_colors<const DEBUG: bool>(col: usize, row: usize) -> (usize, usize) {
        todo!()
    }

    #[cfg(test)]
    mod w_h_from_col_depchan {
        use super::*;
        mod y_colors_x_outputs_pieces {
            use super::*;
            use w_h_from_col_row__y_colors_x_outputs_pieces as w_h_from_col_row;
            const ARRANGEMENT: Arrangement = Arrangement::Y_Colors_X_OutputsPieces;

            #[test]
            fn for_image_size() {
                const DEPTH_CHANNELS: usize = 4;
                const NUMBER_OF_INPUTS: usize = DEPTH_CHANNELS * NN_INPUT_SIZE_PER_COLOR_CHANNEL; // 1536
                const NUMBER_OF_OUTPUTS: usize = 100;
                const W_EXPECTED: usize = {
                    NUMBER_OF_OUTPUTS * _6 * (_8 + SPACING_BETWEEN_OUTPUTS)
                    - SPACING_BETWEEN_OUTPUTS * (SPACING_BETWEEN_PIECES - 1)
                };
                const H_EXPECTED: usize = {
                    DEPTH_CHANNELS * (_8 + SPACING_BETWEEN_COLORS)
                    - SPACING_BETWEEN_COLORS
                };
                let (w_actual, h_actual) = w_h_from_col_row::<true>(NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS);
                let (w_actual, h_actual) = wh_max_to_image_size(w_actual, h_actual, ARRANGEMENT);
                assert_eq!((W_EXPECTED, H_EXPECTED), (w_actual, h_actual)); // bc we don't want empty pixels at right and bottom.
            }

            #[test]
            fn for_col_0() {
                let col = 0;
                assert_eq!(
                    (1-1, 1-1), // -1 bc indices start from 0, not 1.
                    w_h_from_col_row::<true>(col, 0)
                );
            }
            #[test]
            fn for_col_3() {
                let col = 3;
                assert_eq!(
                    (4-1, 1-1), // -1 bc indices start from 0, not 1.
                    w_h_from_col_row::<true>(col, 0)
                );
            }
            #[test]
            fn for_col_10() {
                let col = 10;
                assert_eq!(
                    (3-1, 2-1), // -1 bc indices start from 0, not 1.
                    w_h_from_col_row::<true>(col, 0)
                );
            }

            #[test]
            fn for_col_0_p_64() { // 0 + 64
                // this test is based on `for_col_0`.
                let col = 0 + 64;
                assert_eq!(
                    (1-1+_8+SPACING_BETWEEN_PIECES, 1-1),
                    w_h_from_col_row::<true>(col, 0)
                );
            }
            #[test]
            fn for_col_3_p_64() { // 3 + 64
                // this test is based on `for_col_3`.
                let col = 3 + 64;
                assert_eq!(
                    (4-1+_8+SPACING_BETWEEN_PIECES, 1-1),
                    w_h_from_col_row::<true>(col, 0)
                );
            }
            #[test]
            fn for_col_10_p_64() { // 10 + 64
                // this test is based on `for_col_10`.
                let col = 10 + 64;
                assert_eq!(
                    (3-1+_8+SPACING_BETWEEN_PIECES, 2-1),
                    w_h_from_col_row::<true>(col, 0)
                );
            }
        }

        mod y_colors_x_pieces_outputs {
            use super::*;
            use w_h_from_col_row__y_colors_x_pieces_outputs as w_h_from_col_row;
            const ARRANGEMENT: Arrangement = Arrangement::Y_Colors_X_PiecesOutputs;

            #[ignore]
            #[test]
            fn for_image_size() {
                const DEPTH_CHANNELS: usize = 4;
                const NUMBER_OF_INPUTS: usize = DEPTH_CHANNELS * NN_INPUT_SIZE_PER_COLOR_CHANNEL; // 1536
                const NUMBER_OF_OUTPUTS: usize = 100;
                const W_EXPECTED: usize = {
                    NUMBER_OF_OUTPUTS * _6 * (_8 + SPACING_BETWEEN_OUTPUTS)
                    - SPACING_BETWEEN_OUTPUTS * (SPACING_BETWEEN_PIECES + 1)
                };
                const H_EXPECTED: usize = {
                    DEPTH_CHANNELS * (_8 + SPACING_BETWEEN_COLORS) - SPACING_BETWEEN_COLORS
                };
                let (w_actual, h_actual) = w_h_from_col_row::<true>(NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS);
                let (w_actual, h_actual) = wh_max_to_image_size(w_actual, h_actual, ARRANGEMENT);
                assert_eq!((W_EXPECTED, H_EXPECTED), (w_actual, h_actual)); // bc we don't want empty pixels at right and bottom.
            }

            // TODO: other tests
        }

        mod y_outputs_x_colors_pieces {
            // TODO: tests
        }
        mod y_outputs_x_pieces_colors {
            // TODO: tests
        }
        mod y_pieces_x_colors_outputs {
            // TODO: tests
        }
        mod y_pieces_x_outputs_colors {
            // TODO: tests
        }
    }


    #[test]
    fn assert_number_of_files_ranks_eq_8() {
        assert_eq!(8, _8);
    }
}


// BENCHMARKS:

#[cfg(test)]
mod benchmarks {
    extern crate test;
    use test::{Bencher, black_box};
    use rand::{Rng, thread_rng};
    use crate::float_type::float;

    mod af_straightforward_vs_pade_approx {
        use super::*;
        mod sigmoid {
            use super::*;
            const X_MIN: float = -10.;
            const X_MAX: float = 10.;
            #[bench]
            fn straightforward(bencher: &mut Bencher) {
                use crate::math_functions::sigmoid;
                // TODO?: make `static mut`
                let x = thread_rng().gen_range(X_MIN..X_MAX);
                bencher.iter(|| {
                    let x = black_box(x);
                    sigmoid(x)
                })
            }
            #[bench]
            #[ignore]
            fn pade_approx(bencher: &mut Bencher) {
                // use crate::math_functions_pade_approx::sigmoid;
                let x = thread_rng().gen_range(X_MIN..X_MAX);
                #[expect(unused_variables)]
                bencher.iter(|| {
                    let x = black_box(x);
                    // sigmoid(x)
                })
            }
        }
        // mod sigmoid_v {
        //     use super::*;
        //     mod straightforward {
        //         use super::*;
        //         #[bench]
        //         fn _1(bencher: &mut Bencher) {
        //             const N: usize = 1;
        //             bencher.iter(|| {
        //                 sigmoid_v()
        //             })
        //         }
        //     }
        // }
        // TODO: mod tanh {}
    }
}



fn index_to_number(i: usize) -> usize {
    // bc if index is 0 then it's number 1, 1=>2, 2=>3, and so on
    i + 1
}
fn number_to_index(n: usize) -> usize {
    assert!(n > 0);
    // bc if index is 0 then it's number 1, 1=>2, 2=>3, and so on
    n - 1
}
fn number_to_index_checked(n: usize) -> Option<usize> {
    if n > 0 { Some(number_to_index(n)) } else { None }
}


/// Enum of variants of Number of Depth Channels to use for NN's input.
#[allow(dead_code)] // TODO(lter, when fixed): change `allow` to `expect`.
#[repr(usize)]
enum NumberOfDepthChannels {
    /// Channels: White, Black.
    /// => `NN_INPUT_SIZE` = 768
    Two = 2,
    /// Channels: White, Black, White and (maybe negative) Black.
    /// => `NN_INPUT_SIZE` = 1152
    Three { use_opposite_signs: bool } = 3,
    /// Channels: White, Black, White and Black, White and Negative Black.
    /// => `NN_INPUT_SIZE` = 1536
    Four = 4,
}
impl NumberOfDepthChannels {
    /// Returns discriminant - number used to represent enum's variant number.
    ///
    /// src: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
    const fn discriminant(&self) -> usize {
        unsafe { *(self as *const Self as *const usize) }
    }
}

#[cfg(test)]
mod number_of_depth_channels {
    /// These tests are critical bc if they fail it will probably lead to big and scary UB.
    mod discriminant {
        use crate::NumberOfDepthChannels as NODC;
        #[test]
        fn two() {
            assert_eq!(
                2,
                NODC::Two.discriminant()
            );
        }
        #[test]
        fn three_false() {
            assert_eq!(
                3,
                NODC::Three { use_opposite_signs: false }.discriminant()
            );
        }
        #[test]
        fn three_true() {
            assert_eq!(
                3,
                NODC::Three { use_opposite_signs: true }.discriminant()
            );
        }
        #[test]
        fn four() {
            assert_eq!(
                4,
                NODC::Four.discriminant()
            );
        }
    }
}

#[test]
fn const_assert_number_of_different_chess_pieces_eq_6() {
    assert_eq!(6, NUMBER_OF_DIFFERENT_CHESS_PIECES);
}
#[test]
fn const_assert_number_of_squares_on_chess_board_eq_64() {
    assert_eq!(64, NUMBER_OF_SQUARES_ON_CHESS_BOARD);
}



trait ExtensionDivMod where Self: Sized {
    fn div_mod(&self, modulo: Self) -> (Self, Self);
    fn mod_div(&self, modulo: Self) -> (Self, Self);
}
impl ExtensionDivMod for usize {
    fn div_mod(&self, modulo: Self) -> (Self, Self) {
        (self / modulo, self % modulo)
    }
    fn mod_div(&self, modulo: Self) -> (Self, Self) {
        (self % modulo, self / modulo)
    }
}

