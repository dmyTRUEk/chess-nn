//! Main file

#![feature(
    // array_chunks,
    // array_windows,
    // adt_const_params,
    // const_trait_impl,
    file_create_new,
    iter_map_windows,
    let_chains,
    // slice_group_by,
)]

#![deny(
    dead_code,
    unreachable_patterns,
)]


use std::{
    cmp::Ordering,
    collections::HashMap,
    str::FromStr,
};

use chess::{Action, Board, BoardBuilder, ChessMove, Color, File, Game, GameResult, MoveGen, Piece, Rank, Square};
use math_functions::exp;
use rand::{Rng, seq::SliceRandom, thread_rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

mod extensions;
mod float_type;
mod linalg_types;
mod math_functions;
// mod neural_network_col;
mod neural_network_row;
mod utils_io;

use crate::{
    float_type::float,
    neural_network_row::{
        ChessNeuralNetwork,
        layers::LayerSpecs as LS,
        vector_type::Vector,
    },
    utils_io::{flush, print_and_flush, prompt, wait_for_enter},
};



const FILENAMES_ALL_DATA: &[&str] = &[
    "positions/lt_part1_evaluated_2023-11-11_00-18-58",
    "positions/lt_part2_evaluated_2023-11-12_12-48-37",
    "positions/lt_part3_evaluated_2023-11-13_09-46-52",
    "positions/lt_part4_evaluated_2023-11-13_19-26-50",
    "positions/lt_part5_evaluated_2023-11-16_21-16-25",
    "positions/pc_part1_evaluated_2023-11-12_15-17-11",
    "positions/pc_part2_evaluated_2023-11-13_09-35-55",
    "positions/pc_part3_evaluated_2023-11-13_14-34-22",
    "positions/pc_part4_evaluated_2023-11-13_18-45-15",
    "positions/pc_part5_evaluated_2023-11-13_19-27-46",
    "positions/pc_part6_evaluated_2023-11-14_09-01-36",
    "positions/pc_part7_evaluated_2023-11-14_13-26-40",
    "positions/pc_part8_evaluated_2023-11-15_09-36-39",
    "positions/pc_part9_evaluated_2023-11-16_15-59-12",
    "positions/pc_part10_evaluated_2023-11-16_18-54-26",
];
const FILENAME_TO_SAVE_POSITIONS: &str = "positions/lt_or_pc_partN";

mod fully_connected_layer_initial_values {
    use crate::float_type::float;
    pub const W_MIN: float = -0.15; // this fixes getting NaN for at least Relu & Gaussian
    pub const W_MAX: float =  0.1;
    pub const S_MIN: float = -0.1;
    pub const S_MAX: float =  0.1;
}

const TRAIN_TO_TEST_RATIO: float = 0.9;

const NN_INPUT_SIZE: usize = 768; // 2 * 6 * 64

/// Starting learning rate, it will gradually decrease with epochs.
const LEARNING_RATE_0: float = 0.01;
const LEARNING_RATE_EXP_K: float = 2.;
const TRAINING_EPOCHS: u32 = 1_000;

const TOURNAMENTS_NUMBER: u32 = 10;
const DEFAULT_RATING: float = 1_000.;
const NN_RESULT_RANDOM_CHOICE: Option<(float, float)> = Some((0.9, 1.1));
const PLAY_GAME_MOVES_LIMIT: u32 = 500;

const PLAY_WITH_NN_AFTER_TRAINING: bool = true;



fn main() {
    // generate_and_save_random_positions(10_000);
    // return;

    print_and_flush("Creating Neural Networks... ");
    let mut ais: Vec<AI> = vec![
        AI::new(
            "Weighted Sum",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(1),
            ]),
        ),

        // AI::new(
        //     "FC 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::FullyConnected(10),
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::FullyConnected(50),
        //         LS::FullyConnected(20),
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        // AI::new(
        //     "FC-Abs 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_Abs,
        //         LS::FullyConnected(10),
        //         LS::AF_Abs,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC-Abs 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_Abs,
        //         LS::FullyConnected(50),
        //         LS::AF_Abs,
        //         LS::FullyConnected(20),
        //         LS::AF_Abs,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        AI::new(
            "FC-BinaryStep 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_BinaryStep,
                LS::FullyConnected(10),
                LS::AF_BinaryStep,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-BinaryStep 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_BinaryStep,
                LS::FullyConnected(50),
                LS::AF_BinaryStep,
                LS::FullyConnected(20),
                LS::AF_BinaryStep,
                LS::FullyConnected(1),
            ]),
        ),

        AI::new(
            "FC-Elu 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Elu,
                LS::FullyConnected(10),
                LS::AF_Elu,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Elu 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Elu,
                LS::FullyConnected(50),
                LS::AF_Elu,
                LS::FullyConnected(20),
                LS::AF_Elu,
                LS::FullyConnected(1),
            ]),
        ),

        AI::new(
            "FC-Gaussian 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Gaussian,
                LS::FullyConnected(10),
                LS::AF_BinaryStep,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Gaussian 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Gaussian,
                LS::FullyConnected(50),
                LS::AF_Gaussian,
                LS::FullyConnected(20),
                LS::AF_BinaryStep,
                LS::FullyConnected(1),
            ]),
        ),

        // AI::new(
        //     "FC-LeakyRelu 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_LeakyRelu,
        //         LS::FullyConnected(10),
        //         LS::AF_LeakyRelu,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC-LeakyRelu 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_LeakyRelu,
        //         LS::FullyConnected(50),
        //         LS::AF_LeakyRelu,
        //         LS::FullyConnected(20),
        //         LS::AF_LeakyRelu,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        // TODO: check if activation function implemented correctly.
        // AI::new(
        //     "FC-MaxOut 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_MaxOut,
        //         LS::FullyConnected(10),
        //         LS::AF_MaxOut,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "AF_MaxOut 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_MaxOut,
        //         LS::FullyConnected(50),
        //         LS::AF_MaxOut,
        //         LS::FullyConnected(20),
        //         LS::AF_MaxOut,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        AI::new(
            "FC-Relu 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Relu,
                LS::FullyConnected(10),
                LS::AF_Relu,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Relu 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Relu,
                LS::FullyConnected(50),
                LS::AF_Relu,
                LS::FullyConnected(20),
                LS::AF_Relu,
                LS::FullyConnected(1),
            ]),
        ),

        AI::new(
            "FC-Sigmoid 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Sigmoid,
                LS::FullyConnected(10),
                LS::AF_Sigmoid,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Sigmoid 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Sigmoid,
                LS::FullyConnected(50),
                LS::AF_Sigmoid,
                LS::FullyConnected(20),
                LS::AF_Sigmoid,
                LS::FullyConnected(1),
            ]),
        ),

        AI::new(
            "FC-Signum 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Signum,
                LS::FullyConnected(10),
                LS::AF_Signum,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Signum 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Signum,
                LS::FullyConnected(50),
                LS::AF_Signum,
                LS::FullyConnected(20),
                LS::AF_Signum,
                LS::FullyConnected(1),
            ]),
        ),

        // AI::new(
        //     "FC-SignSqrtAbs 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_SignSqrtAbs,
        //         LS::FullyConnected(10),
        //         LS::AF_SignSqrtAbs,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC-SignSqrtAbs 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_SignSqrtAbs,
        //         LS::FullyConnected(50),
        //         LS::AF_SignSqrtAbs,
        //         LS::FullyConnected(20),
        //         LS::AF_SignSqrtAbs,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        // AI::new(
        //     "FC-Silu 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_Silu,
        //         LS::FullyConnected(10),
        //         LS::AF_Silu,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC-Silu 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_Silu,
        //         LS::FullyConnected(50),
        //         LS::AF_Silu,
        //         LS::FullyConnected(20),
        //         LS::AF_Silu,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        // UNIMPLEMENTED
        // AI::new(
        //     "FC-SoftMax 100-10",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_SoftMax,
        //         LS::FullyConnected(10),
        //         LS::AF_SoftMax,
        //         LS::FullyConnected(1),
        //     ]),
        // ),
        // AI::new(
        //     "FC-SoftMax 100-50-20",
        //     ChessNeuralNetwork::from_layers_specs(vec![
        //         LS::FullyConnected(100),
        //         LS::AF_SoftMax,
        //         LS::FullyConnected(50),
        //         LS::AF_SoftMax,
        //         LS::FullyConnected(20),
        //         LS::AF_SoftMax,
        //         LS::FullyConnected(1),
        //     ]),
        // ),

        AI::new(
            "FC-SoftPlus 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_SoftPlus,
                LS::FullyConnected(10),
                LS::AF_SoftPlus,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-SoftPlus 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_SoftPlus,
                LS::FullyConnected(50),
                LS::AF_SoftPlus,
                LS::FullyConnected(20),
                LS::AF_SoftPlus,
                LS::FullyConnected(1),
            ]),
        ),

        AI::new( // CRAZY BUG??: breaks (gives NaN) on `LEARNING_RATE_0` = 0.1?!?
            "FC-Tanh 100-10",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Tanh,
                LS::FullyConnected(10),
                LS::AF_Tanh,
                LS::FullyConnected(1),
            ]),
        ),
        AI::new(
            "FC-Tanh 100-50-20",
            ChessNeuralNetwork::from_layers_specs(vec![
                LS::FullyConnected(100),
                LS::AF_Tanh,
                LS::FullyConnected(50),
                LS::AF_Tanh,
                LS::FullyConnected(20),
                LS::AF_Tanh,
                LS::FullyConnected(1),
            ]),
        ),

    ];
    println!("{} created.", ais.len());
    assert!(ais.len() > 1, "number of AIs should be > 1, else its not interesting, but it is {}", ais.len());

    // let mut rng = thread_rng();
    // ais.shuffle(&mut rng);

    print!("Loading data from {} files... ", FILENAMES_ALL_DATA.len()); flush();
    let all_data = load_all_data_str(FILENAMES_ALL_DATA);
    println!("{} lines loaded.", all_data.len());

    print_and_flush("Loading data... ");
    let all_data = AnyData::from(all_data);
    println!("{} samples loaded.", all_data.xy.len());

    print!("Splitting data to train and test datasets with `train/test ratio`={TRAIN_TO_TEST_RATIO}... "); flush();
    let train_and_test_data = TrainAndTestData::from(all_data, TRAIN_TO_TEST_RATIO);
    println!("Done.");

    // TODO?: z-transformation

    println!("Starting training Neural Networks...\n");
    // TODO?: if NN returns NaN => recreate it few times, else delete it
    train_nns(&mut ais, train_and_test_data);

    println!("Playing {TOURNAMENTS_NUMBER} tournaments to set ratings...");
    for i in 1..=TOURNAMENTS_NUMBER {
        print!("#{i}:\t"); flush();
        play_tournament(
            &mut ais,
            PlayTournametConfig {
                print_games_results: true,
                print_games: i == TOURNAMENTS_NUMBER,
            }
        );
    }
    let ais = ais;
    fn print_ais_ratings(ais: &Vec<AI>, is_first_time: bool) {
        let maybe_after_n_tournaments_str = if is_first_time { format!(" after {TOURNAMENTS_NUMBER} tournaments") } else { "".to_string() };
        println!("AIs' ratings{maybe_after_n_tournaments_str}:");
        for (i, ai) in ais.iter().enumerate() {
            println!("#{i}: {r:.2} - {n}", i=i+1, r=ai.rating, n=ai.name);
        }
    }
    print_ais_ratings(&ais, true);

    if !PLAY_WITH_NN_AFTER_TRAINING { return }

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
        let line = prompt("So what's it gonna be, huuh? ");
        enum NeuralNetworkToPlayWith {
            Best,
            Worst,
            Index(usize),
            Name(String),
        }
        type NNTPW = NeuralNetworkToPlayWith;
        let nn_to_play_with: NNTPW = match line.as_str() {
            CMD_QUIT_SHORT | CMD_QUIT_FULL => { break }
            CMD_LIST_SHORT | CMD_LIST_FULL => { print_ais_ratings(&ais, false); continue }
            CMD_BEST_STR => NNTPW::Best,
            CMD_WORST_STR => NNTPW::Worst,
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
        const CMD_SIDE_TO_PLAY_WHITE_SHORT: &str = "w";
        const CMD_SIDE_TO_PLAY_WHITE_FULL : &str = "white";
        const CMD_SIDE_TO_PLAY_BLACK_SHORT: &str = "b";
        const CMD_SIDE_TO_PLAY_BLACK_FULL : &str = "black";
        println!("Choose side to play:");
        println!("- white: `w` or `white`");
        println!("- black: `b` or `black`");
        println!("- return back: anything else");
        let line = prompt("Choose wisely. ");
        let human_side_to_play: Color = match line.as_str() {
            CMD_SIDE_TO_PLAY_WHITE_SHORT | CMD_SIDE_TO_PLAY_WHITE_FULL => Color::White,
            CMD_SIDE_TO_PLAY_BLACK_SHORT | CMD_SIDE_TO_PLAY_BLACK_FULL => Color::Black,
            _ => { continue } // ask everything from start again
        };
        let config = PlayGameConfig {
            wait_for_enter_after_every_move: false,
            ..PlayGameConfig::all(human_side_to_play)
        };
        println!("Good luck! In any unclear situation use `s` to surrender or `q` to quit");
        let (winner, game_moves) = play_game(&nn_to_play_with, &nn_to_play_with, config);
        let Ok(winner) = winner else { continue };
        println!(
            "{who_vs_who}: winner={winner:?}, moves: ' {moves} '\n",
            who_vs_who = match human_side_to_play {
                Color::White => "HUMAN vs NN_BEST",
                Color::Black => "NN_BEST vs HUMAN",
            },
            moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
        );
    }
}


fn train_nns(ais: &mut Vec<AI>, train_and_test_data: TrainAndTestData) {
    let TrainAndTestData { mut train_data, mut test_data } = train_and_test_data;
    let mut rng = thread_rng();
    for epoch in 0..TRAINING_EPOCHS {
        let learning_rate = learning_rate_from_epoch(epoch);
        train_data.xy.shuffle(&mut rng);
        test_data.xy.shuffle(&mut rng);
        let mut msg = format!("Epoch {ep}/{TRAINING_EPOCHS}:\n", ep=epoch+1);
        msg += &ais
            // .iter_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(i, ai)| {
                let avg_train_error = train_step(&mut ai.nn, &train_data, learning_rate);
                let avg_test_error = calc_avg_test_error(&ai.nn, &test_data);
                format!("NN#{i}\tavg_train_error = {avg_train_error}\tavg_test_error = {avg_test_error}\t{name}", i=i+1, name=ai.name)
            })
            // .reduce(|acc, el| acc + " " + &el)
            // .unwrap_or_default()
            .collect::<Vec<String>>()
            .join("\n");
        println!("{msg}\n");
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
    }
    let avg_error = total_error / (test_data.xy.len() as float);
    avg_error.sqrt()
}

fn learning_rate_from_epoch(epoch: u32) -> float {
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


#[allow(dead_code)]
fn generate_and_save_random_positions(fens_number: usize) {
    let fens = generate_random_positions(fens_number);
    save_random_positions(fens);
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
            let random_move_index = rng.gen_range(0..possible_moves.len());
            let random_move = possible_moves[random_move_index];
            game.make_move(random_move);
            if game.result().is_some() { break }
            if game.can_declare_draw() { break }
            fens.push(game.current_position().to_string());
        }
    }
    fens
}

fn save_random_positions(fens: Vec<String>) {
    use std::io::Write;
    let mut file = ::std::fs::File::create_new(FILENAME_TO_SAVE_POSITIONS).unwrap();
    for fen in fens {
        writeln!(file, "{fen}").unwrap();
    }
}

fn load_all_data_str(filenames: &[&str]) -> Vec<String> {
    use ::std::io::BufRead;
    let mut all_data_str = filenames
        .into_iter()
        .flat_map(|filename| {
            let file = ::std::fs::File::open(filename).unwrap();
            ::std::io::BufReader::new(file)
                .lines()
                .map(|line| line.unwrap())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<String>>();
    all_data_str.shrink_to_fit();
    all_data_str
}



// mod pieces_values {
//     use chess::{Color, Piece};

//     use crate::float_type::float;

//     pub const NONE: float = 0.;

//     // pub const PAWN  : float = 1.;
//     // pub const KNIGHT: float = 2.7;
//     // pub const BISHOP: float = 3.;
//     // pub const ROOK  : float = 5.;
//     // pub const QUEEN : float = 7.;
//     // pub const KING  : float = 15.;
//     pub const PAWN  : float = 0.1;
//     pub const KNIGHT: float = 0.2;
//     pub const BISHOP: float = 0.4;
//     pub const ROOK  : float = 0.6;
//     pub const QUEEN : float = 0.8;
//     pub const KING  : float = 1.;

//     pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> float {
//         let Some((piece, color)) = option_piece_and_color else { return NONE };
//         let value = match piece {
//             Piece::Pawn   => PAWN,
//             Piece::Knight => KNIGHT,
//             Piece::Bishop => BISHOP,
//             Piece::Rook   => ROOK,
//             Piece::Queen  => QUEEN,
//             Piece::King   => KING,
//         };
//         match color {
//             Color::White => value,
//             Color::Black => -value,
//         }
//     }
// }

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
            let square = unsafe { Square::new(index) }; // SAFETY: this is safe bc `index` is from 0 to 64 (not including)
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



fn board_to_vector_for_nn(board: Board) -> Vector {
    fn board_to_vector(board: Board) -> Vector {
        let mut vector: Vector = Vector::zeros(NN_INPUT_SIZE);
        let board_builder: BoardBuilder = board.into();
        for index_in_64 in 0..64 {
            let square: Square = unsafe { Square::new(index_in_64) }; // SAFETY: this is safe bc `index_in_64` is from 0 to 64 (not including)
            let option_piece_and_color: Option<(Piece, Color)> = board_builder[square];
            if let Some(piece_and_color) = option_piece_and_color {
                let index_of_64 = match piece_and_color {
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
                vector[64*index_of_64 + (index_in_64 as usize)] = 1.;
            }
        }
        vector
    }
    let input_for_nn = board_to_vector(board);
    // if board.side_to_move() == Color::Black {
    //     // TODO?: check what is the correct way
    //     // input_for_nn = Vector::from_iterator(NN_INPUT_SIZE, input_for_nn.into_iter().rev().map(|&x| x));
    //     // input_for_nn = -input_for_nn;
    //     // after 100 epochs:
    //     // 0,0 ->
    //     // 0,1 ->
    //     // 1,0 ->
    //     // 1,1 ->
    // }
    // println!("{input_for_nn:?}");
    input_for_nn
}

fn analyze(board: Board, nn: &ChessNeuralNetwork) -> float {
    let input_for_nn = board_to_vector_for_nn(board);
    // println!("input_for_nn = {:?}", array_board);
    nn.process_input(input_for_nn)
}


// /// Return `pieces_sum_white - pieces_sum_black`
// fn board_to_pieces_sum(board: Board) -> float {
//     let board_builder: BoardBuilder = board.into();
//     let mut pieces_sum = 0.;
//     for i in 0..64 {
//         let square = unsafe { Square::new(i) }; // SAFETY: this is safe bc `i` is from 0 to 64 (not including)
//         let option_piece_and_color = board_builder[square];
//         pieces_sum += pieces_values::get(option_piece_and_color);
//     }
//     pieces_sum
// }




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
    Quit,
}
fn get_move_from_human(game: &Game) -> MoveByHuman {
    loop {
        const CMD_QUIT: &str = "q";
        const CMD_SURRENDER: &str = "s";
        let line = prompt("Your move: ");
        match line.as_str() {
            CMD_QUIT => { return MoveByHuman::Quit }
            CMD_SURRENDER => { return MoveByHuman::Surrender }
            _ => {}
        }
        let move_ = string_to_chess_move(&line);
        if let Some(move_) = move_ && game.current_position().legal(move_) {
            return MoveByHuman::Move(move_);
        }
        println!("Illegal move.");
    }
}


struct PlayGameConfig {
    pub get_game_moves: bool,
    pub show_logs: bool,
    pub wait_for_enter_after_every_move: bool,
    pub human_color: Option<Color>,
}
impl PlayGameConfig {
    fn none() -> Self {
        Self { get_game_moves: false, show_logs: false, wait_for_enter_after_every_move: false, human_color: None }
    }
    fn all(human_color: Color) -> Self {
        Self { get_game_moves: true, show_logs: true, wait_for_enter_after_every_move: true, human_color: Some(human_color) }
    }
}

#[derive(Debug)]
enum PlayGameError {
    Quit,
}

fn play_game(
    nn_white: &ChessNeuralNetwork,
    nn_black: &ChessNeuralNetwork,
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
    let mut rng = thread_rng();

    maybe_print_position(game.current_position());

    let mut move_number: u32 = 0;
    while game.result() == None && move_number < 2*PLAY_GAME_MOVES_LIMIT {
        move_number += 1;
        // if config.show_logs {
        //     println!("move_number = {move_number}");
        // }

        let side_to_move: Color = game.current_position().side_to_move();

        // if vs human
        if let Some(human_color) = config.human_color && human_color == side_to_move {
            let move_by_human = get_move_from_human(&game);
            let move_ = match move_by_human {
                MoveByHuman::Quit => { return (Err(PlayGameError::Quit), None) }
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

        #[derive(Debug, Copy, Clone)]
        struct MoveWithMark { move_: ChessMove, score: float, weight: float }
        impl MoveWithMark {
            fn get_weighted_score(&self) -> float { self.score * self.weight }
        }

        let mut omwm_best: Option<MoveWithMark> = None;
        let mut mwms: Vec<MoveWithMark> = vec![]; // used only if `config.show_logs`
        let legal_moves = MoveGen::new_legal(&game.current_position());
        for move_ in legal_moves {
            let board_possible: Board = game.current_position().make_move_new(move_);

            let score: float = analyze(board_possible, nn_to_make_move);

            let weight = if let Some((w_min, w_max)) = NN_RESULT_RANDOM_CHOICE { rng.gen_range(w_min..w_max) } else { 1. };

            let mwm_possible = MoveWithMark { move_, score, weight };
            if config.show_logs {
                mwms.push(mwm_possible);
            }

            omwm_best = match omwm_best {
                None => { // executes at first cycle of the loop, when `omwm_best` isn't set yet
                    Some(mwm_possible)
                }
                Some(/* mwm_best @ */ MoveWithMark { score, .. }) if score.is_nan() => { // executes if best is NaN
                    Some(mwm_possible)
                }
                Some(mwm_best) => {
                    // assert!(mwm_possible.score.is_finite()); // allowed to be ±infinite?
                    let best_is = if side_to_move == Color::White { Ordering::Greater } else { Ordering::Less };
                    match mwm_possible.get_weighted_score().partial_cmp(&mwm_best.get_weighted_score()) {
                        Some(ordering) if ordering == best_is => { // executes if new is better
                            Some(mwm_possible)
                        }
                        _ => omwm_best, // don't change best
                    }
                }
            };
        }

        if config.show_logs {
            mwms.sort_by(|mwm1, mwm2| mwm1.get_weighted_score().total_cmp(&mwm2.get_weighted_score()));
            for mwm in mwms {
                let MoveWithMark { move_, score, weight } = mwm;
                let weighted_score = mwm.get_weighted_score();
                // TODO: better formatting
                println!("{move_:<5} -> score = {score:<18} weight = {weight:<18} -> weighted_score = {weighted_score}");
            }
        }

        match omwm_best {
            Some(mwm_best) => {
                if config.show_logs {
                    println!("making move: {}", mwm_best.move_);
                }
                game.make_move(mwm_best.move_);
            }
            None => {
                if config.show_logs {
                    println!("game have ended, because no move can be made, i suppose");
                }
                break
            }
        }

        if game.can_declare_draw() {
            game.declare_draw();
        }

        maybe_print_position(game.current_position());
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

fn elo_rating_delta(x: float) -> float {
    100. / ( 1. + (10. as float).powf(x / 400.) )
}

const MOVES_NOT_PROVIDED: &str = "moves not provided";

struct PlayTournametConfig { print_games_results: bool, print_games: bool }
/// Plays tournament and sorts AIs.
fn play_tournament(ais: &mut Vec<AI>, config: PlayTournametConfig) {
    let ais_number = ais.len();

    let mut tournament_statistics: HashMap<Winner, u32> = HashMap::new();

    let nn_game_results: Vec<Option<Winner>> = (0..ais_number.pow(2)) // ij
        .collect::<Vec<usize>>()
        .into_par_iter()
        .map(|ij| (ij / ais_number, ij % ais_number))
        .map(|(i, j)| if i != j { Some((ais[i].nn.clone(), ais[j].nn.clone())) } else { None })
        .map(|option_ai_ij| {
            option_ai_ij.map(|(ai_i, ai_j)| {
                let game_res = play_game(&ai_i, &ai_j, PlayGameConfig::none());
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

            let winner: Winner = nn_game_results[i*ais_number+j].unwrap();
            if config.print_games_results {
                let winner_char: char = match winner {
                    Winner::White => 'W',
                    Winner::Black => 'B',
                    Winner::Draw => '.',
                };
                print!("{winner_char}");
            }

            let counter = tournament_statistics.entry(winner).or_insert(0);
            *counter += 1;

            (ais[i].rating, ais[j].rating) = updated_ratings(ais[i].rating, ais[j].rating, winner);
            // if SHOW_TRAINING_LOGS {
            //     println!("new ratings: i={}, j={}", player_i.rating, player_j.rating);
            //     println!();
            // }

            // players[i].rating = player_w.rating;
            // players[j].rating = player_b.rating;
        }
        if config.print_games_results {
            print!(" ");
        }
    }
    if config.print_games_results {
        println!();
    }

    // sort AIs:
    ais.sort_by(|ai1, ai2| ai2.rating.partial_cmp(&ai1.rating).unwrap());

    if config.print_games {
        println!("\nstats: {:?}", tournament_statistics);

        let ratings_sorted: Vec<float> = ais.iter().map(|p| p.rating).collect();
        print!("final ratings (sorted): [");
        for i in 0..ratings_sorted.len() {
            print!("{r:.2}", r=ratings_sorted[i]);
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
                    ..PlayGameConfig::none()
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
                    ..PlayGameConfig::none()
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
                    ..PlayGameConfig::none()
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
                    ..PlayGameConfig::none()
                }
            );
            println!(
                "WORST vs SELF: winner={winner:?}, moves: ' {moves} '\n",
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }
    }
}


fn updated_ratings(mut white_rating: float, mut black_rating: float, winner: Winner) -> (float, float) {
    // const WINNER_SCALE: float = 1.;
    const LOSE_SCALE: float = 1.;
    const DRAW_SCALE_STRONGER: float = 5.;
    const DRAW_SCALE_WEAKER  : float = 5.;
    let delta_rating_w = elo_rating_delta(white_rating - black_rating);
    let delta_rating_b = elo_rating_delta(black_rating - white_rating);
    match winner {
        Winner::White => {
            white_rating += delta_rating_w;
            black_rating -= delta_rating_w / LOSE_SCALE;
        }
        Winner::Black => {
            white_rating -= delta_rating_b / LOSE_SCALE;
            black_rating += delta_rating_b;
        }
        Winner::Draw => {
            // let delta_rating_min: float = delta_rating_w.min(delta_rating_b);
            let delta_rating_max: float = delta_rating_w.max(delta_rating_b);
            match white_rating.partial_cmp(&black_rating).unwrap() {
                Ordering::Greater => {
                    white_rating -= delta_rating_max / DRAW_SCALE_STRONGER;
                    black_rating += delta_rating_max / DRAW_SCALE_WEAKER;
                }
                Ordering::Less => {
                    white_rating += delta_rating_max / DRAW_SCALE_WEAKER;
                    black_rating -= delta_rating_max / DRAW_SCALE_STRONGER;
                }
                _ => {}
            }
        }
    }
    (white_rating, black_rating)
}

