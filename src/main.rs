#![feature(test)]

/// Main file

extern crate test;

pub mod utils_io;
pub mod neural_network;
pub mod activation_functions;

use std::{collections::HashMap, str::FromStr};

use chess::*;
use arrayfire::{device_count, device_info, info, set_device};
use rand::{Rng, prelude::ThreadRng, thread_rng};

use crate::{
    utils_io::*,
    neural_network::*,
    // activation_functions::get_random_activation_function,
};



const PLAYERS_AMOUNT: usize = 10;

const GENERATIONS: u32 = 10;
const GEN_TO_START_WATCHING: u32 = 300;

const ALLOW_WIN_BY_POINTS: bool = true;

// additional neuron for choosing move a bit random
const USE_NOISE: bool = false;
const NEURONS_IN_FIRST_LAYER: usize = if !USE_NOISE { 64 } else { 65 };

const PLAY_WITH_NN_AFTER_TRAINING: bool = false;

const MOVES_LIMIT: u32 = 300;

pub enum ComputingUnit {
    CPU,
    GPU,
}
// TODO: make this compile time check and then dont use `arrayfire` at all
pub const COMPUTING_UNIT: ComputingUnit = ComputingUnit::CPU;
// pub const COMPUTING_UNIT: ComputingUnit = ComputingUnit::GPU;



fn main() {
    let nn_heights: Vec<usize> = vec![
        // 64, 700, 600, 500, 400, 300, 200, 100, 1
        // 64, 100, 100, 100, 100, 100, 100, 1
        // 64, 60, 40, 20, 10, 1
        // 64, 10000, 10000, 10000, 1
        // 64, 1000, 1000, 1000, 1
        // 64, 200, 200, 200, 1
        // 64, 100, 100, 100, 1
        64, 300, 70, 20, 1
        // 64, 60, 40, 20, 1
        // 64, 20, 20, 20, 1
        // 64, 15, 20, 11, 1,
        // 64, 200, 200, 1
        // 64, 100, 100, 1
        // 64, 10, 10, 1
        // 64, 1000, 1
        // 64, 100, 1
        // 64, 10, 1
        // 64, 1
    ];
    // TODO: refactor so i dont have to write 1 at the end

    let nn_heights: Vec<usize> = {
        let mut nn_heights_new: Vec<usize> = nn_heights;
        if USE_NOISE {
            nn_heights_new[0] += 1;
        }
        nn_heights_new
    };

    assert!(
        nn_heights.len() >= 2,
        "nn_heights.len()={}, should be >= 2, else its useless", nn_heights.len()
    );
    assert_eq!(
        nn_heights[0],
        NEURONS_IN_FIRST_LAYER,
        "nn_heights[0]={}, should be == 64or65, else its impossible", nn_heights[0]
    );
    assert_eq!(
        nn_heights[nn_heights.len()-1],
        1,
        "nn_heights[last]={}, should be == 1, else its useless", nn_heights[nn_heights.len()-1]
    );
    assert!(
        PLAYERS_AMOUNT > 1,
        "PLAYERS_AMOUNT={} should be > 1, else its useless",
        PLAYERS_AMOUNT
    );

    match COMPUTING_UNIT {
        ComputingUnit::CPU => {
            println!("using computing unit: CPU");
        }
        ComputingUnit::GPU => {
            println!("using computing unit: GPU");
            println!("devices avalaible: {}", device_count());
            set_device(0);
            println!("{:?}", device_info());
            info();
        }
    }

    let mut rng: ThreadRng = thread_rng();

    let (weight_min, weight_max): (f32, f32) = (-5.0, 5.0);
    let (consts_min, consts_max): (f32, f32) = (-5.0, 5.0);

    let mut nns: Vec<NeuralNetwork> = (0..PLAYERS_AMOUNT)
        .map( |_i|
            NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max, &mut rng)
            // NeuralNetwork::with_consts(&nn_heights, 1.0, 1.0, activation_functions::ActivationFunction::Sigmoid))
            // NeuralNetwork::with_smart_random(&nn_heights, &mut rng))
        )
        .collect();
    let mut nns_old: Vec<NeuralNetwork>;
    let mut new_best_same_counter: u32 = 0;

    for generation in 0..=GENERATIONS {
        println!("generation: {generation} / {GENERATIONS}");

        nns_old = nns.clone();

        nns = play_tournament(nns, generation, &mut rng);

        if generation < GENERATIONS {
            fn generation_to_evolve_factor(gen: u32, gens: u32) -> f32 {
                // ( -(gen as f32) / (gens as f32).sqrt() ).exp()
                // ( -(gen as f32) / (gens as f32).powf(0.8) ).exp()
                // ( -(gen as f32) / (gens as f32) ).exp()
                // ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
                // 0.3 * ( -(gen as f32) / (gens as f32) ).exp()
                0.999 * ( -(gen as f32) / (gens as f32) ).exp()
                // 0.8 * ( - 5.0 * (gen as f32) / (gens as f32) ).exp()
                // 0.1 * ( -(gen as f32) / (gens as f32) ).exp()
                // 0.1 * ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
            }
            let evolution_factor: f32 = generation_to_evolve_factor(generation, GENERATIONS);
            println!("evolving with evolution_factor = {}%", 100.0*evolution_factor);

            let approx_neurons_to_evolve: f32 = evolution_factor*((sum_vec(&nn_heights)-nn_heights[0]) as f32);
            println!("approx neurons_to_evolve = {approx_neurons_to_evolve}");

            // first part is best nns so dont evolve them, but second part will be evolved
            const SAVE_BEST_N: usize = 1 + PLAYERS_AMOUNT / 4;
            for i in SAVE_BEST_N..PLAYERS_AMOUNT {
                nns[i] = nns[i%SAVE_BEST_N].clone();
                nns[i].evolve(evolution_factor, &mut rng);
            }
            let len = nns.len();
            // nns[len-2] = NeuralNetwork::with_consts(&nn_heights, 0.01, 0.0, get_random_activation_function());
            // nns[len-1] = NeuralNetwork::with_smart_random(&nn_heights);
            nns[len-1] = NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max, &mut rng);
        }

        if nns_old[0] == nns[0] && generation > 0 {
            new_best_same_counter += 1;
            println!("WARNING: new best is same {new_best_same_counter} times!!!");
        }
        else {
            new_best_same_counter = 0;
        }

        println!("\n");

    }

    println!("evolution finished successfuly!");

    // println!("best_nn = {}\n\n", nns[0]);

    if !PLAY_WITH_NN_AFTER_TRAINING {
        return;
    }

    loop {
        print_and_flush("Choose side to play (w/b): ");
        let line: String = read_line();
        let side_to_play: Color = match line.chars().nth(0).unwrap() {
            'w' => { Color::White }
            'b' => { Color::Black }
            // 'q' => { break; }
            _ => { continue; }
        };
        let (who_won, game_moves): (EnumWhoWon, Option<String>) = play_game(
            &nns[0],
            &nns[0],
            PlayGameConfig {
                get_game: true,
                show_log: true,
                wait_for_enter_after_every_move: false,
                human_color: Some(side_to_play),
            },
            &mut rng
        );
        println!(
            "{who_vs_who}: winner={who_won:?}, af={af:?}, moves: ' {moves} '\n",
            who_vs_who = match side_to_play {
                Color::White => { "HUMAN vs NN_BEST" }
                Color::Black => { "NN_BEST vs HUMAN" }
            },
            af = nns[0].get_activation_function(),
            moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
        );
    }
}



const MOVES_NOT_PROVIDED: &str = "moves not provided";

fn fen_to_human_viewable(fen: String, beautiful_output: bool) -> String {
    let mut res: String = "".to_string();
    res += &"\n".to_string();
    for (_i, c) in fen.chars().enumerate() {
        if c == ' ' {
            break;
        }
        else if c == '/' {
            res += &"\n".to_string();
        }
        else {
            match c.to_string().parse::<u32>() {
                Ok(v) => {
                    for _ in 0..v {
                        res += &". ".to_string();
                    }
                    res = res[0..res.len()-1].to_string();
                }
                Err(_e) => {
                    if beautiful_output {
                        /*
                        ♖♘♗♕♔♗♘♖
                        ♙♙♙♙♙♙♙♙
                        ♟♟♟♟♟♟♟♟
                        ♜♞♝♛♚♝♞♜
                        */
                        let beautiful_c: char = match c {
                            'r' => '♖',
                            'n' => '♘',
                            'b' => '♗',
                            'q' => '♕',
                            'k' => '♔',
                            'p' => '♙',
                            'R' => '♜',
                            'N' => '♞',
                            'B' => '♝',
                            'Q' => '♛',
                            'K' => '♚',
                            'P' => '♟',
                            _ => {
                                panic!();
                            }
                        };
                        res += &beautiful_c.to_string();
                    }
                    else {
                        res += &c.to_string();
                    }
                }
            }
            res += &" ".to_string();
        }
    }
    res
}

fn board_to_human_viewable(board: Board, beautiful_output: bool) -> String {
    let fen: String = board.to_string();
    fen_to_human_viewable(fen, beautiful_output)
}



// enum PiecesValue {
//     Pawn,
//     Knight,
//     Bishop,
//     Rook,
//     Queen,
//     King,
// }
// impl PiecesValue {
//     #[inline(always)]
//     fn value(&self) -> f32 {
//         match self {
//             PiecesValue::Pawn   => { 1.0 }
//             PiecesValue::Knight => { 2.5 }
//             PiecesValue::Bishop => { 3.0 }
//             PiecesValue::Rook   => { 5.0 }
//             PiecesValue::Queen  => { 7.0 }
//             PiecesValue::King   => { 20.0 }
//         }
//     }
// }

struct PiecesValue {
    pub pawn: f32,
    pub knight: f32,
    pub bishop: f32,
    pub rook: f32,
    pub queen: f32,
    pub king: f32,
}
const PIECES_VALUE: PiecesValue = PiecesValue {
    pawn:   1.0,
    knight: 2.7,
    bishop: 3.0,
    rook:   5.0,
    queen:  9.0,
    king:   20.0,
};

fn board_to_vec_for_nn(board: &Board, rng: &mut ThreadRng) -> Vec<f32> {
    let mut input_for_nn: Vec<f32> = vec![0.0; NEURONS_IN_FIRST_LAYER];
    let mut n: usize = 0;
    for c in board.to_string().chars() {
        match c {
            ' ' => { break }
            '/' => { continue }
            '1' => { n += 0 }
            '2' => { n += 1 }
            '3' => { n += 2 }
            '4' => { n += 3 }
            '5' => { n += 4 }
            '6' => { n += 5 }
            '7' => { n += 6 }
            '8' => { n += 7 }
            'p' => { input_for_nn[n] = -PIECES_VALUE.pawn }
            'n' => { input_for_nn[n] = -PIECES_VALUE.knight }
            'b' => { input_for_nn[n] = -PIECES_VALUE.bishop }
            'r' => { input_for_nn[n] = -PIECES_VALUE.rook }
            'q' => { input_for_nn[n] = -PIECES_VALUE.queen }
            'k' => { input_for_nn[n] = -PIECES_VALUE.king }
            'P' => { input_for_nn[n] = PIECES_VALUE.pawn }
            'N' => { input_for_nn[n] = PIECES_VALUE.knight }
            'B' => { input_for_nn[n] = PIECES_VALUE.bishop }
            'R' => { input_for_nn[n] = PIECES_VALUE.rook }
            'Q' => { input_for_nn[n] = PIECES_VALUE.queen }
            'K' => { input_for_nn[n] = PIECES_VALUE.king }

            // 'p' => { input_for_nn[n] = -PiecesValue::Pawn.value() }
            // 'n' => { input_for_nn[n] = -PiecesValue::Knight.value() }
            // 'b' => { input_for_nn[n] = -PiecesValue::Bishop.value() }
            // 'r' => { input_for_nn[n] = -PiecesValue::Rook.value() }
            // 'q' => { input_for_nn[n] = -PiecesValue::Queen.value() }
            // 'k' => { input_for_nn[n] = -PiecesValue::King.value() }
            // 'P' => { input_for_nn[n] = PiecesValue::Pawn.value() }
            // 'N' => { input_for_nn[n] = PiecesValue::Knight.value() }
            // 'B' => { input_for_nn[n] = PiecesValue::Bishop.value() }
            // 'R' => { input_for_nn[n] = PiecesValue::Rook.value() }
            // 'Q' => { input_for_nn[n] = PiecesValue::Queen.value() }
            // 'K' => { input_for_nn[n] = PiecesValue::King.value() }

            _ => { panic!() }
        }
        n += 1;
    }
    // input_for_nn[64] = match board.side_to_move() {
    //     Color::White => { 1.0 }
    //     Color::Black => { -1.0 }
    // };
    if USE_NOISE {
        input_for_nn[64] = rng.gen_range(-10.0..10.0);
    }
    if board.side_to_move() == Color::Black {
        input_for_nn.reverse();
        for i in 0..64 {
            input_for_nn[i] *= -1.0;
        }
    }
    input_for_nn
}

fn analyze(board: &Board, nn: &NeuralNetwork, rng: &mut ThreadRng) -> f32 {
    let input_for_nn: Vec<f32> = board_to_vec_for_nn(board, rng);
    // println!("input_for_nn = {:?}", array_board);
    nn.process_input(&input_for_nn)[0]
}



#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
enum EnumWhoWon {
    White,
    Black,
    Draw,
    WhiteByPoints,
    BlackByPoints,
    DrawByPoints,
}

fn actions_to_string(actions: Vec<Action>) -> String {
    let mut res: String = "".to_string();
    for action in actions {
        match action {
            Action::MakeMove(chess_move) => {
                res += &chess_move.to_string();
            }
            Action::OfferDraw(color) => {
                res += &format!("offer_draw_{:?}", color);
            }
            Action::AcceptDraw => {
                res += "accept_draw";
            }
            Action::DeclareDraw => {
                res += "declare_draw";
            }
            Action::Resign(color) => {
                res += &format!("resign_{:?}", color);
            }
        }
        res += " ";
    }
    res.pop();
    res
}



fn string_to_chess_move(line: String) -> Option<ChessMove> {
    let line: String = line[..line.len()-1].to_string();
    if line.len() < 4 || line.len() > 5 { return None; }
    let line: Vec<char> = line.chars().collect();
    Some(ChessMove::new(
        Square::make_square(
            if let Ok(rank) = Rank::from_str(&line[1].to_string()) { rank } else { return None; },
            if let Ok(file) = File::from_str(&line[0].to_string()) { file } else { return None; }
        ),
        Square::make_square(
            if let Ok(rank) = Rank::from_str(&line[3].to_string()) { rank } else { return None; },
            if let Ok(file) = File::from_str(&line[2].to_string()) { file } else { return None; }
        ),
        if line.len() == 4 { None } else {
            Some(match line[4] {
                'q' => { Piece::Queen }
                'r' => { Piece::Rook }
                'b' => { Piece::Bishop }
                'n' => { Piece::Knight }
                _ => { return None; }
            })
        }
    ))
}


struct PlayGameConfig {
    pub get_game: bool,
    pub show_log: bool,
    pub wait_for_enter_after_every_move: bool,
    pub human_color: Option<Color>,
}

fn play_game(
    nn_white: &NeuralNetwork,
    nn_black: &NeuralNetwork,
    config: PlayGameConfig,
    rng: &mut ThreadRng,
) -> (EnumWhoWon, Option<String>) {
    // assert_eq!(nn_white.weight[0].len(), NEURONS_IN_FIRST_LAYER);
    // assert_eq!(nn_black.weight[0].len(), NEURONS_IN_FIRST_LAYER);

    let mut moves_amount: u32 = 0;

    // let mut board_now = Board::default();
    // let mut game = Game::new_with_board(Board::from_fen(FEN_INIT_POSITION.to_string()).unwrap());
    let mut game = Game::new();

    // if config.show_log {
    //     println!("{}", game.current_position());
    // }

    #[derive(Debug, Copy, Clone)]
    struct MoveWithMark {
        pub chess_move: ChessMove,
        pub mark: f32,
    }

    while game.result() == None && moves_amount < 2*MOVES_LIMIT {
        moves_amount += 1;

        let possible_moves = MoveGen::new_legal(&game.current_position());
        let mut omwm_best: Option<MoveWithMark> = None;
        let mut mwms: Vec<MoveWithMark> = vec![];

        let side_to_move: Color = game.current_position().side_to_move();

        // if vs human
        // this if is nested, because at moment of writing this code it was unstable
        if let Some(human_color) = config.human_color { if human_color == side_to_move {
            let mut move_: Option<ChessMove> = None;
            while move_.is_none() {
                print_and_flush("Your move: ");
                let line: String = read_line();
                move_ = string_to_chess_move(line);
                if move_.is_some() && game.current_position().legal(move_.unwrap()) {
                    break;
                }
                println!("That move is illegal.");
            }
            game.make_move(move_.unwrap());
            continue;
        }}

        for move_ in possible_moves {
            let board_possible: Board = game.current_position().make_move_new(move_);

            // println!("{}", board_to_string_with_fen(board_possible, false));

            // let side_to_move: Color = board_possible.side_to_move();

            let mark_possible: f32 = analyze(
                &board_possible,
                match side_to_move {
                    Color::White => { &nn_white }
                    Color::Black => { &nn_black }
                },
                rng
            );
            let mwm_possible = MoveWithMark{chess_move: move_, mark: mark_possible};
            if config.show_log {
                // println!("{} -> {}", mwm_possible.chess_move, mwm_possible.mark);
                mwms.push(mwm_possible);
            }

            match omwm_best {
                None => {
                    omwm_best = Some(mwm_possible);
                }
                Some(ref mwm_best) => {
                    // TODO: remove this?, because:
                    // giving black_nn reversed input
                    // let sign = match side_to_move {
                    //     Color::White => { 1.0 }
                    //     Color::Black => { -1.0 }
                    // };
                    // if sign * (mwm_possible.mark - mwm_best.mark) > 0.0 {
                    if (mwm_possible.mark - mwm_best.mark) >= 0.0 {
                        omwm_best = Some(mwm_possible);
                    }
                }
            }
        }

        match omwm_best {
            Some(mwm_best) => {
                // board_now = make_move(board_now, mwm_best.chess_move);
                // println!("{}", game.current_position());

                if config.show_log {
                    println!("making move: {}", mwm_best.chess_move);
                }
                game.make_move(mwm_best.chess_move);

                // println!("{}", game.current_position());
            }
            None => {
                if config.show_log {
                    println!("game have ended, because no move can be made, i suppose");
                }
                break;
            }
        }

        if game.can_declare_draw() {
            if ALLOW_WIN_BY_POINTS {
                // moves_amount = 2*MOVES_LIMIT;
                break;
            }
            else {
                game.declare_draw();
            }
        }

        if config.show_log {
            mwms.sort_by(|mwm1, mwm2| mwm1.mark.partial_cmp(&mwm2.mark).unwrap());
            for mwm in mwms {
                println!("{move_} -> {mark:.2}", move_=mwm.chess_move.to_string(), mark=mwm.mark);
            }
            println!("moves_amount = {moves_amount}");
            println!("{}", board_to_human_viewable(game.current_position(), true));
            // println!("{}", game.current_position());
            // println!("{:?}", game);
            if config.wait_for_enter_after_every_move {
                wait_for_enter();
            }
        }

    }

    let create_game_str_if_needed = || {
        if config.get_game {
            Some(actions_to_string(game.actions().to_vec()))
        } else {
            None
        }
    };

    if moves_amount < 2*MOVES_LIMIT && !ALLOW_WIN_BY_POINTS {   // true victory/lose:
        let game_res: GameResult = game.result().unwrap();
        if config.show_log {
            println!("game result: {:?}", game_res);
        }
        match game_res {
            GameResult::WhiteCheckmates | GameResult::BlackResigns => {
                // return EnumWhoWon::White;
                return (EnumWhoWon::White, create_game_str_if_needed());
            }
            GameResult::WhiteResigns | GameResult::BlackCheckmates => {
                // return EnumWhoWon::Black;
                return (EnumWhoWon::Black, create_game_str_if_needed());
            }
            GameResult::Stalemate | GameResult::DrawAccepted | GameResult::DrawDeclared => {
                // return EnumWhoWon::Draw;
                return (EnumWhoWon::Draw, create_game_str_if_needed());
            }
        }
    }
    else {   // by points
        if config.show_log {
            println!("game result: true draw or {}+ moves", 2*MOVES_LIMIT);
            println!("so winner will be calculated by pieces");
        }
        let mut piece_sum_white: f32 = 0.0;
        let mut piece_sum_black: f32 = 0.0;
        for (_i, c) in board_to_human_viewable(game.current_position(), false).to_string().chars().enumerate() {
            match c {
                'p' => { piece_sum_black += PIECES_VALUE.pawn }
                'n' => { piece_sum_black += PIECES_VALUE.knight }
                'b' => { piece_sum_black += PIECES_VALUE.bishop }
                'r' => { piece_sum_black += PIECES_VALUE.rook }
                'q' => { piece_sum_black += PIECES_VALUE.queen }
                // 'k' => { piece_sum_black += PIECES_VALUE.king }
                'P' => { piece_sum_white += PIECES_VALUE.pawn }
                'N' => { piece_sum_white += PIECES_VALUE.knight }
                'B' => { piece_sum_white += PIECES_VALUE.bishop }
                'R' => { piece_sum_white += PIECES_VALUE.rook }
                'Q' => { piece_sum_white += PIECES_VALUE.queen }
                // 'K' => { piece_sum_white += PIECES_VALUE.king }

                // 'p' => { piece_sum_black += PiecesValue::Pawn.value() }
                // 'n' => { piece_sum_black += PiecesValue::Knight.value() }
                // 'b' => { piece_sum_black += PiecesValue::Bishop.value() }
                // 'r' => { piece_sum_black += PiecesValue::Rook.value() }
                // 'q' => { piece_sum_black += PiecesValue::Queen.value() }
                // // 'k' => { piece_sum_black += PiecesValue::King.value() }
                // 'P' => { piece_sum_white += PiecesValue::Pawn.value() }
                // 'N' => { piece_sum_white += PiecesValue::Knight.value() }
                // 'B' => { piece_sum_white += PiecesValue::Bishop.value() }
                // 'R' => { piece_sum_white += PiecesValue::Rook.value() }
                // 'Q' => { piece_sum_white += PiecesValue::Queen.value() }
                // // 'K' => { piece_sum_white += PiecesValue::King.value() }

                _ => { continue; }
            }
        }
        if piece_sum_white > piece_sum_black {
            // return EnumWhoWon::White;
            return (EnumWhoWon::WhiteByPoints, create_game_str_if_needed());
        }
        else if piece_sum_black > piece_sum_white {
            // return EnumWhoWon::Black;
            return (EnumWhoWon::BlackByPoints, create_game_str_if_needed());
        }
        else {
            // return EnumWhoWon::Draw;
            return (EnumWhoWon::DrawByPoints, create_game_str_if_needed());
        }
    }
}



fn sum_vec(vec: &Vec<usize>) -> usize {
    let mut res: usize = 0;
    for item in vec {
        res += item;
    }
    res
}



#[derive(Clone, Debug)]
struct Player {
    pub nn: NeuralNetwork,
    pub rating: f32,
}

fn logistic(x: f32) -> f32 {
    100.0 / ( 1.0 + 10.0_f32.powf(x/400.0) )
}

fn play_tournament(
    nns: Vec<NeuralNetwork>,
    gen: u32,
    rng: &mut ThreadRng
) -> Vec<NeuralNetwork> {
    const DEFAULT_RATING: f32 = 1000.0;
    let mut players: Vec<Player> = nns.into_iter().map(|nn| Player{nn, rating: DEFAULT_RATING}).collect();

    let mut tournament_statistics: HashMap<EnumWhoWon, u32> = HashMap::new();

    let show_log: bool = true;
    for i in 0..PLAYERS_AMOUNT {
        for j in 0..PLAYERS_AMOUNT {
            if i == j { continue; }
            // i->w->white, j->b->black
            // let mut player_w: Player = players[i].clone();
            // let mut player_b: Player = players[j].clone();

            let game_res: (EnumWhoWon, Option<String>) = play_game(
                &players[i].nn,
                &players[j].nn,
                PlayGameConfig {
                    get_game: false,
                    show_log: false,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                },
                rng
            );
            let game_res_who_won: EnumWhoWon = game_res.0;

            // if white wins
            let delta_rating_w: f32 = logistic(players[i].rating - players[j].rating);
            // if black wins
            let delta_rating_b: f32 = logistic(players[j].rating - players[i].rating);

            let counter = tournament_statistics.entry(game_res_who_won).or_insert(0);
            *counter += 1;

            match game_res_who_won {
                EnumWhoWon::White => {
                    if show_log {
                        print_and_flush("W");
                    }
                    players[i].rating += delta_rating_w;
                    // players[j].rating -= delta_rating_w;
                    players[j].rating -= delta_rating_w / 5.0;
                }
                EnumWhoWon::Black => {
                    if show_log {
                        print_and_flush("B");
                    }
                    // players[i].rating -= delta_rating_b;
                    players[i].rating -= delta_rating_b / 5.0;
                    players[j].rating += delta_rating_b;
                }
                EnumWhoWon::Draw => {
                    if show_log {
                        print_and_flush("D");
                    }
                    let delta_rating_min: f32 = delta_rating_w.min(delta_rating_b);
                    if players[i].rating > players[j].rating {
                        players[i].rating -= delta_rating_min / 3.0;
                        players[j].rating += delta_rating_min / 3.0;
                        // player_i.rating -= 1.0;
                        // player_j.rating += 1.0;
                    }
                    else if players[j].rating > players[i].rating {
                        players[i].rating += delta_rating_min / 3.0;
                        players[j].rating -= delta_rating_min / 3.0;
                        // player_i.rating += 1.0;
                        // player_j.rating -= 1.0;
                    }
                    else { // equal
                        // nothing
                    }
                }
                EnumWhoWon::WhiteByPoints => {
                    if show_log {
                        print_and_flush("w");
                    }
                    players[i].rating += delta_rating_w / 20.0;
                    players[j].rating -= delta_rating_w / 20.0;
                    // player_i.rating += 3.0;
                    // player_j.rating -= 3.0;
                }
                EnumWhoWon::BlackByPoints => {
                    if show_log {
                        print_and_flush("b");
                    }
                    players[i].rating -= delta_rating_b / 20.0;
                    players[j].rating += delta_rating_b / 20.0;
                    // player_i.rating -= 3.0;
                    // player_j.rating += 3.0;
                }
                EnumWhoWon::DrawByPoints => {
                    if show_log {
                        print_and_flush("d");
                    }
                    if players[i].rating > players[j].rating {
                        players[i].rating -= delta_rating_w / 20.0;
                        players[j].rating += delta_rating_w / 20.0;
                        // player_i.rating -= 0.3;
                        // player_j.rating += 0.3;
                    }
                    else if players[j].rating > players[i].rating {
                        players[i].rating += delta_rating_w / 20.0;
                        players[j].rating -= delta_rating_w / 20.0;
                        // player_i.rating += 0.3;
                        // player_j.rating -= 0.3;
                    }
                    else { // equal
                        // nothing
                    }
                }
            }
            // if show_log {
            //     println!("new ratings: i={}, j={}", player_i.rating, player_j.rating);
            //     println!();
            // }

            // players[i].rating = player_w.rating;
            // players[j].rating = player_b.rating;
        }
        print!(" ");
    }
    println!();

    // sort players:
    let players_sorted: Vec<Player> = {
        let mut players_sorted = players.clone();
        players_sorted.sort_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap());
        players_sorted
    };

    if show_log {
        println!("\nstats: {:?}", tournament_statistics);

        let ratings_sorted: Vec<f32> = players_sorted.iter().map(|p| p.rating).collect();
        print!("final ratings (sorted): [");
        for i in 0..ratings_sorted.len() {
            print!("{r:.0}", r=ratings_sorted[i]);
            if i != ratings_sorted.len()-1 {
                print!(", ");
            }
        }
        println!("]\n");

        {
            let (who_won, game_moves) = play_game(
                &players[0].nn,
                &players[0].nn,
                PlayGameConfig {
                    get_game: true,
                    show_log: gen >= GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                },
                rng
            );
            println!(
                "BEST vs SELF: winner={who_won:?}, af1={af:?}, moves: ' {moves} '\n",
                af = players[0].nn.get_activation_function(),
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }

        {
            let (who_won, game_moves) = play_game(
                &players[0].nn,
                &players[1].nn,
                PlayGameConfig {
                    get_game: true,
                    show_log: gen >= GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                },
                rng
            );
            println!(
                "BEST vs BEST2: winner={who_won:?}, af2={af:?}, moves: ' {moves} '\n",
                af = players[1].nn.get_activation_function(),
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }

        {
            let (who_won, game_moves) = play_game(
                &players[0].nn,
                &players.last().unwrap().nn,
                PlayGameConfig {
                    get_game: true,
                    show_log: gen >= GEN_TO_START_WATCHING,
                    wait_for_enter_after_every_move: false,
                    human_color: None,
                },
                rng
            );
            println!(
                "BEST vs WORST: winner={who_won:?}, af2={af:?}, moves: ' {moves} '\n",
                af = players.last().unwrap().nn.get_activation_function(),
                moves = game_moves.unwrap_or(MOVES_NOT_PROVIDED.to_string()),
            );
        }
    }

    players_sorted.into_iter().map(|p| p.nn).collect()
}




#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_main(b: &mut Bencher) {
        b.iter(|| {
            main();
        });
    }
}

