/// Main file

pub mod utils_io;
pub mod random;
pub mod activation_functions;
pub mod neural_network;

use std::collections::HashMap;

use chess::*;
use arrayfire::{info, set_device, device_count, device_info};

use crate::utils_io::*;
use crate::random::*;
use crate::activation_functions::get_random_activation_function;
use crate::neural_network::*;



const PLAYERS_AMOUNT: usize = 20;

const GENERATIONS: u32 = 1000;

const ALLOW_WIN_BY_POINTS: bool = false;

// the 65's neuron needed for choosing move a bit random
const USE_65_NEURONS: bool = false;

const MOVES_LIMIT: u32 = 200;

pub enum ComputingUnit {
    CPU,
    GPU,
}
pub const COMPUTING_UNIT: ComputingUnit = ComputingUnit::CPU;



fn main() {
    // let nn_heights: Vec<usize> = vec![64, 700, 600, 500, 400, 300, 200, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 100, 100, 100, 1];
    let nn_heights: Vec<usize> = vec![64, 80, 60, 40, 20, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1000, 1000, 1000, 1];
    // let nn_heights: Vec<usize> = vec![64, 200, 200, 200, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 60, 40, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 20, 20, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 200, 200, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 10, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1000, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1];

    let nn_heights: Vec<usize> = {
        let mut nn_heights_new: Vec<usize> = nn_heights;
        if USE_65_NEURONS {
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
        if !USE_65_NEURONS { 64 } else { 65 },
        "nn_heights[0]={}, should be == 65, else its impossible", nn_heights[0]
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

    let (weight_min, weight_max): (f32, f32) = (-0.05, 0.05);
    let (consts_min, consts_max): (f32, f32) = (-0.05, 0.05);

    let mut nns: Vec<NeuralNetwork> = (0..PLAYERS_AMOUNT)
        .map(|_| NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max)).collect();
        // .map(|_i| NeuralNetwork::with_const_weights(&nn_heights, 1.0)).collect();
    let mut nns_old: Vec<NeuralNetwork>;
    let mut new_best_same_counter: u32 = 0;

    for generation in 0..=GENERATIONS {
        println!("generation: {} / {}", generation, GENERATIONS);

        nns_old = nns.clone();

        nns = play_tournament(nns, true);

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
            println!("approx neurons_to_evolve = {}", approx_neurons_to_evolve);

            // first part is best nns so dont evolve them, but second part will be evolved
            const SAVE_BEST_N: usize = 1 + PLAYERS_AMOUNT / 3;
            for i in SAVE_BEST_N..PLAYERS_AMOUNT {
                nns[i] = nns[i%SAVE_BEST_N].clone();
                nns[i].evolve(evolution_factor);
            }
            let len = nns.len();
            nns[len-2] = NeuralNetwork::with_consts(&nn_heights, 0.01, 0.0, get_random_activation_function());
            nns[len-1] = NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max);
        }

        if nns_old[0] == nns[0] && generation > 0 {
            new_best_same_counter += 1;
            println!("CAUTION: NEW BEST IS SAME {} TIMES!!!", new_best_same_counter);
        }
        else {
            new_best_same_counter = 0;
        }

        println!("\n");

    }

    println!("evolution finished successfuly!");

    // println!("best_nn = {}\n\n", nns[0]);
}





fn fen_to_human_viewable(fen: String, beautiful_output: bool) -> String {
    let mut res: String = "".to_string();
    // println!("{}", board);
    // res += &board.to_string();
    res += &"\n".to_string();

    for (_i, c) in fen.chars().enumerate() {
        if c == ' ' {
            break;
        }
        else if c == '/' {
            // println!();
            res += &"\n".to_string();
        }
        else {
            match c.to_string().parse::<u32>() {
                Ok(v) => {
                    for _ in 0..v {
                        // print!(". ");
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
                        // print!("{}", beautiful_c);
                        res += &beautiful_c.to_string();
                    }
                    else {
                        // print!("{}", c);
                        res += &c.to_string();
                    }
                }
            }
            // print!(" ");
            res += &" ".to_string();
        }
    }
    // println!();
    return res;
}

// fn fen_to_human_viewable_with_fen(fen: String, beautiful_output: bool) -> String {
//     fen.clone() + &fen_to_human_viewable(fen, beautiful_output)
// }

// TODO: make rid of it
fn board_to_human_viewable(board: Board, beautiful_output: bool) -> String {
    // board_to_string(Board::from_fen(fen).unwrap(), beautiful_output)
    fen_to_human_viewable(board.to_string(), beautiful_output)
}

// fn board_to_human_viewable_with_fen(board: Board, beautiful_output: bool) -> String {
//     board.to_string() + &board_to_human_viewable(board, beautiful_output)
// }


// fn reverse_colors_on_board(board_in: Board) -> Board {
//     let mut fen_board_out: String = "".to_string();

//     let board_in_string = board_in.to_string();

//     let splited = board_in_string.split(" ").collect::<Vec<&str>>();

//     let mut fen_left: String = splited[0].to_string();
//     let mut fen_right: String = splited[1..].join(" ").to_string();
//     // println!("{}", fen_left);
//     // println!("{}", fen_right);

//     let mut fen_left_new: String = "".to_string();
//     for c in fen_left.chars().into_iter() {
//         if c.is_lowercase() {
//             fen_left_new += &c.to_ascii_uppercase().to_string();
//         }
//         else if c.is_uppercase() {
//             fen_left_new += &c.to_ascii_lowercase().to_string();
//         }
//         else {
//             fen_left_new += &c.to_string();
//         }
//     }
//     fen_left_new = fen_left_new.chars().rev().collect::<String>();
//     // println!("{}", fen_left_new);
//     // println!("{}", fen_right);

//     fen_board_out = fen_left_new + " " + &fen_right;

//     println!("{}", fen_board_out);
//     return Board::from_fen(fen_board_out).unwrap();
//     // return Board::from_fen(board_in.to_string()).unwrap();
// }



// enum PiecesValue {
//     Pawn = 1.0,
//     Knight = 2.5,
//     Bishop = 3.0,
//     Rook = 5.0,
//     Queen = 7.0,
//     King = 10.0,
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
    // pawn: 1.0,
    // knight: 2.7,
    // bishop: 3.0,
    // rook: 5.0,
    // queen: 9.0,
    // king: 100.0,
    pawn: 0.01,
    knight: 0.02,
    bishop: 0.03,
    rook: 0.04,
    queen: 0.05,
    king: 0.06,
};

fn analyze(board: Board, nn: NeuralNetwork) -> f32 {
    let mut array_board: Vec<f32> = vec![0.0; if !USE_65_NEURONS { 64 } else { 65 }];
    let mut n: usize = 0;
    for (_i, c) in board_to_human_viewable(board, false).to_string().chars().enumerate() {
        // println!("i = {}, c = '{}'", i, c);
        let value: f32 = match c {
            'p' => -PIECES_VALUE.pawn,
            'n' => -PIECES_VALUE.knight,
            'b' => -PIECES_VALUE.bishop,
            'r' => -PIECES_VALUE.rook,
            'q' => -PIECES_VALUE.queen,
            'k' => -PIECES_VALUE.king,

            'P' => PIECES_VALUE.pawn,
            'N' => PIECES_VALUE.knight,
            'B' => PIECES_VALUE.bishop,
            'R' => PIECES_VALUE.rook,
            'Q' => PIECES_VALUE.queen,
            'K' => PIECES_VALUE.king,

            '.' => 0.0,
            _ => {
                continue;
            }
        };
        // println!("adding symbol: '{}' -> {}", c, value);
        array_board[n] = value;
        n += 1;
    }
    if USE_65_NEURONS {
        array_board[64] = random_f32(-10.0, 10.0);
    }
    // println!("array_board = {:?}", array_board);
    return nn.process_input(&array_board)[0];
}



#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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
    res
}



fn play_game(
    nn_white: NeuralNetwork, 
    nn_black: NeuralNetwork, 
    show_log: bool, 
    get_game: bool
) -> (EnumWhoWon, Option<String>) {
    // assert_eq!(nn_white.weight[0].len(), if !USE_65_NEURONS { 64 } else { 65 });
    // assert_eq!(nn_black.weight[0].len(), if !USE_65_NEURONS { 64 } else { 65 });

    let mut moves_amount: u32 = 0;

    // let mut board_now = Board::default();
    // let mut game = Game::new_with_board(Board::from_fen(FEN_INIT_POSITION.to_string()).unwrap());
    let mut game = Game::new();

    // println!("{}", board_to_string_with_fen(board_now, true));
    if show_log {
        println!("{}", game.current_position());
    }

    struct MoveWithMark {
        pub chess_move: ChessMove,
        pub mark: f32,
    }

    while game.result() == None && moves_amount < 2*MOVES_LIMIT {
        moves_amount += 1;

        let possible_moves = MoveGen::new_legal(&game.current_position());
        let mut omwm_best: Option<MoveWithMark> = None;

        let side_to_move: Color = game.current_position().side_to_move();

        for move_ in possible_moves {
            let board_possible: Board = game.current_position().make_move_new(move_);

            // println!("{}", board_to_string_with_fen(board_possible, false));

            // let side_to_move: Color = board_possible.side_to_move();

            // let mark_possible: f32 = match side_to_move {
            //     Color::White => {
            //         analyze(board_possible, nn_white.clone())
            //     }
            //     Color::Black => {
            //         analyze(board_possible, nn_black.clone())
            //     }
            // };

            let mark_possible: f32 = analyze(
                board_possible,
                match side_to_move {
                    Color::White => { nn_white.clone() }
                    Color::Black => { nn_black.clone() }
                }
            );
            let mwm_possible = MoveWithMark{chess_move: move_, mark: mark_possible};
            if show_log {
                println!("{} -> {}", mwm_possible.chess_move, mwm_possible.mark);
            }

            match omwm_best {
                None => {
                    omwm_best = Some(mwm_possible);
                }
                Some(ref mwm_best) => {
                    let sign = match side_to_move {
                        Color::White => { 1.0 }
                        Color::Black => { -1.0 }
                    };
                    if sign * (mwm_possible.mark - mwm_best.mark) > 0.0 {
                        omwm_best = Some(mwm_possible);
                    }
                }
            }
        }

        match omwm_best {
            Some(mwm_best) => {
                // board_now = make_move(board_now, mwm_best.chess_move);
                // println!("{}", game.current_position());

                if show_log {
                    println!("making move: {}", mwm_best.chess_move);
                }
                game.make_move(mwm_best.chess_move);

                // println!("{}", game.current_position());
            }
            None => {
                if show_log {
                    println!("game have ended, because no move can be made, i suppose");
                }
                break;
            }
        }

        if game.can_declare_draw() {
            if ALLOW_WIN_BY_POINTS {
                moves_amount = 2*MOVES_LIMIT;
                break;
            }
            else {
                game.declare_draw();
            }
        }

        if show_log {
            println!("move_n = {}", moves_amount);
            println!("{}", board_to_human_viewable(game.current_position(), true));
            // println!("{}", game.current_position());
            // println!("{:?}", game);
            // wait_for_enter();
        }

    }

    let create_game_str_if_needed = || {
        if get_game {
            Some(actions_to_string(game.actions().to_vec()))
        } else {
            None
        }
    };

    if moves_amount < 2*MOVES_LIMIT {   // true victory/lose:
        let game_res: GameResult = game.result().unwrap();
        if show_log {
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
        if show_log {
            println!("game result: {}+ moves", 2*MOVES_LIMIT);
            println!("so winner will be calculated by pieces");
        }
        let mut piece_sum_white: f32 = 0.0;
        let mut piece_sum_black: f32 = 0.0;
        for (_i, c) in board_to_human_viewable(game.current_position(), false).to_string().chars().enumerate() {
            match c {
                'p' => piece_sum_black += PIECES_VALUE.pawn,
                'n' => piece_sum_black += PIECES_VALUE.knight,
                'b' => piece_sum_black += PIECES_VALUE.bishop,
                'r' => piece_sum_black += PIECES_VALUE.rook,
                'q' => piece_sum_black += PIECES_VALUE.queen,
                // 'k' => piece_sum_black += PIECES_VALUE.king,

                'P' => piece_sum_white += PIECES_VALUE.pawn,
                'N' => piece_sum_white += PIECES_VALUE.knight,
                'B' => piece_sum_white += PIECES_VALUE.bishop,
                'R' => piece_sum_white += PIECES_VALUE.rook,
                'Q' => piece_sum_white += PIECES_VALUE.queen,
                // 'K' => piece_sum_white += PIECES_VALUE.king,

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

fn play_tournament(nns: Vec<NeuralNetwork>, show_log: bool) -> Vec<NeuralNetwork> {
    const DEFAULT_RATING: f32 = 1000.0;
    let mut players: Vec<Player> = nns.into_iter().map(|nn| Player{nn, rating: DEFAULT_RATING}).collect();

    let mut tournament_statistics: HashMap<EnumWhoWon, u32> = HashMap::new();

    for i in 0..PLAYERS_AMOUNT {
        for j in 0..PLAYERS_AMOUNT {
            if i == j { continue; }
            let mut player_i: Player = players[i].clone();
            let mut player_j: Player = players[j].clone();

            let game_res = play_game(player_i.nn.clone(), player_j.nn.clone(), false, false);
            let game_res_who_won: EnumWhoWon = game_res.0;

            let delta_rating_1 = logistic(player_j.rating - player_i.rating);
            let delta_rating_2 = logistic(player_i.rating - player_j.rating);

            let counter = tournament_statistics.entry(game_res_who_won.clone()).or_insert(0);
            *counter += 1;

            match game_res_who_won {
                EnumWhoWon::White => {
                    if show_log {
                        // print!("White won! ");
                        print!("W");
                        flush();
                    }
                    player_i.rating += delta_rating_2;
                    // player_j.rating -= delta_rating_2;
                    player_j.rating += delta_rating_2 / 4.0;
                    // player_i.rating += 10.0;
                    // player_j.rating -= 10.0;
                }
                EnumWhoWon::Black => {
                    if show_log {
                        // print!("Black won! ");
                        print!("B");
                        flush();
                    }
                    // player_i.rating -= delta_rating_1;
                    player_i.rating += delta_rating_1 / 4.0;
                    player_j.rating += delta_rating_1;
                    // player_i.rating -= 10.0;
                    // player_j.rating += 10.0;
                }
                EnumWhoWon::Draw => {
                    if show_log {
                        // print!("Draw! ");
                        print!("D");
                        flush();
                    }
                    if player_i.rating > player_j.rating {
                        player_i.rating -= delta_rating_2 / 20.0;
                        player_j.rating += delta_rating_2 / 20.0;
                        // player_i.rating -= 1.0;
                        // player_j.rating += 1.0;
                    }
                    else if player_j.rating > player_i.rating {
                        player_i.rating += delta_rating_2 / 20.0;
                        player_j.rating -= delta_rating_2 / 20.0;
                        // player_i.rating += 1.0;
                        // player_j.rating -= 1.0;
                    }
                    else { // equal
                        // nothing
                    }
                }
                EnumWhoWon::WhiteByPoints => {
                    if show_log {
                        // print!("WhiteByPoints won! ");
                        print!("w");
                        flush();
                    }
                    player_i.rating += delta_rating_2 / 20.0;
                    player_j.rating -= delta_rating_2 / 20.0;
                    // player_i.rating += 3.0;
                    // player_j.rating -= 3.0;
                }
                EnumWhoWon::BlackByPoints => {
                    if show_log {
                        // print!("BlackByPoints won! ");
                        print!("b");
                        flush();
                    }
                    player_i.rating -= delta_rating_1 / 20.0;
                    player_j.rating += delta_rating_1 / 20.0;
                    // player_i.rating -= 3.0;
                    // player_j.rating += 3.0;
                }
                EnumWhoWon::DrawByPoints => {
                    if show_log {
                        // print!("DrawByPoints! ");
                        print!("d");
                        flush();
                    }
                    if player_i.rating > player_j.rating {
                        player_i.rating -= delta_rating_2 / 20.0;
                        player_j.rating += delta_rating_2 / 20.0;
                        // player_i.rating -= 0.3;
                        // player_j.rating += 0.3;
                    }
                    else if player_j.rating > player_i.rating {
                        player_i.rating += delta_rating_2 / 20.0;
                        player_j.rating -= delta_rating_2 / 20.0;
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

            players[i].rating = player_i.rating;
            players[j].rating = player_j.rating;
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

        let ratings_sorted: Vec<f32> = players_sorted.clone().iter().map(|p| p.rating).collect();
        println!("final ratings (sorted): {:?}", ratings_sorted);

        {
            let player_best: Player = players_sorted[0].clone();
            let (who_won, game_moves) = play_game(player_best.nn.clone(), player_best.nn.clone(), false, true);
            println!(
                "BEST vs SELF: winner={who_won:?}, af={:?}, moves: ' {}'",
                player_best.nn.get_activation_function(),
                game_moves.unwrap(),
            );
        }

        {
            let player_best_1: Player = players_sorted[0].clone();
            let player_best_2: Player = players_sorted[1].clone();
            let (who_won, game_moves) = play_game(player_best_1.nn.clone(), player_best_2.nn.clone(), false, true);
            println!(
                "BEST vs BEST2: winner={who_won:?}, af1={:?}, af2={:?}, moves: ' {}'",
                player_best_1.nn.get_activation_function(),
                player_best_2.nn.get_activation_function(),
                game_moves.unwrap(),
            );
        }

        {
            let player_best: Player = players_sorted[0].clone();
            let player_worst: Player = players_sorted[players_sorted.len()-1].clone();
            let (who_won, game_moves) = play_game(player_best.nn.clone(), player_worst.nn.clone(), false, true);
            println!(
                "BEST vs WORST: winner={who_won:?}, af1={:?}, af2={:?}, moves: ' {}'",
                player_best.nn.get_activation_function(),
                player_worst.nn.get_activation_function(),
                game_moves.unwrap()
            );
        }
    }

    players_sorted.into_iter().map(|p| p.nn).collect()
}


