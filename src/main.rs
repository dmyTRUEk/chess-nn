/// Main file

pub mod utils_io;
pub mod neural_network;
pub mod activation_functions;

use std::collections::HashMap;

use chess::*;
use arrayfire::{device_count, device_info, info, set_device};
use rand::{Rng, prelude::ThreadRng, thread_rng};

use crate::{
    utils_io::*,
    neural_network::*,
    // activation_functions::get_random_activation_function,
};



const PLAYERS_AMOUNT: usize = 20;

const GENERATIONS: u32 = 1000;

const ALLOW_WIN_BY_POINTS: bool = false;

// additional neuron for choosing move a bit random
const USE_NOISE: bool = false;

const MOVES_LIMIT: u32 = 200;

pub enum ComputingUnit {
    CPU,
    GPU,
}
pub const COMPUTING_UNIT: ComputingUnit = ComputingUnit::CPU;



fn main() {
    // let nn_heights: Vec<usize> = vec![64, 700, 600, 500, 400, 300, 200, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 100, 100, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 60, 40, 20, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1000, 1000, 1000, 1];
    // let nn_heights: Vec<usize> = vec![64, 200, 200, 200, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 1];
    let nn_heights: Vec<usize> = vec![64, 60, 40, 20, 1];
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
        if !USE_NOISE { 64 } else { 65 },
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

    let (weight_min, weight_max): (f32, f32) = (-0.5, 0.5);
    let (consts_min, consts_max): (f32, f32) = (-0.5, 0.5);

    let mut nns: Vec<NeuralNetwork> = (0..PLAYERS_AMOUNT)
        .map(|_| NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max)).collect();
        // .map(|_| NeuralNetwork::with_smart_random(&nn_heights)).collect();
        // .map(|_i| NeuralNetwork::with_const_weights(&nn_heights, 1.0)).collect();
    let mut nns_old: Vec<NeuralNetwork>;
    let mut new_best_same_counter: u32 = 0;

    for generation in 0..=GENERATIONS {
        println!("generation: {generation} / {GENERATIONS}");

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
            println!("approx neurons_to_evolve = {approx_neurons_to_evolve}");

            // first part is best nns so dont evolve them, but second part will be evolved
            const SAVE_BEST_N: usize = 1 + PLAYERS_AMOUNT / 4;
            for i in SAVE_BEST_N..PLAYERS_AMOUNT {
                nns[i] = nns[i%SAVE_BEST_N].clone();
                nns[i].evolve(evolution_factor);
            }
            let len = nns.len();
            // nns[len-2] = NeuralNetwork::with_consts(&nn_heights, 0.01, 0.0, get_random_activation_function());
            // nns[len-1] = NeuralNetwork::with_smart_random(&nn_heights);
            nns[len-1] = NeuralNetwork::with_random(&nn_heights, weight_min, weight_max, consts_min, consts_max);
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
}





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

fn board_to_vec_for_nn(board: &Board) -> Vec<f32> {
    let mut input_for_nn: Vec<f32> = vec![0.0; if !USE_NOISE { 64 } else { 65 }];
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
            _ => { panic!() }
        }
        n += 1;
    }
    // input_for_nn[64] = match board.side_to_move() {
    //     Color::White => { 1.0 }
    //     Color::Black => { -1.0 }
    // };
    if USE_NOISE {
        let mut rng: ThreadRng = thread_rng();
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

fn analyze(board: &Board, nn: &NeuralNetwork) -> f32 {
    let input_for_nn: Vec<f32> = board_to_vec_for_nn(board);
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



fn play_game(
    nn_white: &NeuralNetwork,
    nn_black: &NeuralNetwork,
    show_log: bool,
    get_game: bool
) -> (EnumWhoWon, Option<String>) {
    // assert_eq!(nn_white.weight[0].len(), if !USE_NOISE { 64 } else { 65 });
    // assert_eq!(nn_black.weight[0].len(), if !USE_NOISE { 64 } else { 65 });

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

            let mark_possible: f32 = analyze(
                &board_possible,
                match side_to_move {
                    Color::White => { &nn_white }
                    Color::Black => { &nn_black }
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
            println!("moves_amount = {moves_amount}");
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
            // i->w->white, j->b->black
            // let mut player_w: Player = players[i].clone();
            // let mut player_b: Player = players[j].clone();

            let game_res: (EnumWhoWon, Option<String>) = play_game(
                &players[i].nn,
                &players[j].nn,
                false,
                false
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
                        // print_and_flush("White won! ");
                        print_and_flush("W");
                        flush();
                    }
                    players[i].rating += delta_rating_w;
                    players[j].rating -= delta_rating_w;
                    // player_j.rating -= delta_rating_2 / 4.0;
                    // player_i.rating += 10.0;
                    // player_j.rating -= 10.0;
                }
                EnumWhoWon::Black => {
                    if show_log {
                        // print_and_flush("Black won! ");
                        print_and_flush("B");
                        flush();
                    }
                    players[i].rating -= delta_rating_b;
                    // player_i.rating += delta_rating_1 / 4.0;
                    players[j].rating += delta_rating_b;
                    // player_i.rating -= 10.0;
                    // player_j.rating += 10.0;
                }
                EnumWhoWon::Draw => {
                    if show_log {
                        // print_and_flush("Draw! ");
                        print_and_flush("D");
                        flush();
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
                        // print_and_flush("WhiteByPoints won! ");
                        print_and_flush("w");
                        flush();
                    }
                    players[i].rating += delta_rating_w / 20.0;
                    players[j].rating -= delta_rating_w / 20.0;
                    // player_i.rating += 3.0;
                    // player_j.rating -= 3.0;
                }
                EnumWhoWon::BlackByPoints => {
                    if show_log {
                        // print_and_flush("BlackByPoints won! ");
                        print_and_flush("b");
                        flush();
                    }
                    players[i].rating -= delta_rating_b / 20.0;
                    players[j].rating += delta_rating_b / 20.0;
                    // player_i.rating -= 3.0;
                    // player_j.rating += 3.0;
                }
                EnumWhoWon::DrawByPoints => {
                    if show_log {
                        // print_and_flush("DrawByPoints! ");
                        print_and_flush("d");
                        flush();
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
            let (who_won, game_moves) = play_game(&players[0].nn, &players[0].nn, false, true);
            println!(
                "BEST vs SELF: winner={who_won:?}, af1={:?}, moves: ' {} '\n",
                players[0].nn.get_activation_function(),
                game_moves.unwrap(),
            );
        }

        {
            let (who_won, game_moves) = play_game(&players[0].nn, &players[1].nn, false, true);
            println!(
                "BEST vs BEST2: winner={who_won:?}, af2={:?}, moves: ' {} '\n",
                players[1].nn.get_activation_function(),
                game_moves.unwrap(),
            );
        }

        {
            let (who_won, game_moves) = play_game(&players[0].nn, &players.last().unwrap().nn, false, true);
            println!(
                "BEST vs WORST: winner={who_won:?}, af2={:?}, moves: ' {} '\n",
                players.last().unwrap().nn.get_activation_function(),
                game_moves.unwrap()
            );
        }
    }

    players_sorted.into_iter().map(|p| p.nn).collect()
}

