pub mod neural_network;                                                                       
pub mod neuron;
pub mod activation_functions;
pub mod random;

use std::io::Write;
use std::collections::HashMap;

use chess::*;

use crate::neural_network::*;
// use crate::random::*;



// #[warn(non_camel_case_types)]
// type myfloat = f32;



const FEN_INIT_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";



fn fen_to_human_viewable (fen: String, beautiful_output: bool) -> String {
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
                    for _j in 0..v {
                        // print!(". ");
                        res += &". ".to_string();
                    }
                    res = res[0..res.len()-1].to_string();
                },
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
                },
            }
            // print!(" ");
            res += &" ".to_string();
        }
    }
    // println!();
    return res;
}

fn fen_to_human_viewable_with_fen (fen: String, beautiful_output: bool) -> String {
    fen.clone() + &fen_to_human_viewable(fen, beautiful_output)
}

fn board_to_human_viewable (board: Board, beautiful_output: bool) -> String {
    // board_to_string(Board::from_fen(fen).unwrap(), beautiful_output)
    fen_to_human_viewable(board.to_string(), beautiful_output)
}

fn board_to_human_viewable_with_fen (board: Board, beautiful_output: bool) -> String {
    board.to_string() + &board_to_human_viewable(board, beautiful_output)
}


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
    pawn: 1.0,
    knight: 2.5,
    bishop: 3.0,
    rook: 5.0,
    queen: 7.0,
    king: 10.0,
};

fn analyze(board: Board, mut nn: NeuralNetwork) -> f32 {
    let mut array_board: Vec<f32> = vec![0.0; 64];
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
    // println!("array_board = {:?}", array_board);
    return nn.process_input(&array_board)[0];
}



fn wait_for_enter () {
    let mut line: String = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}



#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum EnumWhoWon {
    White,
    Black,
    Draw,
}
// impl PartialEq for EnumWhoWon {
//     fn eq (&self, other: &Self) -> bool {
        
//     }
// }

struct MoveWithMark {
    pub chess_move: ChessMove,
    pub mark: f32,
}

fn make_move (board: Board, chess_move: ChessMove) -> Board {
    let mut board_res: Board = board;
    board.make_move(chess_move, &mut board_res);
    board_res
}

fn actions_to_string (actions: Vec<Action>) -> String {
    let mut res: String = "".to_string();
    for action in actions {
        match action {
            Action::MakeMove(chess_move) => {
                res += &chess_move.to_string();
            },
            Action::OfferDraw(color) => {
                res += &format!("offer_draw_{:?}", color);
            },
            Action::AcceptDraw => {
                res += "accept_draw";
            },
            Action::DeclareDraw => {
                res += "declare_draw";
            },
            Action::Resign(color) => {
                res += &format!("resign_{:?}", color);
            },
        }
        res += " ";
    }
    res
}



const MOVES_LIMIT: u32 = 150;

fn play_game (nn_white: NeuralNetwork, nn_black: NeuralNetwork, show_log: bool, get_game: bool) -> (EnumWhoWon, Option<String>) {
    assert_eq!(nn_white.neurons[0].len(), 64);
    assert_eq!(nn_black.neurons[0].len(), 64);

    let mut move_n: u32 = 0;

    // let mut board_now = Board::default();
    // let mut game = Game::new_with_board(Board::from_fen(FEN_INIT_POSITION.to_string()).unwrap());
    let mut game = Game::new();

    // println!("{}", board_to_string_with_fen(board_now, true));
    if show_log {
        println!("{}", game.current_position());
    }

    while game.result() == None && move_n < MOVES_LIMIT {
        move_n += 1;

        let moves = MoveGen::new_legal(&game.current_position());
        let mut omwm_best: Option<MoveWithMark> = None;

        for move_possible in moves {
            let board_possible: Board = make_move(game.current_position(), move_possible);
            
            // println!("{}", board_to_string_with_fen(board_possible, false));
            
            let side_to_move: Color = board_possible.side_to_move();

            let mark_possible: f32 = match side_to_move {
                Color::White => {
                    analyze(board_possible, nn_white.clone())
                },
                Color::Black => {
                    analyze(board_possible, nn_black.clone())
                }
            };
            let mwm_possible = MoveWithMark{chess_move: move_possible, mark: mark_possible};
            if show_log {
                println!("{} -> {}", mwm_possible.chess_move, mwm_possible.mark);
            }

            match omwm_best {
                None => {
                    omwm_best = Some(mwm_possible);
                },
                Some(ref mwm_best) => {
                    if (mwm_possible.mark - mwm_best.mark) * match side_to_move {Color::White=>{1.0}, Color::Black=>{-1.0}} > 0.0 {
                        omwm_best = Some(mwm_possible);
                    }
                }
            }
            // println!("\n\n");
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
            },
            None => {
                if show_log {
                    println!("game have ended, because no move can be made, i suppose");
                }
                break;
            }
        }

        if game.can_declare_draw() {
            // game.declare_draw();
            move_n = MOVES_LIMIT;
            break;
        }

        if show_log {
            println!("move_n = {}", move_n);
            println!("{}", board_to_human_viewable(game.current_position(), true));
            // println!("{}", game.current_position());
            // println!("{:?}", game);
            // wait_for_enter();
        }

    }

    let create_game_str_if_needed = || {
        if !get_game {
            None
        } else {
            Some(actions_to_string(game.actions().to_vec()))
        }
    };

    if move_n >= MOVES_LIMIT {
        if show_log {
            println!("game result: {}+ moves", MOVES_LIMIT);
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
            return (EnumWhoWon::White, create_game_str_if_needed());
        }
        else if piece_sum_black > piece_sum_white {
            // return EnumWhoWon::Black;
            return (EnumWhoWon::Black, create_game_str_if_needed() );
        }
        else {
            // return EnumWhoWon::Draw;
            return (EnumWhoWon::Draw, create_game_str_if_needed() );
        }
    }
    else {
        let game_res: GameResult = game.result().unwrap();
        if show_log {
            println!("game result: {:?}", game_res);
        }
        match game_res {
            GameResult::WhiteCheckmates | GameResult::BlackResigns => {
                // return EnumWhoWon::White;
                return (EnumWhoWon::White, create_game_str_if_needed() );
            },
            GameResult::WhiteResigns | GameResult::BlackCheckmates => {
                // return EnumWhoWon::Black;
                return (EnumWhoWon::Black, create_game_str_if_needed() );
            },
            GameResult::Stalemate | GameResult::DrawAccepted | GameResult::DrawDeclared => {
                // return EnumWhoWon::Draw;
                return (EnumWhoWon::Draw, create_game_str_if_needed() );
            }
        }
    }
}



// fn min (a: f32, b: f32) -> f32 {
//     if a < b { a } else { b }
// }
// fn max (a: f32, b: f32) -> f32 {
//     if a > b { a } else { b }
// }
// fn sort_asc (v: Vec<f32>) -> Vec<f32> {
//     let mut res_v = v;
//     res_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
//     res_v
// }
// fn sort_desc (v: Vec<f32>) -> Vec<f32> {
//     let mut res_v = v;
//     res_v.sort_by(|a, b| b.partial_cmp(a).unwrap());
//     res_v
// }

#[derive(Clone, Debug)]
struct Player {
    pub nn: NeuralNetwork,
    pub rating: f32,
}

// impl PartialEq for Player {
//     fn eq (&self, other: &Self) -> bool {
//         return false;
//     }
// }
// impl Eq for Player {}

fn logistic (x: f32) -> f32 {
    100.0 / ( 1.0 + 10.0_f32.powf(x/400.0) )
}

fn play_tournament (nns: Vec<NeuralNetwork>, loops_amount: u32, show_log: bool) -> Vec<NeuralNetwork> {
    let default_rating: f32 = 1000.0;
    let mut players: Vec<Player> = nns.into_iter().map(|nn| Player{nn: nn, rating: default_rating}).collect();
    let players_amount: usize = players.len();

    let game_n_max = 2 * players_amount * (players_amount - 1) / 2;
    let mut game_n = 0;
    let mut tournament_statistics: HashMap<EnumWhoWon, u32> = HashMap::new();

    for loop_n in 1..=loops_amount {
        if show_log && loops_amount > 1 {
            print!("loop {} / {}: ", loop_n, loops_amount);
            std::io::stdout().flush().unwrap();
        }
        for i in 0..players_amount {
            // println!();
            for j in 0..players_amount {
                if i == j { continue; }
                if show_log {
                    game_n += 1;
                    // print!("  {}/{}: ", game_n, game_n_max);
                }

                let mut player_i: Player = players[i].clone();
                let mut player_j: Player = players[j].clone();

                let game_res = play_game(player_i.nn.clone(), player_j.nn.clone(), false, false);
                let game_res_who_won: EnumWhoWon = game_res.0;

                let delta_rating_1 = logistic(player_j.rating - player_i.rating);
                let delta_rating_2 = logistic(player_i.rating - player_j.rating);

                // if show_log {
                //     println!("old ratings: i={}, j={}", player_i.rating, player_j.rating);
                // }
                
                let counter = tournament_statistics.entry(game_res_who_won.clone()).or_insert(0);
                *counter += 1;

                // match tournament_statistics.get(&game_res_who_won) {
                //     None => {
                //         // tournament_statistics[&game_res_who_won] = 0;
                //         tournament_statistics.insert(game_res_who_won.clone(), 0);
                //     },
                //     Some(val) => {
                //         val += 1;
                //     }
                // }
                
                match game_res_who_won {
                    EnumWhoWon::White => {
                        if show_log {
                            // print!("White won! ");
                            print!("W");
                            std::io::stdout().flush().unwrap();
                        }
                        player_i.rating += delta_rating_2;
                        player_j.rating -= delta_rating_2;
                    },
                    EnumWhoWon::Black => {
                        if show_log {
                            // print!("Black won! ");
                            print!("B");
                            std::io::stdout().flush().unwrap();
                        }
                        player_i.rating -= delta_rating_1;
                        player_j.rating += delta_rating_1;
                    },
                    EnumWhoWon::Draw => {
                        if show_log {
                            // print!("Draw! ");
                            print!("D");
                            std::io::stdout().flush().unwrap();
                        }
                        if player_i.rating > player_j.rating {
                            player_i.rating -= delta_rating_2 / 10.0;
                            player_j.rating += delta_rating_2 / 10.0;
                        }
                        else if player_j.rating > player_i.rating {
                            player_i.rating += delta_rating_2 / 10.0;
                            player_j.rating -= delta_rating_2 / 10.0;
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
        }

        if show_log && loops_amount > 1 {
            let ratings: Vec<f32> = players.clone().iter().map(|p| p.rating).collect();
            println!("\ncurrent ratings: \nplayers: {:?}\n", ratings);
        }
    }

    // sort players:
    let players_sorted: Vec<Player> = {
        let mut players_sorted = players.clone();
        players_sorted.sort_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap());
        players_sorted
    };

    if show_log {
        println!("\n\nstats: {:?}", tournament_statistics);

        let ratings_sorted: Vec<f32> = players_sorted.clone().iter().map(|p| p.rating).collect();
        println!("\nfinal ratings (sorted): {:?}", ratings_sorted);

        let player_best: Player = players_sorted[0].clone();
        let (who_won, game_moves) = play_game(player_best.nn.clone(), player_best.nn.clone(), false, true);
        println!("\ngame_moves of best NNs: '{}', winner={:?}\n", game_moves.unwrap(), who_won);
    }

    // player_best.nn
    players_sorted.into_iter().map(|p| p.nn).collect()
}



fn main () {
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 1];
    let nn_heights: Vec<usize> = vec![64, 60, 40, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 20, 20, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 1000, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1];
    
    assert!(nn_heights.len() >= 2, "nn_heights.len()={}, should be >= 2, else its useless", nn_heights.len());
    assert!(nn_heights[0] == 64, "nn_heights[0]={}, should be == 64, else its impossible", nn_heights[0]);
    assert!(nn_heights[nn_heights.len()-1] == 1, "nn_heights[last]={}, should be == 1, else its useless", nn_heights[nn_heights.len()-1]);

    let weight_min: f32 = -1.0;
    let weight_max: f32 = 1.0;

    let players_amount: usize = 20;
    assert!(players_amount > 1, "players_amount={} should be > 1, else its useless", players_amount);

    // let mut nns: Vec<NeuralNetwork> = (0..players_amount).map(|_i| create_nn_with_const_weights(&nn_heights, 1.0)).collect();
    let mut nns: Vec<NeuralNetwork> = (0..players_amount).map(|_i| create_nn_with_random_weights(&nn_heights, weight_min, weight_max)).collect();
    let mut nns_old: Vec<NeuralNetwork>;
    let mut new_best_same_counter: u32 = 0;

    let generations: u32 = 1000;


    for generation in 0..=generations {
        println!("generation: {} / {}", generation, generations);

        nns_old = nns.clone();

        let loops_amount: u32 = 1;
        nns = play_tournament(
            nns,
            loops_amount,
            true
        );

        // println!("nn_best = {}", nns[0]);

        if generation < generations {
            fn generation_to_evolve_factor (gen: u32, gens: u32) -> f32 {
                // ( -(gen as f32) / (gens as f32).sqrt() ).exp()
                // ( -(gen as f32) / (gens as f32).powf(0.8) ).exp()
                // ( -(gen as f32) / (gens as f32) ).exp()
                // ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
                // 0.3 * ( -(gen as f32) / (gens as f32) ).exp()
                // 0.3 * ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
                // 0.1 * ( -(gen as f32) / (gens as f32) ).exp()
                0.1 * ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
            }
            let evolution_factor: f32 = generation_to_evolve_factor(generation, generations);
            println!("evolving with evolution_factor = {}%", 100.0*evolution_factor);

            // first part is best nns so dont evolve them, but second part will be evolved
            for i in (players_amount/3).max(1)..players_amount {
                nns[i] = nns[0].clone();
                nns[i].evolve(evolution_factor);
            }
        }

        // if generation % 10 == 0 {
        //     for i in 0..players_amount {
        //         println!("nns[{}] = {}", i, nns[i]);
        //     }
        // }

        // assert_ne!(nns_old, nns);

        if nns_old[0] == nns[0] && generation > 0 {
            new_best_same_counter += 1;
            println!("CAUTION: NEW BEST IS SAME {} TIMES!!!", new_best_same_counter);
        }
        else {
            new_best_same_counter = 0;
        }

        // for i in 0..players_amount {
        //     println!("nns[{}] = {}", i, nns[i]);
        // }

        // for i in 0..players_amount {
        //     for j in 0..players_amount {
        //         println!("i={}, j={}", i, j);
        //         if i == j {
        //             assert_eq!(nns[i], nns[j]);
        //             continue;
        //         }
        //         assert_ne!(nns[i], nns[j]);
        //     }
        // }

        println!("\n");

    }

    println!("evolution finished successfuly!");
    println!("here is final nns:");
    for i in 0..players_amount {
        println!("nns[{}] = {}\n\n", i, nns[i]);
    }

    println!("best_nn = {}\n\n", nns[0]);

}



