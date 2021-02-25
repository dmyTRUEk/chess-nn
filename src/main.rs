pub mod neural_network;                                                                       
pub mod neuron;
pub mod activation_functions;
pub mod random;

use std::io::Write;

use chess::*;

use crate::neural_network::*;



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



fn analyze(board: Board, mut nn: NeuralNetwork) -> f32 {
    let mut array_board: Vec<f32> = vec![0.0; 64];
    let mut n: usize = 0;
    for (_i, c) in board_to_human_viewable(board, false).to_string().chars().enumerate() {
        // println!("i = {}, c = '{}'", i, c);
        let value: f32 = match c {
            'p' => -1.0,
            'n' => -2.0,
            'b' => -3.0,
            'r' => -5.0,
            'q' => -6.0,
            'k' => -7.0,

            'P' => 1.0,
            'N' => 2.0,
            'B' => 3.0,
            'R' => 5.0,
            'Q' => 6.0,
            'K' => 7.0,

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



enum EnumWhoWon {
    White,
    Black,
    Draw,
}

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

fn play_game (nn_white: NeuralNetwork, nn_black: NeuralNetwork, show_log: bool) -> (EnumWhoWon, String) {
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

            let mark_possible: f32 = analyze(board_possible, nn_white.clone());
            let mwm_possible = MoveWithMark{chess_move: move_possible, mark: mark_possible};
            if show_log {
                println!("{} -> {}", mwm_possible.chess_move, mwm_possible.mark);
            }

            match omwm_best {
                None => {
                    omwm_best = Some(mwm_possible);
                },
                Some(ref mwm_best) => {
                    if mwm_possible.mark > mwm_best.mark {
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

        if show_log {
            println!("move_n = {}", move_n);
            println!("{}", board_to_human_viewable(game.current_position(), true));
            // println!("{}", game.current_position());
            // println!("{:?}", game);
            // wait_for_enter();
        }

    }

    if move_n >= MOVES_LIMIT {
        if show_log {
            println!("game result: {}+ moves", MOVES_LIMIT);
            println!("so winner will be calculated by pieces");
        }
        let mut piece_sum_white: f32 = 0.0;
        let mut piece_sum_black: f32 = 0.0;
        for (_i, c) in board_to_human_viewable(game.current_position(), false).to_string().chars().enumerate() {
            match c {
                'p' => piece_sum_black += 1.0,
                'n' => piece_sum_black += 2.0,
                'b' => piece_sum_black += 3.0,
                'r' => piece_sum_black += 5.0,
                'q' => piece_sum_black += 6.0,
                'k' => piece_sum_black += 7.0,

                'P' => piece_sum_white += 1.0,
                'N' => piece_sum_white += 2.0,
                'B' => piece_sum_white += 3.0,
                'R' => piece_sum_white += 5.0,
                'Q' => piece_sum_white += 6.0,
                'K' => piece_sum_white += 7.0,

                _ => {
                    continue;
                }
            }
        }
        if piece_sum_white > piece_sum_black {
            // return EnumWhoWon::White;
            return (EnumWhoWon::White, actions_to_string(game.actions().to_vec()));
        }
        else if piece_sum_black > piece_sum_white {
            // return EnumWhoWon::Black;
            return (EnumWhoWon::Black, actions_to_string(game.actions().to_vec()));
        }
        else {
            // return EnumWhoWon::Draw;
            return (EnumWhoWon::Draw, actions_to_string(game.actions().to_vec()));
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
                return (EnumWhoWon::White, actions_to_string(game.actions().to_vec()));
            },
            GameResult::WhiteResigns | GameResult::BlackCheckmates => {
                // return EnumWhoWon::Black;
                return (EnumWhoWon::Black, actions_to_string(game.actions().to_vec()));
            },
            GameResult::Stalemate | GameResult::DrawAccepted | GameResult::DrawDeclared => {
                // return EnumWhoWon::Draw;
                return (EnumWhoWon::Draw, actions_to_string(game.actions().to_vec()));
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

#[derive(Debug, Clone)]
struct Player {
    pub nn: NeuralNetwork,
    pub rating: f32,
}

fn logistic (x: f32) -> f32 {
    100.0 / ( 1.0 + 10.0_f32.powf(x/400.0) )
}

fn play_tournament (nns_white: Vec<NeuralNetwork>, nns_black: Vec<NeuralNetwork>, loops_amount: u32, show_log: bool) -> (NeuralNetwork, NeuralNetwork) {
    let default_rating: f32 = 1000.0;
    let mut players_white: Vec<Player> = nns_white.clone().into_iter().map(|nn| Player{nn: nn, rating: default_rating}).collect();
    let mut players_black: Vec<Player> = nns_black.clone().into_iter().map(|nn| Player{nn: nn, rating: default_rating}).collect();

    let game_n_max = players_white.clone().len() * players_black.clone().len();
    let mut game_n = 0;

    for loop_n in 1..=loops_amount {
        if show_log {
            print!("loop {}: ", loop_n);
            std::io::stdout().flush().unwrap();
        }
        for player_white in &mut players_white {
            println!();
            for player_black in &mut players_black {
                if show_log {
                    game_n += 1;
                    print!("  {}/{}: ", game_n, game_n_max);
                }

                let game_res = play_game(
                    player_white.nn.clone(),
                    player_black.nn.clone(),
                    false
                );

                // let delta_rating_white = logistic(player_black.rating - player_white.rating);
                // let delta_rating_black = logistic(player_white.rating - player_black.rating);
                let delta_rating_1 = logistic(player_black.rating - player_white.rating);
                let delta_rating_2 = logistic(player_white.rating - player_black.rating);
                // let delta_rating_bigger = max(delta_rating_1, delta_rating_2);
                // let delta_rating_less   = min(delta_rating_1, delta_rating_2);
                
                match game_res.0 {
                    EnumWhoWon::White => {
                        if show_log {
                            print!("White won! ");
                            std::io::stdout().flush().unwrap();
                            // println!("White won! ");
                            // println!("old ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                        player_white.rating += delta_rating_2;
                        player_black.rating -= delta_rating_2;
                        if show_log {
                            // println!("new ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                    },
                    EnumWhoWon::Black => {
                        if show_log {
                            print!("Black won! ");
                            std::io::stdout().flush().unwrap();
                            // println!("Black won! ");
                            // println!("old ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                        player_white.rating -= delta_rating_1;
                        player_black.rating += delta_rating_1;
                        if show_log {
                            // println!("new ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                    },
                    EnumWhoWon::Draw => {
                        if show_log {
                            print!("Draw! ");
                            std::io::stdout().flush().unwrap();
                            // println!("Draw! ");
                            // println!("old ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                        if player_white.rating > player_black.rating {
                            player_white.rating -= delta_rating_2 / 10.0;
                            player_black.rating += delta_rating_2 / 10.0;
                        }
                        else if player_black.rating > player_white.rating {
                            player_white.rating += delta_rating_2 / 10.0;
                            player_black.rating -= delta_rating_2 / 10.0;
                        }
                        else {  // equal
                            // nothing
                        }
                        if show_log {
                            // println!("new ratings: white={}, black={}", player_white.rating, player_black.rating);
                        }
                    }
                }
                if show_log {
                    // println!();
                }
            }
        }

        if show_log && loops_amount > 1 {
            let ratings_white: Vec<f32> = players_white.clone().into_iter().map(|p| p.rating).collect();
            let ratings_black: Vec<f32> = players_black.clone().into_iter().map(|p| p.rating).collect();
            println!("\ncurrent ratings: \nplayers_white: {:?}\nplayers_black: {:?}\n", ratings_white, ratings_black);
        }
    }

    // sort players:
    let (players_white_sorted, players_black_sorted): (Vec<Player>, Vec<Player>) = {
        let mut players_white_sorted = players_white.clone();
        let mut players_black_sorted = players_black.clone();
        players_white_sorted.sort_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap());
        players_black_sorted.sort_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap());
        (players_white_sorted, players_black_sorted)
    };
    let player_white_best: Player = players_white_sorted[0].clone();
    let player_black_best: Player = players_black_sorted[0].clone();

    if show_log {
        // let ratings_white: Vec<f32> = players_white.clone().into_iter().map(|p| p.rating).collect();
        // let ratings_black: Vec<f32> = players_black.clone().into_iter().map(|p| p.rating).collect();

        // println!("final ratings: \nplayers_white: {:?}\nplayers_black: {:?}\n", ratings_white, ratings_black);
        
        let ratings_white_sorted: Vec<f32> = players_white_sorted.clone().into_iter().map(|p| p.rating).collect();
        let ratings_black_sorted: Vec<f32> = players_black_sorted.clone().into_iter().map(|p| p.rating).collect();

        println!("\nfinal ratings (sorted): \nplayers_white: {:?}\nplayers_black: {:?}",
            ratings_white_sorted,
            ratings_black_sorted
        );

        let (_who_won, game_moves): (EnumWhoWon, String) = play_game(player_white_best.nn.clone(), player_black_best.nn.clone(), false);
        println!("game_moves of best NNs: '{}'", game_moves);
    }

    (player_white_best.nn, player_black_best.nn)
}



fn main () {
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 100, 1];
    let nn_heights: Vec<usize> = vec![64, 60, 40, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 20, 20, 20, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 100, 1];
    // let nn_heights: Vec<usize> = vec![64, 10, 1];
    // let nn_heights: Vec<usize> = vec![64, 1];
    
    assert!(nn_heights.len() >= 2, "nn_heights.len()={}, should be >= 2, else its useless", nn_heights.len());
    assert!(nn_heights[0] == 64, "nn_heights[0]={}, should be == 64, else its impossible", nn_heights[0]);
    assert!(nn_heights[nn_heights.len()-1] == 1, "nn_heights[last]={}, should be == 1, else its useless", nn_heights[nn_heights.len()-1]);

    let weight_min: f32 = -1.0;
    let weight_max: f32 = 1.0;

    let players_amount: usize = 5;
    assert!(players_amount > 1, "players_amount={} should be > 1, else its useless", players_amount);

    let mut nns_white_old: Vec<NeuralNetwork>;
    let mut nns_black_old: Vec<NeuralNetwork>;

    let mut nns_white: Vec<NeuralNetwork> = (0..players_amount).map(|_i| create_nn_with_random_weights(&nn_heights.clone(), weight_min, weight_max)).collect();
    let mut nns_black: Vec<NeuralNetwork> = (0..players_amount).map(|_i| create_nn_with_random_weights(&nn_heights.clone(), weight_min, weight_max)).collect();

    let generations: u32 = 1000;


    for generation in 0..=generations {
        println!("generation: {} / {}", generation, generations);

        let loops_amount: u32 = 1;
        let (nn_white_best, nn_black_best): (NeuralNetwork, NeuralNetwork) = play_tournament(
            nns_white.clone(), nns_black.clone(),
            loops_amount,
            true
        );

        // println!("nn_white_best = {}", nn_white_best);
        // println!("nn_black_best = {}", nn_black_best);

        // assert_eq!(nns_white.len(), nns_black.len());
        
        nns_white_old = nns_white.clone();
        nns_black_old = nns_black.clone();

        for i in 0..players_amount {
            nns_white[i] = nn_white_best.clone();
            nns_black[i] = nn_black_best.clone();
        }

        fn generation_to_evolve_factor (gen: u32, gens: u32) -> f32 {
            // ( -(gen as f32) / (gens as f32).sqrt() ).exp()
            // ( -(gen as f32) / (gens as f32).powf(0.8) ).exp()
            // ( -(gen as f32) / (gens as f32) ).exp()
            // ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
            0.3 * ( - 3.0 * (gen as f32) / (gens as f32) ).exp()
            // 0.3 * ( -(gen as f32) / (gens as f32) ).exp()
            // 0.1 * ( -(gen as f32) / (gens as f32) ).exp()
        }
        let evolution_factor: f32 = generation_to_evolve_factor(generation, generations);

        println!("evolving with evolution_factor = {}%", 100.0*evolution_factor);

        for i in 1..players_amount {
            nns_white[i].evolve(evolution_factor);
            nns_black[i].evolve(evolution_factor);
        }

        // if generation % 10 == 0 {
        //     for i in 0..players_amount {
        //         println!("nns_white[{}] = {}", i, nns_white[i]);
        //         println!("nns_black[{}] = {}", i, nns_black[i]);
        //     }
        // }

        assert_ne!(nns_white_old, nns_white);
        assert_ne!(nns_black_old, nns_black);

        if nns_white_old[0] == nns_white[0] {
            println!("white new best is same");
        }
        if nns_black_old[0] == nns_black[0] {
            println!("black new best is same");
        }

        // for i in 0..players_amount {
        //     println!("nns_white[{}] = {}", i, nns_white[i]);
        //     println!("nns_black[{}] = {}", i, nns_black[i]);
        // }

        // for i in 0..players_amount {
        //     for j in 0..players_amount {
        //         println!("i={}, j={}", i, j);
        //         if i == j {
        //             assert_eq!(nns_white[i], nns_white[j]);
        //             assert_eq!(nns_black[i], nns_black[j]);
        //             continue;
        //         }
        //         assert_ne!(nns_white[i], nns_white[j]);
        //         assert_ne!(nns_black[i], nns_black[j]);
        //     }
        // }

        println!("\n");

    }

}



