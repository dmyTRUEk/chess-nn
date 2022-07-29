// This file contains Input/Output Utils 

use std::io::Write;



pub fn wait_for_enter() {
    let mut line: String = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}



/// print and flush
pub fn print_and_flush<T: std::fmt::Display>(t: T) {
    print!("{t}");
    flush();
}

pub fn flush() {
    std::io::stdout().flush().unwrap();
}

