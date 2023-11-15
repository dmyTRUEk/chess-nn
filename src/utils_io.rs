//! Input/Output Utils

use std::io::Write;


pub fn flush() {
    std::io::stdout().flush().unwrap();
}

pub fn print_and_flush<T: std::fmt::Display>(t: T) {
    print!("{t}");
    flush();
}


pub fn read_line() -> String {
    let mut line: String = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    line.trim().to_string()
}

pub fn wait_for_enter() {
    let _ = read_line();
}

pub fn prompt(s: &str) -> String {
    print_and_flush(s);
    read_line()
}

