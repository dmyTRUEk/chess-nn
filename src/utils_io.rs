// This file contains Input/Output Utils 

use std::io::Write;



pub fn wait_for_enter() {
    let mut line: String = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}



pub fn flush() {
    std::io::stdout().flush().unwrap();
}

