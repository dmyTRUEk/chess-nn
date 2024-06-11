//! Input/Output Utils


pub fn flush() {
	use std::io::{Write, stdout};
	stdout().flush().unwrap();
}

pub fn print_and_flush<T: std::fmt::Display>(t: T) {
	print!("{t}");
	flush();
}

pub fn println<T: std::fmt::Display>(t: T) {
	println!("{t}");
}

pub fn prompt(s: &str) -> String {
	print_and_flush(s);
	read_line()
}

pub fn read_line() -> String {
	use std::io::stdin;
	let mut line: String = String::new();
	stdin().read_line(&mut line).unwrap();
	line.trim().to_string()
}

pub fn wait_for_enter() {
	// No need to "optimize" it be reimplementing without `.trim()` and `.to_string()`
	// bc it's anyway waiting for user to press enter.
	let _ = read_line();
}

