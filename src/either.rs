//! Either type.

#![expect(dead_code)]


pub enum Either<L, R> {
	Left(L),
	Right(R),
}

trait LT {}
trait RT: !LT {}
// impl<L: !RT> LT for L {}
// impl<L> !RT for L {}
// impl<R: !LT> RT for R {}
// impl<R> !LT for R {}

impl<L: LT, R: RT> From<L> for Either<L, R> {
	fn from(l: L) -> Self {
		Self::Left(l)
	}
}
impl<L: LT, R: RT> From<R> for Either<L, R> {
	fn from(r: R) -> Self {
		Self::Right(r)
	}
}

// impl<L: LT, R: RT> Into<Either<L, R>> for L {
//     fn into(self) -> Either<L, R> {
//         Either::Left(self)
//     }
// }
// impl<L: LT, R: RT> Into<Either<L, R>> for R {
//     fn into(self) -> Either<L, R> {
//         Either::Right(self)
//     }
// }



#[test]
fn left_from() {
	assert_eq!(
		Either::<String, i32>::Left("hello".to_string()),
		Either::<String, i32>::from("hello".to_string())
	);
}

#[test]
fn left_into() {
	assert_eq!(
		Either::<String, i32>::Left("hello".to_string()),
		"hello".to_string().into()
	);
}

#[test]
fn right_from() {
	assert_eq!(
		Either::<String, i32>::Right(42),
		Either::<String, i32>::from(42)
	);
}

#[test]
fn right_into() {
	assert_eq!(
		Either::<String, i32>::Right(42),
		42.into()
	);
}

