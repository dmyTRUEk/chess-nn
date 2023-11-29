//! Linear algebra types

use nalgebra::DMatrix;

use crate::float_type::float;


// Defined separately in `neural_network_{row/col}`.
// pub type ColVector = DVector<float>;
// pub type RowVector = RowDVector<float>;

pub type Matrix = DMatrix<float>;

