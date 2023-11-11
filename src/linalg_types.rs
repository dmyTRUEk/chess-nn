//! Linear algebra types

use nalgebra::{DMatrix, RowDVector};

use crate::float_type::float;


// pub type ColVector = DVector<float>;
pub type RowVector = RowDVector<float>;

pub type Matrix = DMatrix<float>;

