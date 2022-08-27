extern crate nalgebra as na;
use na::*;

use std::ops::AddAssign;
use std::{thread,time};


fn fill_matrix<T>(i: usize, j: usize) -> T
where
    T: From<i8>
{
    let mut res = T::from(0);
    if i+1 == j {
        res = T::from(1);
    }
    if j+1 == i {
        res = T::from(1);
    }
    res
}


fn pad_state(mut state: DMatrix<i32>) -> DMatrix<i32>
{
    // Pad left side
    // Remove columns if they are empty
    while state.slice((0,0),(state.nrows(),2)).sum() == 0 {
        state = state.remove_columns_at(&[0]);
    }
    // Add columns if there are lattice points near edge
    if state.slice((0,0),(state.nrows(),1)).sum() > 0 {
        state = state.insert_column(0, 0);
    }

    // Pad top side
    // Remove rows if they are empty
    while state.slice((0,0),(2,state.ncols())).sum() == 0 {
        state = state.remove_rows_at(&[0]);
    }
    // Add columns if there are lattice points near edge
    if state.slice((0,0),(1,state.ncols())).sum() > 0 {
        state = state.insert_row(0, 0);
    }

    // Pad right side
    // Remove columns if they are empty
    while state.slice((0,state.ncols()-2),(state.nrows(),2)).sum() == 0 {
        let n = state.ncols();
        state = state.remove_columns_at(&[n-1]);
    }
    // Add columns if there are lattice points near edge
    if state.slice((0,state.ncols()-1),(state.nrows(),1)).sum() > 0 {
        let n = state.ncols();
        state = state.insert_column(n, 0);
    }

    // Pad bottom side
    // Remove columns if they are empty
    while state.slice((state.nrows()-2,0),(2,state.ncols())).sum() == 0 {
        let n = state.nrows();
        state = state.remove_rows_at(&[n-1]);
    }
    // Add columns if there are lattice points near edge
    if state.slice((state.nrows()-1,0),(1,state.ncols())).sum() > 0 {
        let n = state.nrows();
        state = state.insert_row(n, 0);
    }

    state
}


fn main() {

    let mut state: DMatrix<i32> = dmatrix![
        0, 0, 0, 0, 0;
        0, 0, 1, 0, 0;
        0, 0, 1, 0, 1;
        0, 0, 1, 1, 0;
        0, 0, 0, 0, 0;
        0, 0, 0, 0, 0;
        0, 0, 0, 0, 0;
    ];

    let mut n;
    let mut m;

    let mut c;
    let mut d;
    let mut sum;

    let half_sec = time::Duration::from_millis(500);

    // Pad state
    state = pad_state(state);

    println!("{}", state);
    loop {
        // Do Conway iteration
        n = state.nrows();
        m = state.ncols();
        let d_n = DMatrix::<i32>::from_fn(n, n, &fill_matrix);
        let d_m = DMatrix::<i32>::from_fn(m, m, &fill_matrix);

        thread::sleep(half_sec);
        c = &d_n * &state;
        d = &state * &d_m;
        sum = c+d;
        
        let x_00 = state.slice((0, 0), (n-1, m-1));
        let x_01 = state.slice((0, 1), (n-1, m-1));
        let x_10 = state.slice((1, 0), (n-1, m-1));
        let x_11 = state.slice((1, 1), (n-1, m-1));
        
        sum.slice_mut((0, 0), (n-1, m-1)).add_assign(x_11);
        sum.slice_mut((0, 1), (n-1, m-1)).add_assign(x_10);
        sum.slice_mut((1, 0), (n-1, m-1)).add_assign(x_01);
        sum.slice_mut((1, 1), (n-1, m-1)).add_assign(x_00);

        for (si, cdi) in state.iter_mut().zip((sum).iter_mut()) {
            if *si == 1 && *cdi < 2 {
                *si = 0;
            }
            if *si == 1 && *cdi > 3 {
                *si = 0;
            }
            if *si == 0 && *cdi == 3 {
                *si = 1;
            }
        }
        
        // Pad state
        state = pad_state(state);

        println!("{}", state);
    }
}