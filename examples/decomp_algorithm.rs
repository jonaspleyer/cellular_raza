use rayon::prelude::*;


fn get_decomp_res(n_voxel: usize, n_regions: usize) -> Option<(usize, usize, usize)> {
    // We calculate how many times we need to drain how many voxels
    // Example:
    //      n_voxels    = 59
    //      n_regions   = 6
    //      average_len = (59 / 8).ceil() = (9.833 ...).ceil() = 10
    //
    // try to solve this equation:
    //      n_voxels = average_len * n + (average_len-1) * m
    //      where n,m are whole positive numbers
    //
    // We start with    n = n_regions = 6
    // and with         m = min(0, n_voxel - average_len.pow(2)) = min(0, 59 - 6^2) = 23
    let mut average_len: i64 = (n_voxel as f64 / n_regions as f64).ceil() as i64;

    let residue = |n: i64, m: i64, avg: i64| {n_voxel as i64 - avg*n - (avg-1)*m};

    let mut n = n_regions as i64;
    let mut m = 0;

    for _ in 0..n_regions {
        let r = residue(n, m, average_len);
        if r == 0 {
            return Some((n as usize, m as usize, average_len as usize));
        } else if r > 0 {
            if n==n_regions as i64 {
                // Start from the beginning again but with different value for average length
                average_len += 1;
                n = n_regions as i64;
                m = 0;
            } else {
                n += 1;
                m -= 1;
            }
        // Residue is negative. This means we have subtracted too much and we just decrease n and increase m
        } else {
            n -= 1;
            m += 1;
        }
    }
    None
}



fn main () {
    let _ = (1..10_000_001).into_par_iter().map(|n_voxel| {
        for n_regions in 1..1_000 {
            match get_decomp_res(n_voxel, n_regions) {
                Some(res) => {
                    let (n, m, average_len) = res;
                    if n + m != n_regions {
                        println!("Output variables do not match! {} + {} != {}", n, m, n_regions);
                    }
                    if n*average_len + m*(average_len-1) != n_voxel {
                        println!("Output does not match! {}*{} + {}*{} != {}", n, average_len, m, average_len-1, n_voxel);
                    }
                },
                None => println!("No result for inputs n_voxel: {} n_regions: {}", n_voxel, n_regions),
            }
        }
    }).collect::<Vec<()>>();
}