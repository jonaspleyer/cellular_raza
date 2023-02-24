const VIRIDIS_COLORS: [[f64; 4]; 8] = [
    [0.267004, 0.004874, 0.329415, 1.],
    [0.275191, 0.194905, 0.496005, 1.],
    [0.212395, 0.359683, 0.55171 , 1.],
    [0.153364, 0.497   , 0.557724, 1.],
    [0.122312, 0.633153, 0.530398, 1.],
    [0.288921, 0.758394, 0.428426, 1.],
    [0.626579, 0.854645, 0.223353, 1.],
    [0.993248, 0.906157, 0.143936, 1.]];


/// The single parameter here is meant to be between 0.0 and 1.0
pub fn create_viridis_color(h: f64) -> plotters::prelude::RGBAColor {
    let index_upper = ((7.0*h).ceil() as usize).min(7);
    let index_lower = ((7.0*h).floor() as usize).max(0);
    let relative_distance_between = 7.0*h - index_lower as f64;
    let mut color_values: [f64; 4] = [0.0; 4];
    for i in 0..4 {
        color_values[i] = relative_distance_between * VIRIDIS_COLORS[index_upper][i] + (1.0 - relative_distance_between) * VIRIDIS_COLORS[index_lower][i];
    }
    plotters::prelude::RGBAColor(
        (256.0 * color_values[0]).round() as u8,
        (256.0 * color_values[1]).round() as u8,
        (256.0 * color_values[2]).round() as u8,
        1.0
    )
}
