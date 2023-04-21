use plotters::prelude::{HSLColor,RGBAColor,RGBColor};


/// Defines how colorscales should be used.
pub trait ColorScale<ColorType: plotters::prelude::Color, FloatType=f32>
where
    FloatType: num::Float,
{
    fn get_color(&self, h: FloatType) -> ColorType {
        self.get_color_normalized(h, FloatType::zero(), FloatType::one())
    }

    fn get_color_normalized(&self, h: FloatType, min: FloatType, max: FloatType) -> ColorType;
}


/// Derive a colorscale from some provided colors and interpolate linearly in between them.
///
/// While there are of course many other ways of implementing color interpolation, this is probably
/// the easiest which already prodocues great results.
pub struct DerivedColorScale<ColorType> {
    colors: Vec<ColorType>,
}


impl<ColorType: plotters::prelude::Color + Clone> DerivedColorScale<ColorType> {
    /// Creates a linear interpolation colormap from the specified colors.
    ///
    /// The minimum value corresponds to the first entry and the maximum value to the last one.
    pub fn new(colors: &[ColorType]) -> Self {
        DerivedColorScale { colors: colors.iter().map(|color| color.clone()).collect::<Vec<_>>() }
    }
}


macro_rules! calculate_new_color_value(
    ($relative_difference:expr, $colors:expr, $index_upper:expr, $index_lower:expr, RGBColor) => {
        RGBColor(
            // These equations are a very complicated linear extrapolation with lots of casting between numerical values
            // In principle every cast should be safe which is why we choose to unwrap
            //           (1.0  - r)                   *                                           color_value_1  +                   r *                                           color_value_2
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].0).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].0).unwrap()).round().to_u8().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].1).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].1).unwrap()).round().to_u8().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].2).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].2).unwrap()).round().to_u8().unwrap()
        )
    };
    ($relative_difference:expr, $colors:expr, $index_upper:expr, $index_lower:expr, RGBAColor) => {
        RGBAColor(
            // These equations are a very complicated linear extrapolation with lots of casting between numerical values
            // In principle every cast should be safe which is why we choose to unwrap
            //           (1.0  - r)                   *                                           color_value_1  +                   r *                                           color_value_2
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].0).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].0).unwrap()).round().to_u8().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].1).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].1).unwrap()).round().to_u8().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_u8($colors[$index_upper].2).unwrap() + $relative_difference * FloatType::from_u8($colors[$index_lower].2).unwrap()).round().to_u8().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_f64($colors[$index_upper].3).unwrap() + $relative_difference * FloatType::from_f64($colors[$index_lower].3).unwrap()).to_f64().unwrap()
        )
    };
    ($relative_difference:expr, $colors:expr, $index_upper:expr, $index_lower:expr, HSLColor) => {
        HSLColor(
            // These equations are a very complicated linear extrapolation with lots of casting between numerical values
            // In principle every cast should be safe which is why we choose to unwrap
            //           (1.0  - r)                   *                                           color_value_1  +                   r *                                           color_value_2
            ((FloatType::one() - $relative_difference) * FloatType::from_f64($colors[$index_upper].0).unwrap() + $relative_difference * FloatType::from_f64($colors[$index_lower].0).unwrap()).to_f64().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_f64($colors[$index_upper].1).unwrap() + $relative_difference * FloatType::from_f64($colors[$index_lower].1).unwrap()).to_f64().unwrap(),
            ((FloatType::one() - $relative_difference) * FloatType::from_f64($colors[$index_upper].2).unwrap() + $relative_difference * FloatType::from_f64($colors[$index_lower].2).unwrap()).to_f64().unwrap(),
        )
    };
);


#[macro_export]
macro_rules! define_linear_interpolation_color_scale{
    ($color_scale_name:ident, $number_colors:literal, $color_type:tt, $(($($color_value:expr),+)),+) => {
        #[doc = "The "]
        #[doc = stringify!($color_scale_name)]
        #[doc = " color-scale"]
        pub struct $color_scale_name {}

        impl $color_scale_name {
            const COLORS: [$color_type; $number_colors] = [$($color_type($($color_value),+)),+];
        }

        impl<FloatType: std::fmt::Debug + num::Float + num::FromPrimitive + num::ToPrimitive> ColorScale<$color_type, FloatType> for $color_scale_name {
            fn get_color_normalized(&self, h: FloatType, min: FloatType, max: FloatType) -> $color_type {
                // Ensure that we do have a value in bounds
                let h = h.max(min).min(max);
                // Make sure that we really have a minimal value which is smaller than the maximal value
                assert_eq!(min<max, true);
                // Next calculate a normalized value between 0.0 and 1.0
                let t = (h - min)/(max-min);
                let approximate_index = t * (FloatType::from_usize(Self::COLORS.len()).unwrap() - FloatType::one()).max(FloatType::zero());
                // Calculate which index are the two most nearest of the supplied value
                let index_lower = approximate_index.floor().to_usize().unwrap();
                let index_upper = approximate_index.ceil().to_usize().unwrap();
                // Calculate the relative difference, ie. is the actual value more towards the color of index_upper or index_lower?
                let relative_difference = approximate_index.ceil() - approximate_index;
                // Interpolate the final color linearly
                calculate_new_color_value!(relative_difference, Self::COLORS, index_upper, index_lower, $color_type)
            }
        }

        impl $color_scale_name {
            pub fn get_color<FloatType: std::fmt::Debug + num::Float + num::FromPrimitive + num::ToPrimitive>(h: FloatType) -> $color_type {
                let color_scale = $color_scale_name {};
                color_scale.get_color(h)
            }

            pub fn get_color_normalized<FloatType: std::fmt::Debug + num::Float + num::FromPrimitive + num::ToPrimitive>(h: FloatType, min: FloatType, max: FloatType) -> $color_type {
                let color_scale = $color_scale_name {};
                color_scale.get_color_normalized(h, min, max)
            }
        }
    }
}


define_linear_interpolation_color_scale!{
    ViridisRGBA,
    8,
    RGBAColor,
    ( 68,   1,  84, 1.0),
    ( 70,  50, 127, 1.0),
    ( 54,  92, 141, 1.0),
    ( 39, 127, 143, 1.0),
    ( 31, 162, 136, 1.0),
    ( 74, 194, 110, 1.0),
    (160, 219,  57, 1.0),
    (254, 232,  37, 1.0)
}


define_linear_interpolation_color_scale!{
    ViridisFading,
    8,
    RGBAColor,
    ( 68,   1,  84, 0.5),
    ( 70,  50, 127, 0.5),
    ( 54,  92, 141, 0.5),
    ( 39, 127, 143, 0.5),
    ( 31, 162, 136, 0.5),
    ( 74, 194, 110, 0.375),
    (160, 219,  57, 0.25),
    (254, 232,  37, 0.125)
}


define_linear_interpolation_color_scale!{
    ViridisRGB,
    8,
    RGBColor,
    ( 68,   1,  84),
    ( 70,  50, 127),
    ( 54,  92, 141),
    ( 39, 127, 143),
    ( 31, 162, 136),
    ( 74, 194, 110),
    (160, 219,  57),
    (254, 232,  37)
}


define_linear_interpolation_color_scale!{
    BlackWhiteRGB,
    2,
    RGBColor,
    (  0,   0,   0),
    (255, 255,   255)
}


define_linear_interpolation_color_scale!{
    GreenGrey,
    2,
    RGBColor,
    ( 51, 102,  51),
    ( 51, 255,  51)
}


define_linear_interpolation_color_scale!{
    PinkGrey,
    2,
    RGBColor,
    (102,  52,  83),
    (247, 126, 201)
}


define_linear_interpolation_color_scale!{
    Life,
    4,
    RGBColor,
    (153,   0,   0),
    (255, 153,  51),
    (120, 255, 120),
    ( 51, 255,  51)
}


define_linear_interpolation_color_scale!{
    Dirt,
    4,
    RGBColor,
    (140,  94,  14),
    (199, 151,  68),
    (217, 184, 128),
    (212, 196, 169)
}


define_linear_interpolation_color_scale!{
    Poison,
    4,
    RGBColor,
    ( 11,  54,  38),
    ( 10, 110,  73),
    ( 63, 171, 104),
    (141, 235, 113)
}


define_linear_interpolation_color_scale!{
    Awesome,
    5,
    RGBColor,
    (204, 229, 255),
    (153, 153, 255),
    (178, 102, 255),
    (255,  51, 255),
    (255,   0, 127)
}


macro_rules! implement_color_scale_for_derived_colorscale{
    ($($color_type:tt),+) => {
        $(
            impl<FloatType: num::Float + num::FromPrimitive + num::ToPrimitive> ColorScale<$color_type, FloatType> for DerivedColorScale<$color_type> {
                fn get_color_normalized(&self, h: FloatType, min: FloatType, max: FloatType) -> $color_type {
                    // Ensure that we do have a value in bounds
                    let h = h.min(min).max(max);
                    // Make sure that we really have a minimal value which is smaller than the maximal value
                    assert_eq!(min<max, true);
                    // Next calculate a normalized value between 0.0 and 1.0
                    let t = (max - h)/(max-min);
                    let approximate_index = t * FloatType::from_usize(self.colors.len()).unwrap();
                    // Calculate which index are the two most nearest of the supplied value
                    let index_lower = approximate_index.floor().to_usize().unwrap();
                    let index_upper = approximate_index.ceil().to_usize().unwrap();
                    // Calculate the relative difference, ie. is the actual value more towards the color of index_upper or index_lower?
                    let relative_difference = approximate_index.ceil() - approximate_index;
                    // Interpolate the final color linearly
                    calculate_new_color_value!(relative_difference, self.colors, index_upper, index_lower, $color_type)
                }
            }
        )+
    }
}

implement_color_scale_for_derived_colorscale!{RGBAColor, RGBColor, HSLColor}
