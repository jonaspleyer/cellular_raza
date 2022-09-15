use nalgebra::{Vector3,Vector6};


pub trait Spatial {
    fn pos(&self) -> Vector3<f64>;
    fn speed(&self) -> Vector3<f64>;

    fn set_pos(&mut self, p: Vector3<f64>);
    fn set_speed(&mut self, v: Vector3<f64>);
}


#[derive(Clone,Debug)]
pub struct SpatialModel {
    pos_speed: Vector6<f64>,
}


impl From<(&Vector3<f64>, &Vector3<f64>)> for SpatialModel {
    fn from(pv: (&Vector3<f64>, &Vector3<f64>)) -> Self {
        let p = pv.0;
        let v = pv.1;
        let x = Vector6::<f64>::new(p[0], p[1], p[2], v[0], v[1], v[2]);
        SpatialModel {
            pos_speed: x,
        }
    }
}


impl Spatial for SpatialModel {
    fn pos(&self) -> Vector3<f64> {
        return Vector3::<f64>::from(self.pos_speed.fixed_rows::<3>(0));
    }

    fn speed(&self) -> Vector3<f64> {
        return Vector3::<f64>::from(self.pos_speed.fixed_rows::<3>(3));
    }

    fn set_pos(&mut self, p: Vector3<f64>) {
        self.pos_speed.fixed_rows_mut::<3>(0).set_column(0, &p);
    }

    fn set_speed(&mut self, v: Vector3<f64>) {
        self.pos_speed.fixed_rows_mut::<3>(3).set_column(0, &v);
    }
}
