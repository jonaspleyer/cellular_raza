pub enum StorageIdent {
    Cell,
    Voxel,
    MultiVoxel,
}


impl StorageIdent {
    pub const fn value(self) -> u16 {
        match self {
            StorageIdent::Cell => 1,
            StorageIdent::Voxel => 2,
            StorageIdent::MultiVoxel => 3,
        }
    }
}