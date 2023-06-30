use cellular_raza_core_derive::identical;

#[test]
fn test_1() {
    identical!(Mechanics, Mechanics, assert!(true));
}

#[test]
#[should_panic]
fn test_2() {
    identical!(Chili, Chili, assert!(false));
}

#[test]
fn test_3() {
    identical!(Chili, Cara, assert!(false));
}
