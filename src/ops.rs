use crate::{Operator, Value, ValueInt};
use std::ops::{Add, Div, Mul, Neg, Sub, SubAssign};
use std::{cell::RefCell, rc::Rc};
impl<'a> Add for &'a Value {
    type Output = Value;

    fn add(self, other: &'a Value) -> Value {
        let data = self.0.borrow().data + other.0.borrow().data;
        let operator = Operator::Add;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), other.clone()],
            grad: 0.0,
        })))
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let data = self.0.borrow().data + other.0.borrow().data;
        let operator = Operator::Add;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), other.clone()],
            grad: 0.0,
        })))
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: Self) -> Value {
        self + &other.neg()
    }
}

impl SubAssign for Value {
    fn sub_assign(&mut self, other: Value) {
        *self = &*self - &other;
    }
}

impl<'a> SubAssign<&'a Value> for Value {
    fn sub_assign(&mut self, other: &'a Value) {
        *self = &*self - other;
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Value {
        let data = -self.0.borrow().data;
        let operator = Operator::Mul;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), Value::new(-1.0)],
            grad: 0.0,
        })))
    }
}

impl<'a> Mul for &'a Value {
    type Output = Value;

    fn mul(self, other: &'a Value) -> Value {
        let data = self.0.borrow().data * other.0.borrow().data;
        let operator = Operator::Mul;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), other.clone()],
            grad: 0.0,
        })))
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let data = self.0.borrow().data * other.0.borrow().data;
        let operator = Operator::Mul;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), other.clone()],
            grad: 0.0,
        })))
    }
}

impl<'a> Div for &'a Value {
    type Output = Value;

    fn div(self, other: &'a Value) -> Value {
        self * &other.pow(-1.0)
    }
}
