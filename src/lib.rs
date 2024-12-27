use std::{
    cell::RefCell,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub},
    rc::Rc,
    vec,
};

pub mod mlp;
pub mod ops;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    None,
    Pow,
    Tanh,
    Exp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value(Rc<RefCell<ValueInt>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator: Operator::None,
            prev: vec![],
            grad: 0.0,
        })))
    }

    pub fn zero_grad(&mut self) {
        self.0.borrow_mut().grad = 0.0;
    }

    pub fn set_data(&mut self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn get_ops(&self) -> String {
        let value_int = self.0.borrow();
        if value_int.operator == Operator::None {
            return value_int.data.to_string();
        } else {
            let mut result = String::new();
            result.push('(');
            result.push_str(&value_int.prev[0].get_ops());

            match value_int.operator {
                Operator::Add => result.push('+'),
                Operator::Sub => result.push('-'),
                Operator::Mul => result.push('*'),
                Operator::Pow => result.push('^'),
                Operator::Tanh => result.push_str("tanh("),
                Operator::Exp => result.push_str("exp("),
                Operator::None => {}
            }

            result.push_str(&value_int.prev[1].get_ops());

            if value_int.operator == Operator::Tanh || value_int.operator == Operator::Exp {
                result.push(')');
            }
            result.push(')');
            result
        }
    }

    pub fn tanh(&self) -> Value {
        let data = self.0.borrow().data.tanh();
        let operator = Operator::Tanh;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone()],
            grad: 0.0,
        })))
    }

    pub fn exp(&self) -> Value {
        let data = self.0.borrow().data.exp();
        let operator = Operator::Exp;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone()],
            grad: 0.0,
        })))
    }

    pub fn sigmoid(&self) -> Value {
        &Value::new(1.0) / &(&Value::new(1.0) + &(-self).exp())
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().grad = 1.0;
        self.set_grad(1.0);
    }

    pub fn set_grad(&mut self, grad: f64) {
        let mut self_borrow_mut = self.0.borrow_mut();
        self_borrow_mut.grad += grad;
        let operator = self_borrow_mut.operator;
        let mut prev = self_borrow_mut.prev.clone(); // Clone the previous values to avoid multiple borrows

        drop(self_borrow_mut); // Explicitly drop the mutable borrow

        match operator {
            Operator::Exp => {
                let data = prev[0].0.borrow().data;
                prev[0].set_grad(grad * data.exp());
            }
            Operator::Add => {
                prev[0].set_grad(grad);
                prev[1].set_grad(grad);
            }
            Operator::Sub => {
                prev[0].set_grad(grad);
                prev[1].set_grad(-grad);
            }
            Operator::Mul => {
                let data1 = prev[1].0.borrow().data;
                let data0 = prev[0].0.borrow().data;
                prev[0].set_grad(grad * data1);
                prev[1].set_grad(grad * data0);
            }
            Operator::Pow => {
                let data1 = prev[1].0.borrow().data;
                let data0 = prev[0].0.borrow().data;
                prev[0].set_grad(grad * data1 * data0.powf(data1 - 1.0));
            }
            Operator::Tanh => {
                let data = prev[0].0.borrow().data;
                prev[0].set_grad(grad * (1.0 - data.tanh().powi(2)));
            }
            Operator::None => {}
        }
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> Value {
        self.0.borrow().grad.to_value()
    }

    pub fn pow(&self, n: f64) -> Value {
        let data = self.0.borrow().data.powf(n);
        let operator = Operator::Pow;
        Value(Rc::new(RefCell::new(ValueInt {
            data,
            operator,
            prev: vec![self.clone(), Value::new(n)],
            grad: 0.0,
        })))
    }
}

trait ToValue {
    fn to_value(&self) -> Value;
}

impl ToValue for f64 {
    fn to_value(&self) -> Value {
        Value::new(*self)
    }
}

impl ToValue for f32 {
    fn to_value(&self) -> Value {
        Value::new(*self as f64)
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInt>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Value {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct ValueInt {
    data: f64,
    operator: Operator,
    prev: Vec<Value>,
    grad: f64,
}

impl Into<Value> for f64 {
    fn into(self) -> Value {
        Value::new(self)
    }
}

impl From<Value> for f64 {
    fn from(value: Value) -> f64 {
        value.0.borrow().data
    }
}

#[test]
fn test_operator() {
    let a = Value::new(1.0);
    let b = Value::new(1. / 2.0);
    let c = Value::new(3.0);
    let d = Value::new(4.0);

    let e = &d * &b;
    let f = &a * &c;
    let mut g = &e - &f;
    // g = (d/b) - ac
    // check grad
    g.backward();
    println!("a.grad: {}", a.0.borrow().grad); // -c = -3
    assert_eq!(a.0.borrow().grad, -3.0);
    println!("b.grad: {}", b.0.borrow().grad);
    assert_eq!(b.0.borrow().grad, 4.0);
}

#[test]
fn test_pow() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);

    let e = a.pow(2.0);
    let f = b.pow(3.0);
    let mut g = &e - &f;
    // g = a^2 - b^3
    // check grad
    g.backward();
    println!("a.grad: {}", a.0.borrow().grad); // 2a = 4
    assert_eq!(a.0.borrow().grad, 4.0);
    println!("b.grad: {}", b.0.borrow().grad); // -3b^2 = -27
    assert_eq!(b.0.borrow().grad, -27.0);
}

#[test]
fn neuron() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.881373587);
    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;
    let y = &(&x1w1 + &x2w2) + &b;
    let mut o = y.tanh();
    o.backward();
    println!("y.grad: {}", y.0.borrow().grad); // ~ 0.5
}

#[test]
fn test_same_value() {
    let a = Value::new(1.0);
    let mut c = &a + &a;
    c.backward();
    println!("a.grad: {}", a.0.borrow().grad); // 2
    assert_eq!(a.0.borrow().grad, 2.0);
    let a = Value::new(3.0);
    let mut c = &a * &a;
    c.backward();
    println!("a.grad: {}", a.0.borrow().grad); // 6
    assert_eq!(a.0.borrow().grad, 6.0);
}
