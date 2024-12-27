use crate::Value;
use rand::distributions::{Distribution, Uniform};

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(weights: Vec<Value>, bias: Value) -> Self {
        Neuron { weights, bias }
    }

    pub fn new_random(n_inputs: usize) -> Self {
        let between = Uniform::from(-1.0..1.);
        let mut rng = rand::thread_rng();
        let weights: Vec<Value> = (0..n_inputs)
            .map(|_| Value::new(between.sample(&mut rng)))
            .collect();
        let bias = Value::new(between.sample(&mut rng));
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Value {
        assert_eq!(inputs.len(), self.weights.len());
        let mut sum = self.bias.clone();
        for i in 0..self.weights.len() {
            sum = &sum + &(&self.weights[i] * &inputs[i]);
        }
        sum
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
    func: fn(Value) -> Value,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>, func: fn(Value) -> Value) -> Self {
        Layer { neurons, func }
    }

    pub fn new_random(n_inputs: usize, n_neurons: usize, func: fn(Value) -> Value) -> Self {
        let neurons: Vec<Neuron> = (0..n_neurons)
            .map(|_| Neuron::new_random(n_inputs))
            .collect();
        Layer { neurons, func }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| (self.func)(neuron.forward(inputs.clone())))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new() -> Self {
        MLP { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, inp: usize, out: usize, func: fn(Value) -> Value) {
        let n_inputs = self.layers.last().map_or(inp, |layer| layer.neurons.len());
        let layer = Layer::new_random(n_inputs, out, func);
        self.layers.push(layer);
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.forward(inputs))
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

#[test]
fn test_mlp() {
    let mut mlp = MLP::new();
    let learning_rate = 0.2;
    let sigmo = |x: Value| x.sigmoid();
    let tanh = |x: Value| x.tanh();
    mlp.add_layer(3, 4, sigmo);
    mlp.add_layer(4, 4, tanh);
    mlp.add_layer(4, 1, sigmo);
    let inputs = vec![Value::new(1.0), Value::new(2.0), Value::new(2.0)];

    for _ in 0..30 {
        let output: Vec<Value> = mlp.forward(inputs.clone());
        let mut error: Value = (&1.0.into() - &output[0]).pow(2.);
        println!("error: {:?}", error.data());
        error.backward();
        let params: Vec<Value> = mlp.parameters();
        for mut param in params {
            param.set_data(param.data() - learning_rate * f64::from(param.grad()));
            param.zero_grad();
        }
    }
    let output = mlp.forward(inputs.clone());
    let error = (&1.0.into() - &output[0]).pow(2.);
    assert!(error.data() < 0.1);
}
