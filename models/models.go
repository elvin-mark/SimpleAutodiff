package models

import "simple_autodiff/nn"

func Perceptron(inputs int, outputs int) (model nn.Layer) {
	model = nn.NewSequentialLayer([]nn.Layer{
		nn.NewLinearLayer(inputs, outputs),
	})
	return
}

func MLP(inputs int, hiddens int, outputs int) (model nn.Layer) {
	model = nn.NewSequentialLayer([]nn.Layer{
		nn.NewLinearLayer(inputs, hiddens),
		nn.NewSigmoidLayer(),
		nn.NewLinearLayer(hiddens, outputs),
	})
	return
}
