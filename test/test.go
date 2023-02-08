package main

import (
	"fmt"
	data "simple_autodiff/data"
	datasets "simple_autodiff/datasets"
	models "simple_autodiff/models"
	nn "simple_autodiff/nn"
	utils "simple_autodiff/utils"
)

func TestBackwardBasicOperations() {
	x := data.NewVariable(2)
	y := data.NewVariable(5)
	z := data.NewVariable(6)
	w := data.NewVariable(1)

	a := x.Add(y).Mul(z).Div(x.Add(w))
	a.Backward(1)

	fmt.Println("x = ", x.Val, " y = ", y.Val, " z = ", z.Val, " w = ", w.Val)
	fmt.Println("Operation: a = z(x + y)/(x + w) = ", a.Val)
	fmt.Println("da/dx = ", x.Grad)
	fmt.Println("da/dy = ", y.Grad)
	fmt.Println("da/dz = ", z.Grad)
	fmt.Println("da/dw = ", w.Grad)
}

func TestBackwardFunctions() {
	x := data.NewVariable(2)
	o := data.Sigmoid(x)

	o.Backward(1)

	fmt.Println("x = ", x.Val)
	fmt.Println("o = sigmoid(x)", o.Val)
	fmt.Println("do/dx = ", x.Grad)

	x.ZeroGrad()
	o = data.Tanh(x)

	o.Backward(1)

	fmt.Println("x = ", x.Val)
	fmt.Println("o = tanh(x)", o.Val)
	fmt.Println("do/dx = ", x.Grad)
}

func TestNN() {
	x := [][]*data.Variable{
		{data.NewVariable(1.0), data.NewVariable(2.0)},
	}
	t := [][]*data.Variable{
		{data.NewVariable(3.0), data.NewVariable(5.0)},
	}

	model := nn.NewSequentialLayer([]nn.Layer{
		nn.NewLinearLayer(2, 3),
		nn.NewSigmoidLayer(),
		nn.NewLinearLayer(3, 2),
	})

	optim := nn.NewSGDOptimizer(model.Parameters(), 0.1)
	crit := nn.NewMSELoss()

	for epoch := 0; epoch < 5; epoch++ {
		optim.ZeroGrad()
		o := model.Forward(x)
		l := crit.Calculate(o, t)
		l.Backward(1.0)
		optim.Step()
		fmt.Println("Loss: ", l.Val)
	}

	o := model.Forward(x)
	for i, e := range o[0] {
		fmt.Printf("o[%d] = %f\n", i, e.Val)
	}
}

func TestDataLoader() {
	ds_x, ds_y := datasets.XORDataset()
	dl := datasets.NewDataLoader(ds_x, ds_y, 2)

	model := models.MLP(2, 3, 2)
	optim := nn.NewSGDOptimizer(model.Parameters(), 0.1)
	crit := nn.NewMSELoss()

	for epoch := 0; epoch < 5; epoch++ {
		utils.TrainOneEpoch(model, crit, optim, dl)
	}
}

func main() {
	// TestBackwardBasicOperations()
	// TestBackwardFunctions()
	// TestNN()
	TestDataLoader()
}
