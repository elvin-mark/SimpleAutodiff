package main

import (
	"fmt"

	data "github.com/elvin-mark/SimpleAutodiff/data"
	datasets "github.com/elvin-mark/SimpleAutodiff/datasets"
	models "github.com/elvin-mark/SimpleAutodiff/models"
	nn "github.com/elvin-mark/SimpleAutodiff/nn"
	utils "github.com/elvin-mark/SimpleAutodiff/utils"
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
		nn.NewTanhLayer(),
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
	ds_x, ds_y := datasets.XORDataset(0)
	dl := datasets.NewDataLoader(ds_x, ds_y, 2)

	model := models.MLP(2, 3, 2)
	optim := nn.NewSGDOptimizer(model.Parameters(), 0.1)
	crit := nn.NewMSELoss()

	for epoch := 0; epoch < 5; epoch++ {
		utils.TrainOneEpoch(model, crit, optim, dl)
	}
}

func TestLossFunction() {
	// ds_x, ds_y := datasets.ToyCircleDataset(20, 1.5)
	ds_x, ds_y := datasets.Toy2DDataset(20)
	dl := datasets.NewDataLoader(ds_x, ds_y, 2)
	model := nn.NewSequentialLayer([]nn.Layer{
		nn.NewLinearLayer(2, 10),
		nn.NewReLULayer(),
		nn.NewLinearLayer(10, 5),
		nn.NewReLULayer(),
		nn.NewLinearLayer(5, 2),
	})
	optim := nn.NewSGDOptimizer(model.Parameters(), 0.1)
	crit := nn.NewCrossEntropyLoss()

	for epoch := 0; epoch < 20; epoch++ {
		utils.TrainOneEpoch(model, crit, optim, dl)
	}
	o := model.Forward(ds_x)
	for i, e := range o {
		fmt.Println(e[0].Val, e[1].Val, ds_y[i][0].Val)
	}

}

func main() {
	// TestBackwardBasicOperations()
	// TestBackwardFunctions()
	// TestNN()
	// TestDataLoader()
	TestLossFunction()
}
