# SimpleAutodiff

Simple implementation of an scalar AutoDifferentiation module for learning purpose. This module defines a type `Variable` and mathematical operations for this type. In each operation, a backward callback function `BackwardFn` is generated, the output of the operation is also a variable, and this new variable stores the inputs and outputs of the operation, and defined the backward operation.

# Variable and BackwardFn

The main type used in this module is `Variable`, which is defined as shown below. This object contains information about the actual value `Val` of the variable, the gradient `Grad` with respect to this variable and a way to perfom the backward propagation `BackwardHandler`.

```go
type Variable struct {
	Val             float64
	Grad            float64
	BackwardHandler BackwardFn
}
```

The `BackwardHandler` is defined as the follow interface which defines the `Backward` function. When an instance of this interface is created, the inputs and outputs involved in the operation that resulted into this variable are attached to the instance.

```go
type BackwardFn interface {
	Backward(l float64)
}
```

# Simple Test

This is a simple test showing how AutoDifferentiation works
```go
import (
    "fmt"
	data "github.com/elvin-mark/SimpleAutodiff/data"
)
func main(){
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
```

# Simple Neural Network Training
```go
import (
    "fmt"
	data "github.com/elvin-mark/SimpleAutodiff/data"
	datasets "github.com/elvin-mark/SimpleAutodiff/datasets"
	nn "github.com/elvin-mark/SimpleAutodiff/nn"
	utils "github.com/elvin-mark/SimpleAutodiff/utils"
)

func main(){
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
```
