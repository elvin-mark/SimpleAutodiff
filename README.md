# SimpleAutodiff

Simple implementation of an scalar AutoDifferentiation module for learning purpose. This module defines a type `Variable` and mathematical operations for this type. In each operation, a backward callback function `BackwardFn` is generated, the output of the operation is also variable and this new instance records the inputs and outputs of the operation as well as defined the backward operation.

# Variable and BackwardFn

The main type used in this module is `Variable` which is defined as shown below. This object contains information about the actual value `Val` of the variable, the gradient `Grad` with respect to this variable and a way to perfom the backward propagation `BackwardHandler`.

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
