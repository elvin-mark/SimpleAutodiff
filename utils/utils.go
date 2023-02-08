package utils

import (
	"fmt"
	"simple_autodiff/data"
	"simple_autodiff/nn"
)

func TrainOneEpoch(model nn.Layer, crit nn.Loss, optim nn.Optimizer, dl data.DataLoader) {
	dl.Rewind()
	avgLoss := 0.0
	for dl.HasNext() {
		optim.ZeroGrad()
		x, y := dl.Next()
		o := model.Forward(x)
		l := crit.Calculate(o, y)
		l.Backward(1.0)
		optim.Step()
		avgLoss += l.Val
	}
	fmt.Println("Avg. Loss: ", avgLoss/float64(dl.Len()))
}
