package data

import "math"

func Sigmoid(v *Variable) (o *Variable) {
	o = NewVariable(1 / (1 + math.Exp(-v.Val)))
	o.BackwardHandler = NewSigmoidBackwardFn(v, o)
	return
}

func Tanh(v *Variable) (o *Variable) {
	o = NewVariable((1 - math.Exp(-v.Val)) / (1 + math.Exp(-v.Val)))
	o.BackwardHandler = NewTanhBackwardFn(v, o)
	return
}
