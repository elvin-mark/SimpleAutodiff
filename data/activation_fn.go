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

func ReLU(v *Variable) (o *Variable) {
	o = NewVariable(0)
	if v.Val >= 0 {
		o.Val = v.Val
	}
	o.BackwardHandler = NewReLUBackwardFn(v, o)
	return
}

func LeakyReLU(v *Variable, alpha float64) (o *Variable) {
	o = NewVariable(v.Val)
	if v.Val < 0 {
		o.Val = alpha * v.Val
	}
	o.BackwardHandler = NewLeakyReLUBackwardFn(v, alpha, o)
	return
}

func SILU(v *Variable) (o *Variable) {
	o = NewVariable(v.Val / (1 + math.Exp(-v.Val)))
	o.BackwardHandler = NewSILUBackwardFn(v, o)
	return
}

func GELU(v *Variable) (o *Variable) {
	o = NewVariable(v.Val / (1 + math.Exp(-1.702*v.Val)))
	o.BackwardHandler = NewGELUBackwardFn(v, o)
	return
}
