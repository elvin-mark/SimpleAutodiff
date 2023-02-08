package data

import "math"

type Variable struct {
	Val             float64
	Grad            float64
	BackwardHandler BackwardFn
}

func NewVariable(val float64) (v *Variable) {
	v = &Variable{}
	v.Val = val
	v.Grad = 0
	return v
}

func (v *Variable) ZeroGrad() {
	v.Grad = 0
}

func (v *Variable) Backward(l float64) {
	if v.BackwardHandler == nil {
		v.Grad += l
	} else {
		v.BackwardHandler.Backward(l)
	}
}

func (v *Variable) Add(w *Variable) *Variable {
	o := NewVariable(v.Val + w.Val)
	o.BackwardHandler = NewAddBackwardFn(v, w, o)
	return o
}

func (v *Variable) Sub(w *Variable) *Variable {
	o := NewVariable(v.Val - w.Val)
	o.BackwardHandler = NewSubBackwardFn(v, w, o)
	return o
}

func (v *Variable) Mul(w *Variable) *Variable {
	o := NewVariable(v.Val * w.Val)
	o.BackwardHandler = NewMulBackwardFn(v, w, o)
	return o
}

func (v *Variable) Div(w *Variable) *Variable {
	o := NewVariable(v.Val / w.Val)
	o.BackwardHandler = NewDivBackwardFn(v, w, o)
	return o
}

func (v *Variable) Pow(w float64) *Variable {
	o := NewVariable(math.Pow(v.Val, w))
	o.BackwardHandler = NewPowBackwardFn(v, w, o)
	return o
}
