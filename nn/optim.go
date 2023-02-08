package nn

import "simple_autodiff/data"

type Optimizer interface {
	ZeroGrad()
	Step()
}

type SGDOptimizer struct {
	params []*data.Variable
	lr     float64
}

func NewSGDOptimizer(params []*data.Variable, lr float64) (o Optimizer) {
	return &SGDOptimizer{
		params: params,
		lr:     lr,
	}
}

func (s *SGDOptimizer) ZeroGrad() {
	for _, p := range s.params {
		p.ZeroGrad()
	}
}

func (s *SGDOptimizer) Step() {
	for _, p := range s.params {
		p.Val -= s.lr * p.Grad
	}
}
