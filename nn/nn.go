package nn

import (
	"math/rand"
	"simple_autodiff/data"
)

type Layer interface {
	Forward(x [][]*data.Variable) (o [][]*data.Variable)
	Parameters() (params []*data.Variable)
}

type LinearLayer struct {
	InFeatures  int
	OutFeatures int
	W           [][]*data.Variable
	b           []*data.Variable
}

func NewLinearLayer(InFeatures, OutFeatures int) (l Layer) {
	W := [][]*data.Variable{}
	b := []*data.Variable{}
	for i := 0; i < InFeatures; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < OutFeatures; j++ {
			tmp = append(tmp, data.NewVariable(rand.NormFloat64()))
		}
		W = append(W, tmp)
	}
	for j := 0; j < OutFeatures; j++ {
		b = append(b, data.NewVariable(rand.NormFloat64()))
	}
	return &LinearLayer{
		W:           W,
		b:           b,
		InFeatures:  InFeatures,
		OutFeatures: OutFeatures,
	}
}

func (l *LinearLayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	N := len(x)
	for k := 0; k < N; k++ {
		tmp := []*data.Variable{}
		for j := 0; j < l.OutFeatures; j++ {
			s := data.NewVariable(0)
			for i := 0; i < l.InFeatures; i++ {
				s = s.Add(x[k][i].Mul(l.W[i][j]))
			}
			s = s.Add(l.b[j])
			tmp = append(tmp, s)
		}
		o = append(o, tmp)
	}
	return
}

func (l *LinearLayer) Parameters() (params []*data.Variable) {
	for _, e := range l.W {
		params = append(params, e...)
	}
	params = append(params, l.b...)
	return
}

type SigmoidLayer struct {
}

func NewSigmoidLayer() (l Layer) {
	return &SigmoidLayer{}
}

func (l *SigmoidLayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.Sigmoid(x[i][j]))
		}
		o = append(o, tmp)
	}
	return
}

func (l *SigmoidLayer) Parameters() (params []*data.Variable) {
	return
}

type SequentialLayer struct {
	layers []Layer
}

func NewSequentialLayer(ls []Layer) (l Layer) {
	return &SequentialLayer{
		layers: ls,
	}
}

func (l *SequentialLayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	o = x
	for _, layer := range l.layers {
		o = layer.Forward(o)
	}
	return
}

func (l *SequentialLayer) Parameters() (params []*data.Variable) {
	for _, layer := range l.layers {
		params = append(params, layer.Parameters()...)
	}
	return
}
