package nn

import (
	"github.com/elvin-mark/SimpleAutodiff/data"
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
	W := data.NewMatrix(InFeatures, OutFeatures)
	b := data.NewVector(OutFeatures)
	return &LinearLayer{
		W:           W,
		b:           b,
		InFeatures:  InFeatures,
		OutFeatures: OutFeatures,
	}
}

func (l *LinearLayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	o = data.GEMM(x, l.W, l.b)
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

type TanhLayer struct {
}

func NewTanhLayer() (l Layer) {
	return &TanhLayer{}
}

func (l *TanhLayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.Tanh(x[i][j]))
		}
		o = append(o, tmp)
	}
	return
}

func (l *TanhLayer) Parameters() (params []*data.Variable) {
	return
}

type ReLULayer struct {
}

func NewReLULayer() (l Layer) {
	return &ReLULayer{}
}

func (l *ReLULayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.ReLU(x[i][j]))
		}
		o = append(o, tmp)
	}
	return
}

func (l *ReLULayer) Parameters() (params []*data.Variable) {
	return
}

type LeakyReLULayer struct {
	alpha float64
}

func NewLeakyReLULayer(alpha float64) (l Layer) {
	return &LeakyReLULayer{
		alpha: alpha,
	}
}

func (l *LeakyReLULayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.LeakyReLU(x[i][j], l.alpha))
		}
		o = append(o, tmp)
	}
	return
}

func (l *LeakyReLULayer) Parameters() (params []*data.Variable) {
	return
}

type SILULayer struct {
}

func NewSILULayer() (l Layer) {
	return &SILULayer{}
}

func (l *SILULayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.SILU(x[i][j]))
		}
		o = append(o, tmp)
	}
	return
}

func (l *SILULayer) Parameters() (params []*data.Variable) {
	return
}

type GELULayer struct {
}

func NewGELULayer() (l Layer) {
	return &GELULayer{}
}

func (l *GELULayer) Forward(x [][]*data.Variable) (o [][]*data.Variable) {
	rows := len(x)
	cols := len(x[0])
	for i := 0; i < rows; i++ {
		tmp := []*data.Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, data.GELU(x[i][j]))
		}
		o = append(o, tmp)
	}
	return
}

func (l *GELULayer) Parameters() (params []*data.Variable) {
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
