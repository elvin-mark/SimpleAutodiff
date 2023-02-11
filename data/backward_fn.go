package data

type BackwardFn interface {
	Backward(l float64)
}

type AddBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

// o = x + y
func NewAddBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &AddBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * 1 = l
// dL/dy = dL/do * do/dy = l * 1 = l
func (s *AddBackwadFn) Backward(l float64) {
	s.x.Backward(l)
	s.y.Backward(l)
}

type SubBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

// o = x - y
func NewSubBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &SubBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * 1 = l
// dL/dy = dL/do * do/dy = l * -1 = -l
func (s *SubBackwadFn) Backward(l float64) {
	s.x.Backward(l)
	s.y.Backward(-l)
}

type MulBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

// o = x * y
func NewMulBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &MulBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * y
// dL/dy = dL/do * do/dy = l * x
func (s *MulBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.y.Val)
	s.y.Backward(l * s.x.Val)
}

type DivBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

// o = x / y
func NewDivBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &DivBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * 1 / y = l / y
// dL/dy = dL/do * do/dy = l * (-x / y^2) = - l * x / y^2
func (s *DivBackwadFn) Backward(l float64) {
	s.x.Backward(l / s.y.Val)
	s.y.Backward(-l * s.x.Val / (s.y.Val * s.y.Val))
}

type PowBackwadFn struct {
	x *Variable
	y float64
	o *Variable
}

// o = x ^ y
func NewPowBackwardFn(x *Variable, y float64, o *Variable) (b BackwardFn) {
	return &PowBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * y * x ^ (y-1) = l * y * o / x
func (s *PowBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.y * s.o.Val / s.x.Val)
}

type ExpBackwadFn struct {
	x *Variable
	o *Variable
}

// o = e ^ x
func NewExpBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &ExpBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * e ^ x = l * o
func (s *ExpBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.o.Val)
}

type LogBackwadFn struct {
	x *Variable
	o *Variable
}

// o = log(x)
func NewLogBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &LogBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (1 / x) = l / x
func (s *LogBackwadFn) Backward(l float64) {
	s.x.Backward(l / s.x.Val)
}

type SigmoidBackwadFn struct {
	x *Variable
	o *Variable
}

// o = sigma(x) = 1 / (1 + e ^ (-x)
func NewSigmoidBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &SigmoidBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * o * (1 - o)
func (s *SigmoidBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.o.Val * (1 - s.o.Val))
}

type TanhBackwadFn struct {
	x *Variable
	o *Variable
}

// o = tanh(x) = (1 - e ^(-x))/(1 + e^(-x))
func NewTanhBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &TanhBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (1 - o^2)/2
func (s *TanhBackwadFn) Backward(l float64) {
	s.x.Backward(l * (1 - (s.o.Val * s.o.Val)) / 2)
}

type ReLUBackwadFn struct {
	x *Variable
	o *Variable
}

// o = relu(x) = x if x >=0 else 0
func NewReLUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &ReLUBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (1 if x >= 0 else 0) = l if x >= 0 else 0
func (s *ReLUBackwadFn) Backward(l float64) {
	if s.x.Val >= 0 {
		s.x.Backward(l)
	} else {
		s.x.Backward(0)
	}
}

type LeakyReLUBackwadFn struct {
	x     *Variable
	o     *Variable
	alpha float64
}

// o = leaky_relu(x) = x if x >= 0 else alpha * x
func NewLeakyReLUBackwardFn(x *Variable, alpha float64, o *Variable) (b BackwardFn) {
	return &LeakyReLUBackwadFn{
		x:     x,
		o:     o,
		alpha: alpha,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (1 if x >= 0 else alpha) = l if x >= 0 else l * alpha
func (s *LeakyReLUBackwadFn) Backward(l float64) {
	if s.x.Val >= 0 {
		s.x.Backward(l)
	} else {
		s.x.Backward(l * s.alpha)
	}
}

type SILUBackwadFn struct {
	x *Variable
	o *Variable
}

// o = x * sigma(x)
func NewSILUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &SILUBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (sigma(x) + x * sigma(x) * (1 - sigma(x)))
func (s *SILUBackwadFn) Backward(l float64) {
	tmp := s.o.Val / s.x.Val
	s.x.Backward(l * (tmp + s.o.Val*(1-tmp)))
}

type GELUBackwadFn struct {
	x *Variable
	o *Variable
}

// o ~ x * sigma(1.702 * x)
func NewGELUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &GELUBackwadFn{
		x: x,
		o: o,
	}
}

// l = dL/do
// dL/dx = dL/do * do/dx = l * (sigma(1.702*x) + x *  1.702 * sigma( 1.702*x) * (1 - sigma( 1.702*x)))
func (s *GELUBackwadFn) Backward(l float64) {
	tmp := s.o.Val / s.x.Val
	s.x.Backward(l * (tmp + 1.702*s.o.Val*(1-tmp)))
}
