package data

type BackwardFn interface {
	Backward(l float64)
}

type AddBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

func NewAddBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &AddBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

func (s *AddBackwadFn) Backward(l float64) {
	s.x.Backward(l)
	s.y.Backward(l)
}

type SubBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

func NewSubBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &SubBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

func (s *SubBackwadFn) Backward(l float64) {
	s.x.Backward(l)
	s.y.Backward(-l)
}

type MulBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

func NewMulBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &MulBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

func (s *MulBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.y.Val)
	s.y.Backward(l * s.x.Val)
}

type DivBackwadFn struct {
	x *Variable
	y *Variable
	o *Variable
}

func NewDivBackwardFn(x *Variable, y *Variable, o *Variable) (b BackwardFn) {
	return &DivBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

func (s *DivBackwadFn) Backward(l float64) {
	s.x.Backward(l / s.y.Val)
	s.y.Backward(-l * s.x.Val / (s.y.Val * s.y.Val))
}

type PowBackwadFn struct {
	x *Variable
	y float64
	o *Variable
}

func NewPowBackwardFn(x *Variable, y float64, o *Variable) (b BackwardFn) {
	return &PowBackwadFn{
		x: x,
		y: y,
		o: o,
	}
}

func (s *PowBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.y * s.o.Val / s.x.Val)
}

type SigmoidBackwadFn struct {
	x *Variable
	o *Variable
}

func NewSigmoidBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &SigmoidBackwadFn{
		x: x,
		o: o,
	}
}

func (s *SigmoidBackwadFn) Backward(l float64) {
	s.x.Backward(l * s.o.Val * (1 - s.o.Val))
}

type TanhBackwadFn struct {
	x *Variable
	o *Variable
}

func NewTanhBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &TanhBackwadFn{
		x: x,
		o: o,
	}
}

func (s *TanhBackwadFn) Backward(l float64) {
	s.x.Backward(l * (1 - (s.o.Val * s.o.Val)) / 2)
}

type ReLUBackwadFn struct {
	x *Variable
	o *Variable
}

func NewReLUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &ReLUBackwadFn{
		x: x,
		o: o,
	}
}

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

func NewLeakyReLUBackwardFn(x *Variable, alpha float64, o *Variable) (b BackwardFn) {
	return &LeakyReLUBackwadFn{
		x:     x,
		o:     o,
		alpha: alpha,
	}
}

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

func NewSILUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &SILUBackwadFn{
		x: x,
		o: o,
	}
}

func (s *SILUBackwadFn) Backward(l float64) {
	tmp := s.o.Val / s.x.Val
	s.x.Backward(l * (tmp + s.o.Val*(1-tmp)))
}

type GELUBackwadFn struct {
	x *Variable
	o *Variable
}

func NewGELUBackwardFn(x *Variable, o *Variable) (b BackwardFn) {
	return &GELUBackwadFn{
		x: x,
		o: o,
	}
}

func (s *GELUBackwadFn) Backward(l float64) {
	tmp := s.o.Val / s.x.Val
	s.x.Backward(l * (tmp + 1.702*s.o.Val*(1-tmp)))
}
