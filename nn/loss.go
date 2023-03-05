package nn

import (
	data "github.com/elvin-mark/SimpleAutodiff/data"
)

// General Loss Interface
type Loss interface {
	Calculate(y [][]*data.Variable, t [][]*data.Variable) (l *data.Variable)
}

// Mean Square Error Loss
type MSELoss struct {
}

func NewMSELoss() (l Loss) {
	return &MSELoss{}
}

func (m *MSELoss) Calculate(y [][]*data.Variable, t [][]*data.Variable) (l *data.Variable) {
	l = data.NewVariable(0)
	rows := len(y)
	cols := len(y[0])
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// loss += (y_{ij} - t_{ij})^2
			l = l.Add(y[i][j].Sub(t[i][j]).Pow(2))
		}
	}
	l = l.Div(data.NewVariable(2 * float64(rows)))
	return
}

// Cross Entropy Loss
type CrossEntropyLoss struct {
}

func NewCrossEntropyLoss() (l Loss) {
	return &CrossEntropyLoss{}
}

func (m *CrossEntropyLoss) Calculate(y [][]*data.Variable, t [][]*data.Variable) (l *data.Variable) {
	l = data.NewVariable(0)
	rows := len(y)
	cols := len(y[0])
	for i := 0; i < rows; i++ {
		s := data.NewVariable(0)
		for j := 0; j < cols; j++ {
			// s += e^{y_{ij}}
			s = s.Add(y[i][j].Exp())
		}
		// loss += log(e^{y_{ik}} / s)
		l = l.Add(y[i][int(t[i][0].Val)].Exp().Div(s).Log())
	}
	l = l.Mul(data.NewVariable(-1)).Div(data.NewVariable(float64(rows)))
	return
}
