package nn

import (
	data "github.com/elvin-mark/SimpleAutodiff/data"
)

type Loss interface {
	Calculate(y [][]*data.Variable, t [][]*data.Variable) (l *data.Variable)
}

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
			l = l.Add(y[i][j].Sub(t[i][j]).Pow(2))
		}
	}
	l = l.Div(data.NewVariable(2 * float64(rows)))
	return
}

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
			s = s.Add(y[i][j].Exp())
		}
		l = l.Add(y[i][int(t[i][0].Val)].Exp().Div(s).Log())
	}
	l = l.Mul(data.NewVariable(-1)).Div(data.NewVariable(float64(rows)))
	return
}
