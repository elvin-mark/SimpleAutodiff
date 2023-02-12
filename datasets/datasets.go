package datasets

import (
	"math"
	"math/rand"

	"github.com/elvin-mark/SimpleAutodiff/data"
)

func XORDataset(datasetType int) (x [][]*data.Variable, y [][]*data.Variable) {
	x = [][]*data.Variable{
		{data.NewVariable(0.0), data.NewVariable(0.0)},
		{data.NewVariable(1.0), data.NewVariable(0.0)},
		{data.NewVariable(0.0), data.NewVariable(1.0)},
		{data.NewVariable(1.0), data.NewVariable(1.0)},
	}
	if datasetType == 0 {
		y = [][]*data.Variable{
			{data.NewVariable(1.0), data.NewVariable(0.0)},
			{data.NewVariable(0.0), data.NewVariable(1.0)},
			{data.NewVariable(0.0), data.NewVariable(1.0)},
			{data.NewVariable(1.0), data.NewVariable(0.0)},
		}
	} else {
		y = [][]*data.Variable{
			{data.NewVariable(0.0)},
			{data.NewVariable(1.0)},
			{data.NewVariable(1.0)},
			{data.NewVariable(0.0)},
		}
	}

	return
}

func Toy2DDataset(N int) (x [][]*data.Variable, t [][]*data.Variable) {
	m := rand.NormFloat64()
	b := rand.NormFloat64()
	for i := 0; i < N; i++ {
		tmp := []*data.Variable{
			data.NewVariable(rand.NormFloat64()),
			data.NewVariable(rand.NormFloat64()),
		}
		if m*tmp[0].Val+b > tmp[1].Val {
			t = append(t, []*data.Variable{data.NewVariable(1)})
		} else {
			t = append(t, []*data.Variable{data.NewVariable(0)})
		}
		x = append(x, tmp)
	}
	return
}

func ToyCircleDataset(N int, radius float64) (x [][]*data.Variable, t [][]*data.Variable) {
	for i := 0; i < N; i++ {
		tmp := []*data.Variable{
			data.NewVariable(rand.NormFloat64()),
			data.NewVariable(rand.NormFloat64()),
		}
		r := math.Sqrt(tmp[0].Val*tmp[0].Val + tmp[1].Val*tmp[1].Val)
		if r > radius {
			t = append(t, []*data.Variable{data.NewVariable(1)})
		} else {
			t = append(t, []*data.Variable{data.NewVariable(0)})
		}
		x = append(x, tmp)
	}
	return
}
