package datasets

import "simple_autodiff/data"

func XORDataset() (x [][]*data.Variable, y [][]*data.Variable) {
	x = [][]*data.Variable{
		{data.NewVariable(0.0), data.NewVariable(0.0)},
		{data.NewVariable(1.0), data.NewVariable(0.0)},
		{data.NewVariable(0.0), data.NewVariable(1.0)},
		{data.NewVariable(1.0), data.NewVariable(1.0)},
	}
	y = [][]*data.Variable{
		{data.NewVariable(1.0), data.NewVariable(0.0)},
		{data.NewVariable(0.0), data.NewVariable(1.0)},
		{data.NewVariable(0.0), data.NewVariable(1.0)},
		{data.NewVariable(1.0), data.NewVariable(0.0)},
	}
	return
}
