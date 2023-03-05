package data

import (
	"math/rand"
)

func NewVector(numElems int) (b []*Variable) {
	for j := 0; j < numElems; j++ {
		b = append(b, NewVariable(rand.NormFloat64()))
	}
	return
}

func NewMatrix(rows int, cols int) (W [][]*Variable) {
	for i := 0; i < rows; i++ {
		tmp := []*Variable{}
		for j := 0; j < cols; j++ {
			tmp = append(tmp, NewVariable(rand.NormFloat64()))
		}
		W = append(W, tmp)
	}
	return
}

func GEMM(x [][]*Variable, W [][]*Variable, b []*Variable) (o [][]*Variable) {
	rows1 := len(x)
	cols1 := len(x[0])
	rows2 := len(W)
	cols2 := len(W[0])
	numElems := len(b)
	if cols1 != rows2 || cols2 != numElems {
		panic("dim does not match")
	}
	for k := 0; k < rows1; k++ {
		tmp := []*Variable{}
		for j := 0; j < cols2; j++ {
			s := NewVariable(0)
			for i := 0; i < rows2; i++ {
				s = s.Add(x[k][i].Mul(W[i][j]))
			}
			s = s.Add(b[j])
			tmp = append(tmp, s)
		}
		o = append(o, tmp)
	}
	return
}
