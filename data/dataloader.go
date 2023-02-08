package data

import "math"

type DataLoader struct {
	x         [][]*Variable
	y         [][]*Variable
	batchSize int
	cursor    int
}

func NewDataLoader(x [][]*Variable, y [][]*Variable, batchSize int) DataLoader {
	return DataLoader{
		x:         x,
		y:         y,
		batchSize: batchSize,
		cursor:    0,
	}
}

func (dl *DataLoader) HasNext() bool {
	if dl.cursor >= len(dl.x) {
		return false
	}
	return true
}

func (dl *DataLoader) Next() (x [][]*Variable, y [][]*Variable) {
	x = dl.x[dl.cursor : dl.cursor+dl.batchSize]
	y = dl.y[dl.cursor : dl.cursor+dl.batchSize]
	dl.cursor = dl.cursor + dl.batchSize
	return
}

func (dl *DataLoader) Rewind() {
	dl.cursor = 0
}

func (dl *DataLoader) Len() int {
	return int(math.Ceil(float64(len(dl.x)) / float64(dl.batchSize)))
}