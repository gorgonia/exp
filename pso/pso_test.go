package pso

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_XOR(t *testing.T) {
	trainingData := []Data{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	inputs := 2     // 2 cols
	hiddenSize := 5 // for shits and giggles, why not have a brodcast
	outputs := 1

	iterations := 10000
	p := newParticle(inputs, hiddenSize, outputs)
	p.Train(trainingData, iterations)

	testData := []Data{
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 0}, []float64{0}},
	}

	for _, tt := range testData {
		expected := tt.Output
		actual := p.Predict(tt.Input...)
		assert.Equal(t, expected, actual)
	}
}
