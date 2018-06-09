package pso

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_XOR(t *testing.T) {
	rand.Seed(1337)
	trainingData := []Data{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	inputs := len(trainingData[0].Input) // 2 cols
	hiddenSize := 5                      // for shits and giggles, why not have a brodcast
	outputs := len(trainingData[0].Output)

	iterations := 10000
	p := newParticle(inputs, hiddenSize, outputs, len(trainingData))
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
