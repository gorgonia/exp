package pso

import "math/rand"

const (
	inertiaWeight       = 0.729
	cognitiveWeight     = 1.49445
	maxDeathProbability = 0.1
	randRange           = 10.0
)

func randomizeUniform(arr []float64, positiveRange float64) {
	for i := range arr {
		arr[i] = rand.Float64()*positiveRange - positiveRange
	}
}
