package pso

import (
	"log"
	"math"
	"math/rand"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

//Data x
type Data struct {
	Input  []float64
	Output []float64
}

type particle struct {
	feats              int
	n                  *nn
	velocities         []float64
	bestErrorLoss      float64
	bestLocalPositions []float64
}

func newParticle(feats, hiddenSize, outputs, bucketSize int) *particle {
	n := newNN(feats, hiddenSize, bucketSize)

	var lossVal, predVal G.Value
	G.Read(n.pred, &predVal) // read the predicted value out into something that can be accessed by Go
	G.Read(n.loss, &lossVal) // read the loss value out into something that can be accessed by Go

	p := particle{
		feats:         feats,
		n:             n,
		bestErrorLoss: math.MaxFloat64,
	}
	p.bestLocalPositions = p.n.exportPositions()
	p.velocities = make([]float64, len(p.bestLocalPositions))
	randomizeUniform(p.velocities, randRange)
	return &p
}

func (p *particle) Train(trainingData []Data, iterations int) {
	n := p.n

	trainingDataSize := len(trainingData)
	xSize := len(trainingData[0].Input)
	ySize := len(trainingData[0].Output)
	Xs := make([]float64, trainingDataSize*xSize)
	Ys := make([]float64, trainingDataSize*ySize)
	for i, d := range trainingData {
		for j, x := range d.Input {
			Xs[i*xSize+j] = x
		}
		for j, y := range d.Output {
			Ys[i*ySize+j] = y
		}
	}

	// training
	yVal := T.New(T.WithShape(trainingDataSize), T.WithBacking(Ys))          // make Ys into a Tensor. No additional allocations
	xVal := T.New(T.WithShape(trainingDataSize, p.feats), T.WithBacking(Xs)) // make Xs into a Tensor. No additional allocations are made.

	G.Let(n.x, xVal)
	G.Let(n.y, yVal)
	m := G.NewTapeMachine(n.g)

	for i := 0; i < iterations; i++ {
		currentPositions := n.exportPositions()
		revisedPositions := make([]float64, len(currentPositions)) //just for debugging I'm not overriding the currentPosition slice

		rowCount := len(trainingData)
		xTestVal := T.New(T.WithShape(rowCount, p.feats), T.WithBacking(T.Random(T.Float64, rowCount*p.feats)))
		if err := n.fwd(m, xTestVal); err != nil {
			log.Fatal(err)
		}
		m.Reset()

		lossf64 := n.loss.Value().Data().(float64)

		if lossf64 < p.bestErrorLoss {
			copy(p.bestLocalPositions, currentPositions)
			p.bestErrorLoss = lossf64
			log.Printf("<%d> New best local error found: %f", i, lossf64)
		}

		//update positions+1
		for i, currentVelocity := range p.velocities {
			currentPosition := currentPositions[i]
			bestLocalPosition := p.bestLocalPositions[i]

			oldVelocityFactor := inertiaWeight * currentVelocity

			localRandomness := rand.Float64()
			bestLocationDelta := bestLocalPosition - currentPosition
			localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

			revisedVelocity := oldVelocityFactor + localPositionFactor
			p.velocities[i] = revisedVelocity
			revisedPositions[i] = math.Max(-randRange, math.Min(currentPosition+revisedVelocity, randRange))
		}

		n.importPositions(revisedPositions)

		dieThreshold := maxDeathProbability * (1 - float64(i)/float64(iterations))
		dieF := rand.Float64()
		shouldDie := dieF < dieThreshold
		if shouldDie {
			log.Printf("<%d> Death %f:%f", i, dieF, dieThreshold)
			randomizeUniform(revisedPositions, randRange)
			randomizeUniform(p.velocities, 1)
			n.importPositions(revisedPositions)
		}
	}

	log.Printf("Final error: %f", p.bestErrorLoss)
}

func (p *particle) Predict(inputs ...float64) []float64 {
	xTestVal := T.New(T.WithShape(1, p.feats), T.WithBacking(T.Random(T.Float64, len(inputs))))
	G.Let(p.n.x, xTestVal)

	m := G.NewTapeMachine(p.n.g)
	if err := p.n.fwd(m, xTestVal); err != nil {
		log.Fatal(err)
	}
	m.Reset()

	output := p.n.last.Value().Data().(float64)
	return []float64{output}
}
