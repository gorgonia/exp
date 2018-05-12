package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
)

var Xs = []float64{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
}

var Ys = []float64{
	0,
	1,
	1,
	0,
}

var must = gorgonia.Must

func main() {
	g := gorgonia.NewGraph()
	y := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(4), gorgonia.WithName("X"), gorgonia.WithValue(tensor.New(tensor.WithBacking(Ys), tensor.WithShape(4))))
	x := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(4, 2), gorgonia.WithName("X"), gorgonia.WithValue(tensor.New(tensor.WithBacking(Xs), tensor.WithShape(4, 2))))
	w := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(2, 5), gorgonia.WithName("W"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	b := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(4, 5), gorgonia.WithName("b"), gorgonia.WithInit(gorgonia.Zeroes()))

	l0 := must(gorgonia.Add(must(gorgonia.Mul(x, w)), b))
	l0 = must(gorgonia.Tanh(l0))

	w2 := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(5), gorgonia.WithName("W2"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	b2 := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(4), gorgonia.WithName("b2"), gorgonia.WithInit(gorgonia.Zeroes()))

	last := must(gorgonia.Add(must(gorgonia.Mul(l0, w2)), b2))
	last = must(gorgonia.Tanh(last))

	var lossVal, predVal gorgonia.Value
	gorgonia.Read(last, &predVal)
	// loss
	loss := must(gorgonia.Square(must(gorgonia.Sub(y, last))))
	loss = must(gorgonia.Sum(loss))
	gorgonia.Read(loss, &lossVal)

	// SPACE FOR PSO HERE
	// this errorProp interface is just a suggestion.
	var prop errorProp = gorgonia.Grad
	prop(loss, w, b, w2, b2)
	solver := gorgonia.NewVanillaSolver()

	m := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w, b, w2, b2))
	for i := 0; i < 100; i++ {
		if err := m.RunAll(); err != nil {
			log.Fatal(err)
		}
		if err := solver.Step(gorgonia.Nodes{w, b, w2, b2}); err != nil {
			log.Fatal(err)
		}
		log.Printf("I: %d PRED %1.1f LOSS %v", i, predVal, lossVal)
		m.Reset()
	}
}

type errorProp func(loss *gorgonia.Node, wrt ...*gorgonia.Node) (gorgonia.Nodes, error)
