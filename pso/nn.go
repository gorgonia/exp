package pso

import (
	G "gorgonia.org/gorgonia"
)

var must = G.Must

type layer struct {
	w, b *G.Node
}

type nn struct {
	g                *G.ExprGraph
	x, y             *G.Node
	w, b, w2, b2     *G.Node
	layers           []layer // something like this. Then the newNN() func simply has to change a bit, that's all.
	pred, loss, last *G.Node
}

func newNN(feats, hiddenSize, bucketSize int) *nn {
	g := G.NewGraph()
	y := G.NewVector(g, G.Float64, G.WithShape(bucketSize), G.WithName("Y"))
	x := G.NewMatrix(g, G.Float64, G.WithShape(bucketSize, feats), G.WithName("X"))
	w := G.NewMatrix(g, G.Float64, G.WithShape(feats, hiddenSize), G.WithName("W"), G.WithInit(G.GlorotU(1)))
	b := G.NewMatrix(g, G.Float64, G.WithShape(bucketSize, hiddenSize), G.WithName("b"), G.WithInit(G.Zeroes()))

	l0 := must(G.Add(must(G.Mul(x, w)), b))
	l0 = must(G.Tanh(l0))

	w2 := G.NewVector(g, G.Float64, G.WithShape(hiddenSize), G.WithName("W2"), G.WithInit(G.GlorotU(1)))
	b2 := G.NewVector(g, G.Float64, G.WithShape(bucketSize), G.WithName("b2"), G.WithInit(G.Zeroes()))

	last := must(G.Add(must(G.Mul(l0, w2)), b2))
	last = must(G.Tanh(last))

	loss := must(G.Square(must(G.Sub(y, last))))
	loss = must(G.Sum(loss))

	n := &nn{
		g:    g,
		x:    x,
		y:    y,
		w:    w,
		b:    b,
		w2:   w2,
		b2:   b2,
		pred: last,
		loss: loss,
		last: last,

		layers: []layer{
			{w, b},
			{w2, b2},
		},
	}
	return n
}

func (n *nn) fwd(m G.VM, x G.Value) error {
	// change the metadata of x first. This may cause panics. They're usually for good reasons.
	// if not, file a issue.
	G.WithShape(x.Shape().Clone()...)(n.x)

	// set the value of x
	if err := G.Let(n.x, x); err != nil {
		return nil
	}

	return m.RunAll()
}

func (n *nn) train(x, y G.Value) error {
	// update x and y shapes
	G.WithShape(x.Shape().Clone()...)(n.x)
	G.WithShape(y.Shape().Clone()...)(n.y)

	G.Let(n.x, x)
	G.Let(n.y, y)
	m := G.NewTapeMachine(n.g)
	if err := m.RunAll(); err != nil {
		return err
	}
	// do something? PSO updates here
	m.Reset() // reset the machine state

	return nil
}

func (n *nn) exportPositions() []float64 {
	//generalised for any number of layers
	arr := []float64{}

	for _, l := range n.layers {
		w := l.w.Value().Data().([]float64)
		b := l.b.Value().Data().([]float64)
		arr = append(arr, w...)
		arr = append(arr, b...)
	}

	return arr
}

func (n *nn) importPositions(positions []float64) {
	var start int
	for _, l := range n.layers {
		//TODO Xuanyi magic
		w := l.w.Value().Data().([]float64)
		copy(w, positions[start:])
		start += len(w)
		b := l.b.Value().Data().([]float64)
		copy(b, positions[start:])
		start += len(b) //MAGIC! JAZZHANDS
	}
}

type errorProp func(loss *G.Node, wrt ...*G.Node) (G.Nodes, error)
