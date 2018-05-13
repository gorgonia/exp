package pso

import (
	"gorgonia.org/gorgonia"
)

var must = gorgonia.Must

type layer struct {
	w, b *gorgonia.Node
}

type nn struct {
	g                *gorgonia.ExprGraph
	x, y             *gorgonia.Node
	w, b, w2, b2     *gorgonia.Node
	layers           []layer // something like this. Then the newNN() func simply has to change a bit, that's all.
	pred, loss, last *gorgonia.Node
}

func newNN(feats, hiddenSize int) *nn {
	examplesN := 4 // dummy value, can be changed

	g := gorgonia.NewGraph()
	y := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(examplesN), gorgonia.WithName("Y"))
	x := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(examplesN, feats), gorgonia.WithName("X"))
	w := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feats, hiddenSize), gorgonia.WithName("W"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	b := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(examplesN, hiddenSize), gorgonia.WithName("b"), gorgonia.WithInit(gorgonia.Zeroes()))

	l0 := must(gorgonia.Add(must(gorgonia.Mul(x, w)), b))
	l0 = must(gorgonia.Tanh(l0))

	w2 := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize), gorgonia.WithName("W2"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	b2 := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(examplesN), gorgonia.WithName("b2"), gorgonia.WithInit(gorgonia.Zeroes()))

	last := must(gorgonia.Add(must(gorgonia.Mul(l0, w2)), b2))
	last = must(gorgonia.Tanh(last))

	loss := must(gorgonia.Square(must(gorgonia.Sub(y, last))))
	loss = must(gorgonia.Sum(loss))

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

func (n *nn) fwd(m gorgonia.VM, x gorgonia.Value) error {
	// change the metadata of x first. This may cause panics. They're usually for good reasons.
	// if not, file a issue.
	gorgonia.WithShape(x.Shape().Clone()...)(n.x)

	// set the value of x
	if err := gorgonia.Let(n.x, x); err != nil {
		return nil
	}

	return m.RunAll()
}

func (n *nn) train(x, y gorgonia.Value) error {
	// update x and y shapes
	gorgonia.WithShape(x.Shape().Clone()...)(n.x)
	gorgonia.WithShape(y.Shape().Clone()...)(n.y)

	gorgonia.Let(n.x, x)
	gorgonia.Let(n.y, y)
	m := gorgonia.NewTapeMachine(n.g)
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

type errorProp func(loss *gorgonia.Node, wrt ...*gorgonia.Node) (gorgonia.Nodes, error)
