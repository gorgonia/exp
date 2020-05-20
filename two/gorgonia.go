package g2

import (
	"gorgonia.org/tensor"
)

func (e *Graph) MatMul(a, b, preallocated tensor.Tensor) error {
	aEng := a.Engine().(*Graph) // TODO ERR
	bEng := b.Engine().(*Graph) // TODO ERR
	cEng := preallocated.Engine().(*Graph)

	aName := aEng.nameOf(a)
	bName := bEng.nameOf(b)

	cEng.idOrInsert(preallocated) // TODO what if preallocated already exists as a node? (i.e WithReuse was called)
	cName := aName + "×" + bName
	cEng.name(preallocated, cName)

	return e.StdEng.MatMul(a, b, preallocated)
}

func (e *Graph) AddScalar(a Tensor, b interface{}, leftTensor, opts ...tensor.FuncOpt) (Tensor, error) {
	aEng := a.Engine().(*Graph)

}

type Graph struct {
	tensor.StdEng

	nodes []Node

	// flags
	isStdEng bool
}

func NewGraph() *Graph { return &Graph{} }

func (g *Graph) id(t tensor.Tensor) int64 {
	// search backwards because it's more probable that you're using newer created nodes
	for i := len(g.nodes) - 1; i >= 0; i-- {
		if t == g.nodes[i].Tensor {
			return int64(i)
		}
	}
	return -1
}

func (g *Graph) idOrInsert(t tensor.Tensor) int64 {
	id := g.id(t)
	if id < 0 {
		return g.insert(t)
	}
	return id
}

func (g *Graph) insert(t tensor.Tensor) int64 {
	l := len(g.nodes)
	g.nodes = append(g.nodes, Node{Tensor: t})
	g.nodes[l].id = int64(l)
	return int64(l)
}

func (g *Graph) nameOf(t tensor.Tensor) string {
	id := g.id(t)
	// TODO: if not found?
	return g.nodes[id].name
}

func (g *Graph) name(t tensor.Tensor, s string) error {
	id := g.id(t)
	g.nodes[id].name = s

	return nil //TODO: if not found
}

func (g *Graph) nodeOf(t tensor.Tensor) Node {
	id := g.id(t)
	return g.nodes[id] // TODO: if not found?
}

func (g *Graph) Nodes() (retVal []string) {
	for i := range g.nodes {
		retVal = append(retVal, g.nodes[i].name)
	}
	return retVal
}

func WithName(name string) tensor.ConsOpt {
	return func(t tensor.Tensor) {
		en := t.Engine()
		if e, ok := en.(*Graph); ok {
			id := e.idOrInsert(t)
			e.nodes[id].name = name
		}

	}
}

func AsVar() tensor.ConsOpt {
	return func(t tensor.Tensor) {
		en := t.Engine()
		if e, ok := en.(*Graph); ok {
			e.idOrInsert(t)
		}
	}
}

// Node implements tensor.Tensor
type Node struct {
	tensor.Tensor

	id   int64
	name string
}

func New(g *Graph, name string, opts ...tensor.ConsOpt) Node {
	consOpts := append([]tensor.ConsOpt{tensor.WithEngine(g), AsVar(), WithName(name)}, opts...)
	t := tensor.New(consOpts...)
	return g.nodeOf(t)
}
