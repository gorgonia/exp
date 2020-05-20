package g2

import "gorgonia.org/tensor"

func MatMul(a, b Node) (Node, error) {
	res, err := tensor.MatMul(a.Tensor, b.Tensor)
	if err != nil {
		return Node{id: -1}, err
	}
	eng := a.Engine().(*Graph)
	return eng.nodeOf(res), nil
}
