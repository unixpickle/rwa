// Package rwa implements the Recurrent Weighted Average
// RNN defined in https://arxiv.org/pdf/1703.01253.pdf.
package rwa

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&RWA{}).SerializerType(), DeserializeRWA)
}

// RWA is a Recurrent Weighted Average RNN block.
type RWA struct {
	// SquashFunc is used to squash the rolling average.
	SquashFunc anynet.Layer

	// Init is the unsquashed start state.
	Init *anydiff.Var

	// Encoder is u(x) from the paper.
	Encoder *anynet.FC

	// Masker is g(x,h) from the paper.
	Masker *anynet.AddMixer

	// Context is a(x,h) from the paper.
	Context *anynet.AddMixer
}

// NewRWA creates a randomized RWA with the given number
// of inputs and hidden units.
func NewRWA(c anyvec.Creator, inSize, stateSize int) *RWA {
	oneVec := c.MakeVector(stateSize)
	oneVec.AddScaler(c.MakeNumeric(1))
	return &RWA{
		SquashFunc: anynet.Tanh,
		Init:       anydiff.NewVar(c.MakeVector(stateSize)),
		Encoder:    anynet.NewFC(c, inSize, stateSize),
		Masker: &anynet.AddMixer{
			In1: anynet.NewFC(c, inSize, stateSize),
			In2: anynet.NewFC(c, stateSize, stateSize),
			Out: anynet.Tanh,
		},
		Context: &anynet.AddMixer{
			In1: anynet.NewFC(c, inSize, stateSize),
			In2: anynet.NewFC(c, stateSize, stateSize),
			Out: anynet.Net{},
		},
	}
}

// DeserializeRWA deserializes an RWA.
func DeserializeRWA(d []byte) (*RWA, error) {
	var res RWA
	var initVec *anyvecsave.S
	err := serializer.DeserializeAny(d, &res.SquashFunc, &initVec, &res.Encoder,
		&res.Masker, &res.Context)
	if err != nil {
		return nil, err
	}
	res.Init = anydiff.NewVar(initVec.Vector)
	return &res, nil
}

// Start generates an initial *State.
func (r *RWA) Start(n int) anyrnn.State {
	c := r.Init.Vector.Creator()
	return &State{
		Hidden: anyrnn.NewVecState(r.Init.Vector, n),
		Num:    anyrnn.NewVecState(c.MakeVector(r.Init.Vector.Len()), n),
		Denom:  anyrnn.NewVecState(c.MakeVector(r.Init.Vector.Len()), n),
	}
}

// PropagateStart propagates through the start state.
func (r *RWA) PropagateStart(s anyrnn.StateGrad, g anydiff.Grad) {
	s.(*State).Hidden.PropagateStart(r.Init, g)
}

// Step performs a timestep.
func (r *RWA) Step(s anyrnn.State, in anyvec.Vector) anyrnn.Res {
	batch := s.Present().NumPresent()
	state := s.(*State)
	c := in.Creator()

	inPool := anydiff.NewVar(in)
	hiddenPool := anydiff.NewVar(state.Hidden.Vector)
	numPool := anydiff.NewVar(state.Num.Vector)
	denomPool := anydiff.NewVar(state.Denom.Vector)

	hidden := r.SquashFunc.Apply(hiddenPool, batch)

	intermed := anydiff.Fuse(hidden)
	outs := anydiff.PoolMulti(intermed, func(reses []anydiff.Res) anydiff.MultiRes {
		hidden := reses[0]
		weight := anydiff.Exp(r.Context.Mix(inPool, hidden, batch))
		inEnc := r.Encoder.Apply(inPool, batch)
		inMask := anydiff.Tanh(r.Masker.Mix(inPool, hidden, batch))
		z := anydiff.Mul(inEnc, inMask)

		intermed := anydiff.Fuse(weight, z)
		return anydiff.PoolMulti(intermed, func(reses []anydiff.Res) anydiff.MultiRes {
			weight := reses[0]
			z := reses[1]
			newNum := anydiff.Add(numPool, anydiff.Mul(z, weight))
			newDenom := anydiff.Add(denomPool, weight)
			intermed := anydiff.Fuse(newNum, newDenom)
			return anydiff.PoolMulti(intermed, func(reses []anydiff.Res) anydiff.MultiRes {
				newNum := reses[0]
				newDenom := reses[1]
				invDenom := anydiff.Pow(newDenom, c.MakeNumeric(-1))
				unsquashedHidden := anydiff.Mul(newNum, invDenom)
				intermed := anydiff.Fuse(unsquashedHidden)
				return anydiff.PoolMulti(intermed, func(reses []anydiff.Res) anydiff.MultiRes {
					squashedHidden := r.SquashFunc.Apply(reses[0], batch)
					return anydiff.Fuse(squashedHidden, unsquashedHidden, newNum, newDenom)
				})
			})
		})
	})

	return &blockRes{
		V:      anydiff.NewVarSet(r.Parameters()...),
		OutVec: outs.Outputs()[0],
		OutState: &State{
			Hidden: &anyrnn.VecState{
				PresentMap: state.Present(),
				Vector:     outs.Outputs()[1],
			},
			Num: &anyrnn.VecState{
				PresentMap: state.Present(),
				Vector:     outs.Outputs()[2],
			},
			Denom: &anyrnn.VecState{
				PresentMap: state.Present(),
				Vector:     outs.Outputs()[3],
			},
		},
		InPool:     inPool,
		HiddenPool: hiddenPool,
		NumPool:    numPool,
		DenomPool:  denomPool,
		Res:        outs,
	}
}

// Parameters returns the block's parameters.
func (r *RWA) Parameters() []*anydiff.Var {
	res := []*anydiff.Var{r.Init}
	for _, x := range []interface{}{r.SquashFunc, r.Encoder, r.Masker, r.Context} {
		if p, ok := x.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// an RWA with the serializer package.
func (r *RWA) SerializerType() string {
	return "github.com/unixpickle/rwa.RWA"
}

// Serialize serializes an RWA.
func (r *RWA) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		r.SquashFunc,
		&anyvecsave.S{Vector: r.Init.Vector},
		r.Encoder,
		r.Masker,
		r.Context,
	)
}

// State stores the hidden state of an RWA block or the
// gradient of such a state.
//
// The Num and Denom fields, corresponding to the rolling
// numerators and denominators respectively, begin as 0.
//
// The Hidden field stores the previous, unsquashed hidden
// state.
//
// It is necessary for the Hidden field to be separate
// from the Num and Denom fields so that the network can
// be evaluated at the first timestep.
type State struct {
	Hidden *anyrnn.VecState
	Num    *anyrnn.VecState
	Denom  *anyrnn.VecState
}

func (s *State) Present() anyrnn.PresentMap {
	return s.Hidden.Present()
}

func (s *State) Reduce(p anyrnn.PresentMap) anyrnn.State {
	return &State{
		Hidden: s.Hidden.Reduce(p).(*anyrnn.VecState),
		Num:    s.Num.Reduce(p).(*anyrnn.VecState),
		Denom:  s.Denom.Reduce(p).(*anyrnn.VecState),
	}
}

func (s *State) Expand(p anyrnn.PresentMap) anyrnn.StateGrad {
	return &State{
		Hidden: s.Hidden.Expand(p).(*anyrnn.VecState),
		Num:    s.Num.Expand(p).(*anyrnn.VecState),
		Denom:  s.Denom.Expand(p).(*anyrnn.VecState),
	}
}

type blockRes struct {
	OutState *State
	OutVec   anyvec.Vector
	V        anydiff.VarSet

	InPool     *anydiff.Var
	HiddenPool *anydiff.Var
	NumPool    *anydiff.Var
	DenomPool  *anydiff.Var

	Res anydiff.MultiRes
}

func (b *blockRes) State() anyrnn.State {
	return b.OutState
}

func (b *blockRes) Output() anyvec.Vector {
	return b.OutVec
}

func (b *blockRes) Vars() anydiff.VarSet {
	return b.V
}

func (b *blockRes) Propagate(u anyvec.Vector, s anyrnn.StateGrad, g anydiff.Grad) (anyvec.Vector,
	anyrnn.StateGrad) {
	c := u.Creator()
	down := make([]anyvec.Vector, 4)
	down[0] = u
	if s != nil {
		st := s.(*State)
		down[1] = st.Hidden.Vector
		down[2] = st.Num.Vector
		down[3] = st.Denom.Vector
	} else {
		down[1] = c.MakeVector(b.OutState.Hidden.Vector.Len())
		down[2] = c.MakeVector(b.OutState.Num.Vector.Len())
		down[3] = c.MakeVector(b.OutState.Denom.Vector.Len())
	}
	for _, x := range b.pools() {
		g[x] = c.MakeVector(x.Vector.Len())
	}
	b.Res.Propagate(down, g)

	inDown := g[b.InPool]
	stateDown := &State{
		Hidden: &anyrnn.VecState{
			PresentMap: b.OutState.Present(),
			Vector:     g[b.HiddenPool],
		},
		Num: &anyrnn.VecState{
			PresentMap: b.OutState.Present(),
			Vector:     g[b.NumPool],
		},
		Denom: &anyrnn.VecState{
			PresentMap: b.OutState.Present(),
			Vector:     g[b.DenomPool],
		},
	}

	for _, x := range b.pools() {
		delete(g, x)
	}

	return inDown, stateDown
}

func (b *blockRes) pools() []*anydiff.Var {
	return []*anydiff.Var{b.InPool, b.HiddenPool, b.NumPool, b.DenomPool}
}
