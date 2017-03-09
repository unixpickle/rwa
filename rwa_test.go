package rwa

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestRWAOutput(t *testing.T) {
	block := NewRWA(anyvec32.CurrentCreator(), 3, 2)
	paramVals := [][]float32{
		{0.074941, -1.132446},
		{0.63997, -0.21826, 0.88730, 0.25009, 0.11063, 0.72248},
		{0.051791, 0.479197},
		{0.0099470, 0.8399450, -0.2081483, 0.9820264, 0.3257544, 0.1064337},
		{-0.515952, 0.055721},
		{0.97504, 0.35937, -0.37616, 0.69398},
		{0.60679, 0.44104},
		{0.90352, 0.25258, 0.76472, 0.73948, 0.98564, 0.20552},
		{-0.50516, 0.73835},
		{0.26337, 0.28690, -0.29784, -0.79788},
		{-0.017014, 0.803208},
	}
	for i, x := range block.Parameters() {
		x.Vector.SetData(paramVals[i])
	}

	seq := anyseq.ConstSeqList(anyvec32.CurrentCreator(), [][]anyvec.Vector{
		{
			anyvec32.MakeVectorData([]float32{0.29989, 0.36990, 0.50296}),
			anyvec32.MakeVectorData([]float32{0.37573, 0.29873, 0.22233}),
			anyvec32.MakeVectorData([]float32{0.45905, 0.14858, 0.78369}),
		},
	})
	out := anyrnn.Map(seq, block).Output()
	expected := [][]float32{
		{0.049205, 0.329647},
		{0.12153, 0.39991},
		{0.20841, 0.49782},
	}
	for i, aVec := range out {
		a := aVec.Packed.Data().([]float32)
		x := expected[i]
		for j := range a {
			if math.IsNaN(float64(a[j])) {
				t.Errorf("step %d component %d: got NaN", i, j)
			} else if math.Abs(float64(x[j]-a[j])) > 1e-3 {
				t.Errorf("step %d component %d: expected %v but got %v",
					i, j, x[j], a[j])
			}
		}
	}
}

func TestRWAGradients(t *testing.T) {
	inSeq, inVars := randomTestSequence(3)
	block := NewRWA(anyvec32.CurrentCreator(), 3, 2)
	if len(block.Parameters()) != 11 {
		t.Errorf("expected 11 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return anyrnn.Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}

// randomTestSequence is borrowed from
// https://github.com/unixpickle/anynet/blob/6a8bd570b702861f3c1260a6916723beea6bf296/anyrnn/layer_test.go#L34
func randomTestSequence(inSize int) (anyseq.Seq, []*anydiff.Var) {
	inVars := []*anydiff.Var{}
	inBatches := []*anyseq.ResBatch{}

	presents := [][]bool{{true, true, true}, {true, false, true}}
	numPres := []int{3, 2}
	chunkLengths := []int{2, 3}

	for chunkIdx, pres := range presents {
		for i := 0; i < chunkLengths[chunkIdx]; i++ {
			vec := anyvec32.MakeVector(inSize * numPres[chunkIdx])
			anyvec.Rand(vec, anyvec.Normal, nil)
			v := anydiff.NewVar(vec)
			batch := &anyseq.ResBatch{
				Packed:  v,
				Present: pres,
			}
			inVars = append(inVars, v)
			inBatches = append(inBatches, batch)
		}
	}
	return anyseq.ResSeq(anyvec32.CurrentCreator(), inBatches), inVars
}
