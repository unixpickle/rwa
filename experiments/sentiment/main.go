package main

import (
	"encoding/csv"
	"flag"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/rwa"
	"github.com/unixpickle/serializer"
)

func main() {
	serializer.RegisterTypedDeserializer((&Model{}).SerializerType(), DeserializeModel)
	rand.Seed(time.Now().UnixNano())

	var modelPath string
	var trainingPath string
	var testingPath string
	var batchSize int
	var stepSize float64
	var hidden int
	var lstm bool
	var hybrid bool
	flag.StringVar(&modelPath, "out", "out_net", "output model file")
	flag.StringVar(&trainingPath, "training", "", "training data file")
	flag.StringVar(&testingPath, "testing", "", "testing data file")
	flag.IntVar(&batchSize, "batch", 16, "SGD batch size")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&hidden, "hidden", 384, "number of hidden units")
	flag.BoolVar(&lstm, "lstm", false, "use LSTM instead of RWA")
	flag.BoolVar(&hybrid, "hybrid", false, "use hybrid LSTM-RWA model")
	flag.Parse()

	if trainingPath == "" || testingPath == "" {
		essentials.Die("Required flags: -testing and -training. See -help.")
	}

	c := anyvec32.CurrentCreator()

	var model *Model
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		log.Println("Creating new model...")
		blockMaker := func(in, out int, inScale float32) anyrnn.Block {
			if lstm || (hybrid && inScale > 1) {
				return anyrnn.NewLSTM(c, in, out).ScaleInWeights(inScale)
			} else {
				return rwa.NewRWA(c, in, out).ScaleInWeights(inScale)
			}
		}
		model = &Model{
			Block: anyrnn.Stack{
				blockMaker(0x100, hidden, 16),
				blockMaker(hidden, hidden, 1),
			},
			Out: anynet.Net{
				anynet.NewFC(c, hidden, 1),
			},
		}
	} else {
		log.Println("Loaded existing model.")
	}

	log.Println("Loading samples...")
	training, err := ReadSampleList(trainingPath)
	if err != nil {
		essentials.Die("Load training data:", err)
	}
	validation, err := ReadSampleList(testingPath)
	if err != nil {
		essentials.Die("Load testing data:", err)
	}

	log.Println("Training (ctrl+c to end)...")
	trainer := &anys2s.Trainer{
		Func:    model.Apply,
		Cost:    anynet.SigmoidCE{},
		Params:  model.Parameters(),
		Average: true,
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     training,
		Rater:       anysgd.ConstRater(stepSize),
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			if iter%4 == 0 {
				anysgd.Shuffle(validation)
				bs := essentials.MinInt(batchSize, validation.Len())
				batch, _ := trainer.Fetch(validation.Slice(0, bs))
				cost := anyvec.Sum(trainer.TotalCost(batch.(*anys2s.Batch)).Output())
				log.Printf("iter %d: cost=%v validation=%v", iter, trainer.LastCost, cost)
			} else {
				log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			}
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())

	log.Println("Saving model...")
	if err := serializer.SaveAny(modelPath, model); err != nil {
		essentials.Die("Save model:", err)
	}

	log.Println("Computing validation accuracy...")
	log.Printf("Validation accuracy: %f", model.Accuracy(validation))
}

type Model struct {
	Block anyrnn.Block
	Out   anynet.Layer
}

func DeserializeModel(d []byte) (*Model, error) {
	var res Model
	if err := serializer.DeserializeAny(d, &res.Block, &res.Out); err != nil {
		return nil, err
	}
	return &res, nil
}

func (m *Model) Apply(in anyseq.Seq) anyseq.Seq {
	n := in.Output()[0].NumPresent()
	latent := anyseq.Tail(anyrnn.Map(in, m.Block))
	outs := m.Out.Apply(latent, n)
	return anyseq.ResSeq(in.Creator(), []*anyseq.ResBatch{
		&anyseq.ResBatch{
			Present: in.Output()[0].Present,
			Packed:  outs,
		},
	})
}

func (m *Model) Accuracy(samples *SampleList) float64 {
	// Don't bother using batches; this is never called
	// during the training inner loop, anyway.
	var correct int
	total := samples.Len()
	for i := 0; i < samples.Len(); i++ {
		sample, _ := samples.GetSample(i)
		inSeq := anyseq.ConstSeqList(anyvec32.CurrentCreator(),
			[][]anyvec.Vector{sample.Input})
		out := m.Out.Apply(anyseq.Tail(anyrnn.Map(inSeq, m.Block)), 1)
		actual := anyvec.Sum(out.Output()).(float32) > 0
		desired := samples.Sentiments[i]
		if actual == desired {
			correct++
		}
	}
	return float64(correct) / float64(total)
}

func (m *Model) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, x := range []interface{}{m.Block, m.Out} {
		if x, ok := x.(anynet.Parameterizer); ok {
			res = append(res, x.Parameters()...)
		}
	}
	return res
}

func (m *Model) SerializerType() string {
	return "github.com/unixpickle/rwa/experiments/sentiment.Model"
}

func (m *Model) Serialize() ([]byte, error) {
	return serializer.SerializeAny(m.Block, m.Out)
}

type SampleList struct {
	Tweets     []string
	Sentiments []bool
}

func ReadSampleList(csvFile string) (*SampleList, error) {
	f, err := os.Open(csvFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	var res SampleList
	for _, row := range rows {
		if row[0] == "2" {
			continue
		}
		res.Sentiments = append(res.Sentiments, row[0] == "4")
		res.Tweets = append(res.Tweets, row[len(row)-1])
	}
	return &res, nil
}

func (s *SampleList) Len() int {
	return len(s.Tweets)
}

func (s *SampleList) Swap(i, j int) {
	s.Tweets[i], s.Tweets[j] = s.Tweets[j], s.Tweets[i]
	s.Sentiments[i], s.Sentiments[j] = s.Sentiments[j], s.Sentiments[i]
}

func (s *SampleList) Slice(i, j int) anysgd.SampleList {
	return &SampleList{
		Tweets:     append([]string{}, s.Tweets[i:j]...),
		Sentiments: append([]bool{}, s.Sentiments[i:j]...),
	}
}

func (s *SampleList) Creator() anyvec.Creator {
	return anyvec32.CurrentCreator()
}

func (s *SampleList) GetSample(i int) (*anys2s.Sample, error) {
	data := []byte(s.Tweets[i])
	var input []anyvec.Vector
	for _, x := range data {
		vec := make([]float32, 0x100)
		vec[int(x)] = 1
		input = append(input, anyvec32.MakeVectorData(vec))
	}
	classVal := float32(0)
	if s.Sentiments[i] {
		classVal = 1
	}
	output := []anyvec.Vector{anyvec32.MakeVectorData([]float32{classVal})}
	return &anys2s.Sample{
		Input:  input,
		Output: output,
	}, nil
}
