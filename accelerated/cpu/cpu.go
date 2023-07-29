package cpu

import "github.com/haormj/llama2/accelerated"

type CPU struct {
}

// MatMul implements accelerated.Backend.
func (*CPU) MatMul(xout []float32, x []float32, w []float32, n int, d int) error {
	var val float32

	for i := 0; i < d; i++ {
		val = 0

		for j := 0; j < n; j++ {
			val += w[i*n+j] * x[j]
		}

		xout[i] = val
	}

	return nil
}

// Release implements accelerated.Backend.
func (*CPU) Release() error {
	return nil
}

// SetupContext implements accelerated.Backend.
func (*CPU) SetupContext() error {
	return nil
}

var _ accelerated.Backend = &CPU{}
