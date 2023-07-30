package blackcl

import (
	_ "embed"
	"fmt"

	"github.com/haormj/llama2/accelerated"
	"gitlab.com/microo8/blackcl"
)

//go:embed matmul.cl
var matmulSrc string

const localGroupSize = 64

type OpenCL struct {
	device *blackcl.Device
	kernel *blackcl.Kernel

	bufferCache map[string]map[int]*blackcl.Vector
}

func New() *OpenCL {
	return &OpenCL{
		bufferCache: make(map[string]map[int]*blackcl.Vector),
	}
}

// Release implements opencl.OpenCL.
func (o *OpenCL) Release() error {
	if err := o.device.Release(); err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to release device: %w", err)
	}

	for _, bufferMap := range o.bufferCache {
		for _, buffer := range bufferMap {
			buffer.Release()
		}
	}

	return nil
}

func (o *OpenCL) Buffer(bufferTag string, size int) (*blackcl.Vector, error) {
	if _, ok := o.bufferCache[bufferTag]; !ok {
		o.bufferCache[bufferTag] = make(map[int]*blackcl.Vector)
	}

	buffer, ok := o.bufferCache[bufferTag][size]
	if !ok {
		buffer, err := o.device.NewVector(size)

		if err != nil {
			return nil, fmt.Errorf("accelerated/blackcl: failed to create buffer: %w", err)
		}

		o.bufferCache[bufferTag][size] = buffer

		return buffer, nil
	}

	return buffer, nil
}

// MatMul implements opencl.OpenCL.
func (o *OpenCL) MatMul(xout []float32, x []float32, w []float32, n int, d int) error {
	if len(xout) != d {
		return fmt.Errorf("accelerated/blackcl: xout length must be %d, got %d", d, len(xout))
	}

	xoutDev, err := o.Buffer("xout", len(xout))
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to create xout device vector: %w", err)
	}

	xDev, err := o.Buffer("x", len(x))
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to create x device vector: %w", err)
	}

	xDevCopyComplete := xDev.Copy(x)

	wDev, err := o.Buffer("w", len(w))
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to create w device vector: %w", err)
	}

	wDevCopyComplete := wDev.Copy(w)

	if err := <-xDevCopyComplete; err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to copy x to device: %w", err)
	}

	if err := <-wDevCopyComplete; err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to copy w to device: %w", err)
	}

	globalSize := d
	localSize := 1

	if globalSize%localGroupSize != 0 {
		globalSize /= localGroupSize
		localSize = localGroupSize
	}

	if err = <-o.kernel.Global(globalSize).Local(localSize).Run(xoutDev, xDev, wDev, uint32(n)); err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to run kernel: %w", err)
	}

	xoutHost, err := xoutDev.Data()
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to get xout data: %w", err)
	}

	for i := 0; i < d; i++ {
		xout[i] = xoutHost[i]
	}

	return nil
}

// SetupContext implements opencl.OpenCL.
func (o *OpenCL) SetupContext() error {
	var err error

	o.device, err = blackcl.GetDefaultDevice()
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to get default device: %w", err)
	}

	o.device.AddProgram(matmulSrc)
	o.kernel = o.device.Kernel("matmul")

	return nil
}

var _ accelerated.Backend = &OpenCL{}
