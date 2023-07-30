package opencl

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

func (o *OpenCL) AllocBuffer(bufferTag string, size int) (*blackcl.Vector, error) {
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

// Buffer asynchronously buffers a slice of float32s onto the device.
// Returns a channel of errors that will report when the buffer is done
// copying, or any errors that occurred.
func (o *OpenCL) Buffer(vals []float32) (*blackcl.Vector, <-chan error) {
	errChan := make(chan error, 1)

	buffer, err := o.device.NewVector(len(vals))
	if err != nil {
		errChan <- fmt.Errorf("accelerated/blackcl: failed to create buffer: %w", err)
		return nil, errChan
	}

	go func() {
		if err := <-buffer.Copy(vals); err != nil {
			errChan <- fmt.Errorf("accelerated/blackcl: failed to copy buffer: %w", err)
		} else {
			errChan <- nil
		}

		close(errChan)
	}()

	return buffer, errChan
}

// BufferSize asynchronously buffers a slice of float32s onto the device.
func (o *OpenCL) BufferSize(len int) (*blackcl.Vector, error) {
	buffer, err := o.device.NewVector(len)
	if err != nil {
		return nil, fmt.Errorf("accelerated/blackcl: failed to create buffer: %w", err)
	}

	return buffer, nil
}

// MatMul implements opencl.OpenCL.
func (o *OpenCL) MatMul(xout []float32, x []float32, w []float32, n int, d int) error {
	if len(xout) != d {
		return fmt.Errorf("accelerated/blackcl: xout length must be %d, got %d", d, len(xout))
	}

	xoutDev, err := o.AllocBuffer("xout", len(xout))
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to create xout device vector: %w", err)
	}

	xDev, err := o.AllocBuffer("x", len(x))
	if err != nil {
		return fmt.Errorf("accelerated/blackcl: failed to create x device vector: %w", err)
	}

	xDevCopyComplete := xDev.Copy(x)

	wDev, err := o.AllocBuffer("w", len(w))
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

	if err := <-o.MatMulDevMem(xoutDev, xDev, wDev, n, d); err != nil {
		return err
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

func (o *OpenCL) MatMulDevMem(xout *blackcl.Vector, x *blackcl.Vector, w *blackcl.Vector, n int, d int) <-chan error {
	globalSize := d
	localSize := 1

	if globalSize%localGroupSize != 0 {
		globalSize /= localGroupSize
		localSize = localGroupSize
	}

	errChan := make(chan error, 1)

	go func() {
		if err := <-o.kernel.Global(globalSize).Local(localSize).Run(xout, x, w, uint32(n)); err != nil {
			errChan <- fmt.Errorf("accelerated/blackcl: failed to run matmul: %w", err)
		} else {
			errChan <- nil
		}
	}()

	return errChan
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
