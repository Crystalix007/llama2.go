package goopencl

import (
	_ "embed"
	"fmt"

	"github.com/haormj/llama2/accelerated"
	"github.com/passkeyra/go-opencl/opencl"
)

//go:embed matmul.cl
var matmulSrc string

type OpenCL struct {
	device       opencl.Device
	context      opencl.Context
	commandQueue opencl.CommandQueue
	matmulProg   opencl.Program
	matmulKernel opencl.Kernel
}

// Release implements accelerated.OpenCL.
func (o *OpenCL) Release() error {
	o.context.Release()

	return nil
}

var _ accelerated.Backend = &OpenCL{}

// getFirstDevice returns the first available OpenCL device of type deviceType.
func getFirstDevice(deviceType opencl.DeviceType) opencl.Device {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		panic(err)
	}

	for _, platform := range platforms {
		var devices []opencl.Device
		devices, err = platform.GetDevices(deviceType)
		if err != nil {
			panic(err)
		}

		for _, device := range devices {
			var available bool
			err = device.GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				return device
			}
		}
	}

	panic("No device found")
}

func (o *OpenCL) SetupContext() error {
	o.device = getFirstDevice(opencl.DeviceTypeGPU)

	var err error
	o.context, err = o.device.CreateContext()
	if err != nil {
		return fmt.Errorf("opencl/go-opencl: failed to create context: %w", err)
	}

	o.commandQueue, err = o.context.CreateCommandQueue(o.device)
	if err != nil {
		return err
	}

	o.matmulProg, err = o.context.CreateProgramWithSource(matmulSrc)
	if err != nil {
		return err
	}

	err = o.matmulProg.Build(o.device, nil)
	if err != nil {
		return err
	}

	o.matmulKernel, err = o.matmulProg.CreateKernel("matmul")
	if err != nil {
		return err
	}

	return nil
}

func (o *OpenCL) MatMul(xout, x, w []float32, n, d int) error {
	oclX, _ := o.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, uint64(len(x)))
	oclW, _ := o.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, uint64(len(w)))
	oclXout, _ := o.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, uint64(len(xout)))

	o.commandQueue.EnqueueReadBuffer(oclX, true, x)
	o.commandQueue.EnqueueReadBuffer(oclW, true, w)

	o.matmulKernel.SetArg(0, 8, oclXout)
	o.matmulKernel.SetArg(1, 8, oclX)
	o.matmulKernel.SetArg(2, 8, oclW)
	o.matmulKernel.SetArg(3, 4, uint32(n))

	o.commandQueue.EnqueueNDRangeKernel(o.matmulKernel, uint32(1), []uint64{uint64(d)})

	xoutTemp := make([]float32, len(xout))

	o.commandQueue.EnqueueWriteBuffer(oclXout, true, xoutTemp)

	for i := 0; i < d; i++ {
		xout[i] = xoutTemp[i]
	}

	return nil
}
