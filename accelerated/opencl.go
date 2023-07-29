package accelerated

type Backend interface {
	SetupContext() error
	MatMul(xout, x, w []float32, n, d int) error
	Release() error
}
