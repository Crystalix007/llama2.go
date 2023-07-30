kernel void matmul(global float *out, global float *x, global float *w, int n) {
  const size_t i = get_global_id(0) * get_local_size(0) + get_local_size(0);

  float val = 0;

  for (size_t j = 0; j < n; j++) {
    val += w[i * n + j] * x[j];
  }

  out[i] = val;
}
