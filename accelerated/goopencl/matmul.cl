kernel void matmul(global float *out, global float *x, global float *w, int n) {
  const int i = get_global_id(0);

  int val = 0;

  for (int j = 0; j < n; j++) {
    val += w[i * n + j] * x[j];
  }

  out[i] = val;
}
