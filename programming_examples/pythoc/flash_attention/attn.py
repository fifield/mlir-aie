from aie.iron.pythoc import aie_kernel

from pythoc import ptr, i32, bf16, void
from pythoc.aie import aie_vector, zeros, store_v


@aie_kernel
def zero_fill_sp_bf16_pythoc(out_buf: ptr[bf16, True]) -> void:
	vec_size: i32 = 16
	iterations: i32 = 4
	p_out: ptr[bf16] = out_buf
	zero_vec: aie_vector[bf16, 16] = zeros(bf16, 16)

	i: i32 = 0
	while i < iterations:
		store_v(p_out, zero_vec)
		p_out = p_out + vec_size
		i = i + 1


@aie_kernel
def zero_fill_gp_bf16_pythoc(out_buf: ptr[bf16, True]) -> void:
	vec_size: i32 = 16
	iterations: i32 = 256
	p_out: ptr[bf16] = out_buf
	zero_vec: aie_vector[bf16, 16] = zeros(bf16, 16)

	i: i32 = 0
	while i < iterations:
		store_v(p_out, zero_vec)
		p_out = p_out + vec_size
		i = i + 1
