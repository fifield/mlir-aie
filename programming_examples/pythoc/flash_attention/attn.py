from aie.iron.pythoc import aie_kernel

from pythoc import ptr, i32, bf16, void
from pythoc.aie import aie_vector, broadcast, getExpBf16, load_v, store_v, vector_add, vector_blend, vector_cast, vector_insert, vector_mul, vector_sub, vmax_ltbf16, zeros


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
	vec_size: i32 = 32
	iterations: i32 = 128
	blend_mask: i32 = 0
	p_out: ptr[bf16] = out_buf
	zero_words: aie_vector[i32, 16] = zeros(i32, 16)
	zero_words_opaque: aie_vector[i32, 16] = vector_blend(zero_words, zero_words, blend_mask)
	zero_vec: aie_vector[bf16, 32] = vector_cast(zero_words_opaque, bf16, 32)

	i: i32 = 0
	while i < iterations:
		store_v(p_out, zero_vec)
		p_out = p_out + vec_size
		i = i + 1


@aie_kernel
def zero_fill_g_bf16_pythoc(out_buf: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 128
	blend_mask: i32 = 0
	p_out: ptr[bf16] = out_buf
	zero_words: aie_vector[i32, 16] = zeros(i32, 16)
	zero_words_opaque: aie_vector[i32, 16] = vector_blend(zero_words, zero_words, blend_mask)
	zero_vec: aie_vector[bf16, 32] = vector_cast(zero_words_opaque, bf16, 32)

	i: i32 = 0
	while i < iterations:
		store_v(p_out, zero_vec)
		p_out = p_out + vec_size
		i = i + 1


@aie_kernel
def neg_inf_fill_up_bf16_pythoc(out_buf: ptr[bf16, True]) -> void:
	vec_size: i32 = 16
	iterations: i32 = 4
	p_out: ptr[bf16] = out_buf
	neg_inf_vec: aie_vector[bf16, 16] = broadcast(bf16, 16, -3.389e38)

	i: i32 = 0
	while i < iterations:
		store_v(p_out, neg_inf_vec)
		p_out = p_out + vec_size
		i = i + 1


@aie_kernel
def vector_copy_32elems_pythoc(offset: i32, inputs: ptr[bf16, True], outputs: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 2
	p_in: ptr[bf16] = inputs
	p_out: ptr[bf16] = outputs + offset

	i: i32 = 0
	while i < iterations:
		vec: aie_vector[bf16, 32] = load_v(p_in, 32)
		store_v(p_out, vec)
		p_in = p_in + vec_size
		p_out = p_out + vec_size
		i = i + 1


@aie_kernel
def copy_tile_pythoc(src: ptr[bf16, True], dst: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 128
	blend_mask: i32 = 0
	p_src: ptr[bf16] = src
	p_dst: ptr[bf16] = dst

	i: i32 = 0
	while i < iterations:
		vec: aie_vector[bf16, 32] = load_v(p_src, 32)
		vec_i32: aie_vector[i32, 16] = vector_cast(vec, i32, 16)
		vec_i32_opaque: aie_vector[i32, 16] = vector_blend(vec_i32, vec_i32, blend_mask)
		vec_out: aie_vector[bf16, 32] = vector_cast(vec_i32_opaque, bf16, 32)
		store_v(p_dst, vec_out)
		p_src = p_src + vec_size
		p_dst = p_dst + vec_size
		i = i + 1


@aie_kernel
def mul_r_gp_pythoc(r: ptr[bf16, True], gp: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	block_size: i32 = 64
	rows_per_block: i32 = 8
	col_blocks: i32 = 8
	row_blocks: i32 = 8
	block_stride: i32 = 512

	rb: i32 = 0
	while rb < row_blocks:
		half: i32 = 0
		while half < 2:
			row_start: i32 = rb * rows_per_block + half * 4
			p_r: ptr[bf16] = r + row_start
			r_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, p_r[0])
			r1: aie_vector[bf16, 8] = broadcast(bf16, 8, p_r[1])
			r2: aie_vector[bf16, 8] = broadcast(bf16, 8, p_r[2])
			r3: aie_vector[bf16, 8] = broadcast(bf16, 8, p_r[3])
			r_vec = vector_insert(r_vec, r1, 8)
			r_vec = vector_insert(r_vec, r2, 16)
			r_vec = vector_insert(r_vec, r3, 24)

			base: i32 = rb * block_size + half * vec_size
			cb: i32 = 0
			while cb < col_blocks:
				off: i32 = base + cb * block_stride
				p_gp: ptr[bf16] = gp + off
				v: aie_vector[bf16, 32] = load_v(p_gp, 32)
				v_out: aie_vector[bf16, 32] = vector_mul(v, r_vec)
				store_v(p_gp, v_out)
				cb = cb + 1

			half = half + 1
		rb = rb + 1


@aie_kernel
def exp_up_minus_u_pythoc(up: ptr[bf16, True], u: ptr[bf16, True], r: ptr[bf16, True]) -> void:
	vec_size: i32 = 16
	iterations: i32 = 4
	log2e_vec: aie_vector[bf16, 16] = broadcast(bf16, 16, 0.18033688011112042)
	p_up: ptr[bf16] = up
	p_u: ptr[bf16] = u
	p_r: ptr[bf16] = r

	i: i32 = 0
	while i < iterations:
		up_vec: aie_vector[bf16, 16] = load_v(p_up, 16)
		u_vec: aie_vector[bf16, 16] = load_v(p_u, 16)
		diff: aie_vector[bf16, 16] = vector_sub(up_vec, u_vec)
		scaled: aie_vector[bf16, 16] = vector_mul(diff, log2e_vec)
		exp_vec: aie_vector[bf16, 16] = getExpBf16(scaled)
		store_v(p_r, exp_vec)
		p_up = p_up + vec_size
		p_u = p_u + vec_size
		p_r = p_r + vec_size
		i = i + 1


@aie_kernel
def maximum_up_u_bf16_pythoc(up: ptr[bf16, True], u: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 2
	p_up: ptr[bf16] = up
	p_u: ptr[bf16] = u

	i: i32 = 0
	while i < iterations:
		up_vec: aie_vector[bf16, 32] = load_v(p_up, 32)
		u_vec: aie_vector[bf16, 32] = load_v(p_u, 32)
		out_vec, cmp_mask = vmax_ltbf16(up_vec, u_vec)
		store_v(p_u, out_vec)
		p_up = p_up + vec_size
		p_u = p_u + vec_size
		i = i + 1


@aie_kernel
def add_gp_g_pythoc(gp: ptr[bf16, True], g: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 128
	p_gp: ptr[bf16] = gp
	p_g: ptr[bf16] = g

	i: i32 = 0
	while i < iterations:
		gp_vec: aie_vector[bf16, 32] = load_v(p_gp, 32)
		g_vec: aie_vector[bf16, 32] = load_v(p_g, 32)
		out_vec: aie_vector[bf16, 32] = vector_add(gp_vec, g_vec)
		store_v(p_g, out_vec)
		p_gp = p_gp + vec_size
		p_g = p_g + vec_size
		i = i + 1




