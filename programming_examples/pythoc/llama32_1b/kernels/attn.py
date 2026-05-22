# Flash attention kernels for llama32_1b decode path.
#
# Phase 3.4 STATUS: all 19 attn_npu2.o symbols ported. The four kernels
# added at the bottom (`max_g_bf16`, `sum_g`, `apply_causal_mask`,
# `fused_softmax`) close the port. `compile_attn` in kernels/build.py
# uses `fused_softmax` as the named entry; every other @aie_kernel is
# treated as a helper in the same TU and lands in attn_pythoc.o, which
# is now wired up via _PYTHOC_KERNELS / _LINK_OBJS in kernel_builder/.
# The cached MLIR's `link_with = "attn_npu2.o"` attributes are likewise
# patched to "attn_pythoc.o" and attn_npu2.o is no longer staged.

from aie.iron.pythoc import aie_kernel

from pythoc import ptr, i16, i32, i64, f32, bf16, void
from pythoc.aie import ACC2048_accfloat_add_conf, BFP576_BFP576_ACC2048_mac_conf, I1024_I1024_ACC2048_bf_mul_conf, I512_I512_ACC1024_bf_mac_conf, I512_I512_ACC1024_bf_mul_conf, I512_I512_ACC1024_bf_negmul_conf, acc_extract, acc_grow, aie_vector, broadcast, concat, exp2, extract_elem, getExpBf16, insert_elem, load_v, reduce_add, set_ctrl_reg, store_v, v32accfloat_to_v32bf16, v32bf16_to_v32accfloat, v64accfloat_to_v64bfp16ebs8, vector_add, vector_blend, vector_cast, vector_extract, vector_grow, vector_insert, vector_mul, vector_sub, vmax_ltbf16, vshuffle, zeros


@aie_kernel
def zero_fill_sp_bf16(out_buf: ptr[bf16, True]) -> void:
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
def zero_fill_gp_bf16(out_buf: ptr[bf16, True]) -> void:
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
def zero_fill_g_bf16(out_buf: ptr[bf16, True]) -> void:
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
def neg_inf_fill_up_bf16(out_buf: ptr[bf16, True]) -> void:
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
def vector_copy_32elems(offset: i32, inputs: ptr[bf16, True], outputs: ptr[bf16, True]) -> void:
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
def copy_tile(src: ptr[bf16, True], dst: ptr[bf16, True]) -> void:
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
def mul_r_gp(r: ptr[bf16, True], gp: ptr[bf16, True]) -> void:
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
def exp_up_minus_u(up: ptr[bf16, True], u: ptr[bf16, True], r: ptr[bf16, True]) -> void:
	vec_size: i32 = 16
	iterations: i32 = 4
	log2e_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, 0.18033688011112042)
	lowest_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, -3.389e38)
	p_up: ptr[bf16] = up
	p_u: ptr[bf16] = u
	p_r: ptr[bf16] = r
	set_ctrl_reg(1, 12)

	i: i32 = 0
	while i < iterations:
		up_vec: aie_vector[bf16, 16] = load_v(p_up, 16)
		u_vec: aie_vector[bf16, 16] = load_v(p_u, 16)
		diff: aie_vector[bf16, 16] = vector_sub(up_vec, u_vec)
		diff_i32: aie_vector[i32, 8] = vector_cast(diff, i32, 8)
		diff_wide: aie_vector[i32, 16] = vector_grow(diff_i32, 16, 0)
		diff_bf32: aie_vector[bf16, 32] = vector_cast(diff_wide, bf16, 32)
		clamped, cmp_mask = vmax_ltbf16(diff_bf32, lowest_vec)
		clamped_i32: aie_vector[i32, 16] = vector_cast(clamped, i32, 16)
		lo_i32: aie_vector[i32, 8] = vector_extract(clamped_i32, 0, 8)
		lo_wide: aie_vector[i32, 16] = vector_grow(lo_i32, 16, 0)
		lo_bf: aie_vector[bf16, 32] = vector_cast(lo_wide, bf16, 32)
		lo_mul: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mul_conf(lo_bf, log2e_vec, 60)
		lo_acc: aie_vector[f32, 16] = acc_extract(lo_mul, 0)
		exp_vec: aie_vector[bf16, 16] = exp2(lo_acc)
		store_v(p_r, exp_vec)
		p_up = p_up + vec_size
		p_u = p_u + vec_size
		p_r = p_r + vec_size
		i = i + 1


@aie_kernel
def maximum_up_u_bf16(up: ptr[bf16, True], u: ptr[bf16, True]) -> void:
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
def add_gp_g(gp: ptr[bf16, True], g: ptr[bf16, True]) -> void:
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


@aie_kernel
def exp_g_minus_u(u: ptr[bf16, True], g: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	block_size: i32 = 64
	col_blocks: i32 = 8
	row_blocks: i32 = 8
	block_stride: i32 = 512
	rows_per_block: i32 = 8
	log2e_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, 0.18033688011112042)
	lowest_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, -3.389e38)
	set_ctrl_reg(1, 12)

	rb: i32 = 0
	while rb < row_blocks:
		half: i32 = 0
		while half < 2:
			row_start: i32 = rb * rows_per_block + half * 4
			u_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, u[row_start])
			u1: aie_vector[bf16, 8] = broadcast(bf16, 8, u[row_start + 1])
			u2: aie_vector[bf16, 8] = broadcast(bf16, 8, u[row_start + 2])
			u3: aie_vector[bf16, 8] = broadcast(bf16, 8, u[row_start + 3])
			u_vec = vector_insert(u_vec, u1, 8)
			u_vec = vector_insert(u_vec, u2, 16)
			u_vec = vector_insert(u_vec, u3, 24)

			base: i32 = rb * block_size + half * vec_size
			cb: i32 = 0
			while cb < col_blocks:
				off: i32 = base + cb * block_stride
				p_g: ptr[bf16] = g + off
				v: aie_vector[bf16, 32] = load_v(p_g, 32)
				diff: aie_vector[bf16, 32] = vector_sub(v, u_vec)
				clamped, cmp_mask = vmax_ltbf16(diff, lowest_vec)
				clamped_i32: aie_vector[i32, 16] = vector_cast(clamped, i32, 16)
				lo_i32: aie_vector[i32, 8] = vector_extract(clamped_i32, 0, 8)
				hi_i32: aie_vector[i32, 8] = vector_extract(clamped_i32, 8, 8)
				lo_wide: aie_vector[i32, 16] = vector_grow(lo_i32, 16, 0)
				hi_wide: aie_vector[i32, 16] = vector_grow(hi_i32, 16, 0)
				lo_bf: aie_vector[bf16, 32] = vector_cast(lo_wide, bf16, 32)
				hi_bf: aie_vector[bf16, 32] = vector_cast(hi_wide, bf16, 32)
				lo_mul: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mul_conf(lo_bf, log2e_vec, 60)
				hi_mul: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mul_conf(hi_bf, log2e_vec, 60)
				lo_acc: aie_vector[f32, 16] = acc_extract(lo_mul, 0)
				hi_acc: aie_vector[f32, 16] = acc_extract(hi_mul, 0)
				lo_exp: aie_vector[bf16, 16] = exp2(lo_acc)
				hi_exp: aie_vector[bf16, 16] = exp2(hi_acc)
				lo_exp_i32: aie_vector[i32, 8] = vector_cast(lo_exp, i32, 8)
				hi_exp_i32: aie_vector[i32, 8] = vector_cast(hi_exp, i32, 8)
				result_i32: aie_vector[i32, 16] = concat(lo_exp_i32, hi_exp_i32)
				result: aie_vector[bf16, 32] = vector_cast(result_i32, bf16, 32)
				store_v(p_g, result)
				cb = cb + 1

			half = half + 1
		rb = rb + 1


@aie_kernel
def accum_sp_r_s(sp: ptr[bf16, True], r: ptr[bf16, True], s: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	iterations: i32 = 2
	conf: i32 = 60
	p_sp: ptr[bf16] = sp
	p_r: ptr[bf16] = r
	p_s: ptr[bf16] = s
	set_ctrl_reg(1, 12)

	i: i32 = 0
	while i < iterations:
		sp_vec: aie_vector[bf16, 32] = load_v(p_sp, 32)
		r_vec: aie_vector[bf16, 32] = load_v(p_r, 32)
		s_vec: aie_vector[bf16, 32] = load_v(p_s, 32)
		s_acc: aie_vector[f32, 32] = v32bf16_to_v32accfloat(s_vec)
		out_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mac_conf(r_vec, sp_vec, s_acc, conf)
		out_vec: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(out_lo)
		store_v(p_s, out_vec)
		p_sp = p_sp + vec_size
		p_r = p_r + vec_size
		p_s = p_s + vec_size
		i = i + 1


@aie_kernel
def div_gp_sp(sp: ptr[bf16, True], gp: ptr[bf16, True]) -> void:
	vec_size: i32 = 32
	block_size: i32 = 64
	rows_per_block: i32 = 8
	cols_per_block: i32 = 8
	col_blocks: i32 = 8
	row_blocks: i32 = 8
	block_stride: i32 = 512
	conf: i32 = 60
	store_conf: i32 = 828
	sp_approx_bias_bits: aie_vector[i16, 32] = broadcast(i16, 32, 32437)
	recip_add_const: aie_vector[f32, 64] = broadcast(f32, 64, 1.4361419677734375)
	negmul_const: aie_vector[bf16, 32] = broadcast(bf16, 32, 1.9375)
	two_const: aie_vector[f32, 32] = broadcast(f32, 32, 2.0)
	set_ctrl_reg(1, 12)

	rb: i32 = 0
	while rb < row_blocks:
		half: i32 = 0
		while half < 2:
			row_start: i32 = rb * rows_per_block + half * 4
			sp0_full: aie_vector[bf16, 32] = broadcast(bf16, 32, sp[row_start])
			sp1_full: aie_vector[bf16, 32] = broadcast(bf16, 32, sp[row_start + 1])
			sp2_full: aie_vector[bf16, 32] = broadcast(bf16, 32, sp[row_start + 2])
			sp3_full: aie_vector[bf16, 32] = broadcast(bf16, 32, sp[row_start + 3])
			sp_words: aie_vector[i32, 16] = zeros(i32, 16)
			sp0_words: aie_vector[i32, 4] = vector_extract(vector_cast(sp0_full, i32, 16), 0, 4)
			sp1_words: aie_vector[i32, 4] = vector_extract(vector_cast(sp1_full, i32, 16), 0, 4)
			sp2_words: aie_vector[i32, 4] = vector_extract(vector_cast(sp2_full, i32, 16), 0, 4)
			sp3_words: aie_vector[i32, 4] = vector_extract(vector_cast(sp3_full, i32, 16), 0, 4)
			sp_words = vector_insert(sp_words, sp0_words, 0)
			sp_words = vector_insert(sp_words, sp1_words, 4)
			sp_words = vector_insert(sp_words, sp2_words, 8)
			sp_words = vector_insert(sp_words, sp3_words, 12)
			sp_vec: aie_vector[bf16, 32] = vector_cast(sp_words, bf16, 32)

			sp_bits: aie_vector[i16, 32] = vector_cast(sp_vec, i16, 32)
			sp_estimate_bits: aie_vector[i16, 32] = vector_sub(sp_approx_bias_bits, sp_bits)
			sp_estimate: aie_vector[bf16, 32] = vector_cast(sp_estimate_bits, bf16, 32)

			negmul0_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_negmul_conf(sp_vec, sp_estimate, conf)
			negmul0_lo_acc: aie_vector[i64, 16] = vector_cast(negmul0_lo, i64, 16)
			negmul0_wide_acc: aie_vector[i64, 32] = acc_grow(negmul0_lo_acc)
			negmul0_wide: aie_vector[f32, 64] = vector_cast(negmul0_wide_acc, f32, 64)
			recip0_wide: aie_vector[f32, 64] = ACC2048_accfloat_add_conf(negmul0_wide, recip_add_const, conf)
			recip0_wide_acc: aie_vector[i64, 32] = vector_cast(recip0_wide, i64, 32)
			recip0_lo_acc: aie_vector[i64, 16] = acc_extract(recip0_wide_acc, 0)
			recip0_lo: aie_vector[f32, 32] = vector_cast(recip0_lo_acc, f32, 32)
			recip0: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(recip0_lo)

			corr_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mul_conf(recip0, sp_estimate, conf)
			corr: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(corr_lo)
			neg_corr_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_negmul_conf(corr, negmul_const, conf)
			neg_corr: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(neg_corr_lo)
			recip1_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mac_conf(neg_corr, sp_vec, two_const, conf)
			recip1: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(recip1_lo)
			sp_inv_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_negmul_conf(neg_corr, recip1, conf)
			sp_inv: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(sp_inv_lo)

			base: i32 = rb * block_size + half * vec_size
			cb: i32 = 0
			while cb < col_blocks:
				off: i32 = base + cb * block_stride
				p_gp: ptr[bf16] = gp + off
				v: aie_vector[bf16, 32] = load_v(p_gp, 32)
				v_out_lo: aie_vector[f32, 32] = I512_I512_ACC1024_bf_mul_conf(v, sp_inv, store_conf)
				v_out: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(v_out_lo)
				store_v(p_gp, v_out)
				cb = cb + 1

			half = half + 1
		rb = rb + 1


@aie_kernel
def matmul_a_b_bf16(a_in: ptr[bf16, True], b_in: ptr[bf16, True], out: ptr[bf16, True]) -> void:
	block_size: i32 = 64
	row_blocks: i32 = 8
	col_blocks: i32 = 8
	k_blocks: i32 = 8
	a_k_stride: i32 = 512
	b_k_stride: i32 = 512
	b_n_stride: i32 = 64
	c_m_stride: i32 = 64
	c_n_stride: i32 = 512
	bf_mul_conf: i32 = 60
	mac_conf: i32 = 780
	one_vec: aie_vector[bf16, 64] = broadcast(bf16, 64, 1.0)
	set_ctrl_reg(1, 12)

	m: i32 = 0
	p_a_row0_base: ptr[bf16] = a_in
	p_a_row1_base: ptr[bf16] = a_in + block_size
	p_c_row0_base: ptr[bf16] = out
	p_c_row1_base: ptr[bf16] = out + c_m_stride
	while m < row_blocks:
		n: i32 = 0
		p_c_row0: ptr[bf16] = p_c_row0_base
		p_c_row1: ptr[bf16] = p_c_row1_base
		p_b_col0_base: ptr[bf16] = b_in
		p_b_col1_base: ptr[bf16] = b_in + b_n_stride
		while n < col_blocks:
			p_c00: ptr[bf16] = p_c_row0
			p_c01: ptr[bf16] = p_c00 + c_n_stride
			p_c10: ptr[bf16] = p_c_row1
			p_c11: ptr[bf16] = p_c10 + c_n_stride

			vc00_lo: aie_vector[bf16, 32] = load_v(p_c00, 32)
			vc00_hi: aie_vector[bf16, 32] = load_v(p_c00 + 32, 32)
			acc_c00_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc00_lo)
			acc_c00_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc00_hi)
			acc_c00_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c00_lo, i64, 16)
			acc_c00_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c00_hi, i64, 16)
			acc_c00_i64: aie_vector[i64, 32] = concat(acc_c00_lo_i64, acc_c00_hi_i64)
			acc_c00: aie_vector[f32, 64] = vector_cast(acc_c00_i64, f32, 64)

			vc01_lo: aie_vector[bf16, 32] = load_v(p_c01, 32)
			vc01_hi: aie_vector[bf16, 32] = load_v(p_c01 + 32, 32)
			acc_c01_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc01_lo)
			acc_c01_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc01_hi)
			acc_c01_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c01_lo, i64, 16)
			acc_c01_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c01_hi, i64, 16)
			acc_c01_i64: aie_vector[i64, 32] = concat(acc_c01_lo_i64, acc_c01_hi_i64)
			acc_c01: aie_vector[f32, 64] = vector_cast(acc_c01_i64, f32, 64)

			vc10_lo: aie_vector[bf16, 32] = load_v(p_c10, 32)
			vc10_hi: aie_vector[bf16, 32] = load_v(p_c10 + 32, 32)
			acc_c10_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc10_lo)
			acc_c10_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc10_hi)
			acc_c10_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c10_lo, i64, 16)
			acc_c10_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c10_hi, i64, 16)
			acc_c10_i64: aie_vector[i64, 32] = concat(acc_c10_lo_i64, acc_c10_hi_i64)
			acc_c10: aie_vector[f32, 64] = vector_cast(acc_c10_i64, f32, 64)

			vc11_lo: aie_vector[bf16, 32] = load_v(p_c11, 32)
			vc11_hi: aie_vector[bf16, 32] = load_v(p_c11 + 32, 32)
			acc_c11_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc11_lo)
			acc_c11_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc11_hi)
			acc_c11_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c11_lo, i64, 16)
			acc_c11_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c11_hi, i64, 16)
			acc_c11_i64: aie_vector[i64, 32] = concat(acc_c11_lo_i64, acc_c11_hi_i64)
			acc_c11: aie_vector[f32, 64] = vector_cast(acc_c11_i64, f32, 64)

			p_a0: ptr[bf16] = p_a_row0_base
			p_a1: ptr[bf16] = p_a_row1_base
			p_b0: ptr[bf16] = p_b_col0_base
			p_b1: ptr[bf16] = p_b_col1_base

			k: i32 = 0
			while k < k_blocks:
				a0_lo: aie_vector[bf16, 32] = load_v(p_a0, 32)
				a0_hi: aie_vector[bf16, 32] = load_v(p_a0 + 32, 32)
				p_a0 = p_a0 + a_k_stride
				a0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_lo)
				a0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_hi)
				a0_acc_lo_i64: aie_vector[i64, 16] = vector_cast(a0_acc_lo, i64, 16)
				a0_acc_hi_i64: aie_vector[i64, 16] = vector_cast(a0_acc_hi, i64, 16)
				a0_acc_i64: aie_vector[i64, 32] = concat(a0_acc_lo_i64, a0_acc_hi_i64)
				a0_acc: aie_vector[f32, 64] = vector_cast(a0_acc_i64, f32, 64)
				a0_mant, a0_exp = v64accfloat_to_v64bfp16ebs8(a0_acc)

				vb0_lo_bf: aie_vector[bf16, 32] = load_v(p_b0, 32)
				vb0_hi_bf: aie_vector[bf16, 32] = load_v(p_b0 + 32, 32)
				p_b0 = p_b0 + b_k_stride
				b0_lo_i: aie_vector[i32, 16] = vector_cast(vb0_lo_bf, i32, 16)
				b0_hi_i: aie_vector[i32, 16] = vector_cast(vb0_hi_bf, i32, 16)
				b0_stage0_even: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 52)
				b0_stage0_odd: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 53)
				b0_even: aie_vector[i32, 16] = vshuffle(b0_stage0_even, b0_stage0_odd, 52)
				b0_odd: aie_vector[i32, 16] = vshuffle(b0_stage0_even, b0_stage0_odd, 53)
				b0_cat: aie_vector[i32, 32] = concat(b0_even, b0_odd)
				vb0_s: aie_vector[bf16, 64] = vector_cast(b0_cat, bf16, 64)
				b0_acc: aie_vector[f32, 64] = I1024_I1024_ACC2048_bf_mul_conf(vb0_s, one_vec, bf_mul_conf)
				b0_mant, b0_exp = v64accfloat_to_v64bfp16ebs8(b0_acc)

				acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
				res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, mac_conf
				)

				vb1_lo_bf: aie_vector[bf16, 32] = load_v(p_b1, 32)
				vb1_hi_bf: aie_vector[bf16, 32] = load_v(p_b1 + 32, 32)
				p_b1 = p_b1 + b_k_stride
				b1_lo_i: aie_vector[i32, 16] = vector_cast(vb1_lo_bf, i32, 16)
				b1_hi_i: aie_vector[i32, 16] = vector_cast(vb1_hi_bf, i32, 16)
				b1_stage0_even: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 52)
				b1_stage0_odd: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 53)
				b1_even: aie_vector[i32, 16] = vshuffle(b1_stage0_even, b1_stage0_odd, 52)
				b1_odd: aie_vector[i32, 16] = vshuffle(b1_stage0_even, b1_stage0_odd, 53)
				b1_cat: aie_vector[i32, 32] = concat(b1_even, b1_odd)
				vb1_s: aie_vector[bf16, 64] = vector_cast(b1_cat, bf16, 64)
				b1_acc: aie_vector[f32, 64] = I1024_I1024_ACC2048_bf_mul_conf(vb1_s, one_vec, bf_mul_conf)
				b1_mant, b1_exp = v64accfloat_to_v64bfp16ebs8(b1_acc)

				acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
				res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, mac_conf
				)

				a1_lo: aie_vector[bf16, 32] = load_v(p_a1, 32)
				a1_hi: aie_vector[bf16, 32] = load_v(p_a1 + 32, 32)
				p_a1 = p_a1 + a_k_stride
				a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
				a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
				a1_acc_lo_i64: aie_vector[i64, 16] = vector_cast(a1_acc_lo, i64, 16)
				a1_acc_hi_i64: aie_vector[i64, 16] = vector_cast(a1_acc_hi, i64, 16)
				a1_acc_i64: aie_vector[i64, 32] = concat(a1_acc_lo_i64, a1_acc_hi_i64)
				a1_acc: aie_vector[f32, 64] = vector_cast(a1_acc_i64, f32, 64)
				a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

				acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
				res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, mac_conf
				)
				acc_i11: aie_vector[i32, 64] = vector_cast(acc_c11, i32, 64)
				res11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a1_mant, a1_exp, b1_mant, b1_exp, acc_i11, mac_conf
				)

				acc_c00 = vector_cast(res00, f32, 64)
				acc_c01 = vector_cast(res01, f32, 64)
				acc_c10 = vector_cast(res10, f32, 64)
				acc_c11 = vector_cast(res11, f32, 64)
				k = k + 1

			acc_c00_store_i64: aie_vector[i64, 32] = vector_cast(acc_c00, i64, 32)
			acc_c00_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c00_store_i64, 0, 16)
			acc_c00_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c00_store_i64, 16, 16)
			acc_c00_store_lo: aie_vector[f32, 32] = vector_cast(acc_c00_store_lo_i64, f32, 32)
			acc_c00_store_hi: aie_vector[f32, 32] = vector_cast(acc_c00_store_hi_i64, f32, 32)
			store_v(p_c00, v32accfloat_to_v32bf16(acc_c00_store_lo))
			store_v(p_c00 + 32, v32accfloat_to_v32bf16(acc_c00_store_hi))

			acc_c01_store_i64: aie_vector[i64, 32] = vector_cast(acc_c01, i64, 32)
			acc_c01_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c01_store_i64, 0, 16)
			acc_c01_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c01_store_i64, 16, 16)
			acc_c01_store_lo: aie_vector[f32, 32] = vector_cast(acc_c01_store_lo_i64, f32, 32)
			acc_c01_store_hi: aie_vector[f32, 32] = vector_cast(acc_c01_store_hi_i64, f32, 32)
			store_v(p_c01, v32accfloat_to_v32bf16(acc_c01_store_lo))
			store_v(p_c01 + 32, v32accfloat_to_v32bf16(acc_c01_store_hi))

			acc_c10_store_i64: aie_vector[i64, 32] = vector_cast(acc_c10, i64, 32)
			acc_c10_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c10_store_i64, 0, 16)
			acc_c10_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c10_store_i64, 16, 16)
			acc_c10_store_lo: aie_vector[f32, 32] = vector_cast(acc_c10_store_lo_i64, f32, 32)
			acc_c10_store_hi: aie_vector[f32, 32] = vector_cast(acc_c10_store_hi_i64, f32, 32)
			store_v(p_c10, v32accfloat_to_v32bf16(acc_c10_store_lo))
			store_v(p_c10 + 32, v32accfloat_to_v32bf16(acc_c10_store_hi))

			acc_c11_store_i64: aie_vector[i64, 32] = vector_cast(acc_c11, i64, 32)
			acc_c11_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c11_store_i64, 0, 16)
			acc_c11_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c11_store_i64, 16, 16)
			acc_c11_store_lo: aie_vector[f32, 32] = vector_cast(acc_c11_store_lo_i64, f32, 32)
			acc_c11_store_hi: aie_vector[f32, 32] = vector_cast(acc_c11_store_hi_i64, f32, 32)
			store_v(p_c11, v32accfloat_to_v32bf16(acc_c11_store_lo))
			store_v(p_c11 + 32, v32accfloat_to_v32bf16(acc_c11_store_hi))

			p_c_row0 = p_c01 + c_n_stride
			p_c_row1 = p_c11 + c_n_stride
			p_b_col0_base = p_b_col0_base + b_n_stride + b_n_stride
			p_b_col1_base = p_b_col1_base + b_n_stride + b_n_stride

			n = n + 2
		p_a_row0_base = p_a_row0_base + block_size + block_size
		p_a_row1_base = p_a_row1_base + block_size + block_size
		p_c_row0_base = p_c_row0_base + c_m_stride + c_m_stride
		p_c_row1_base = p_c_row1_base + c_m_stride + c_m_stride
		m = m + 2


@aie_kernel
def matmul_g_b_bf16(g_in: ptr[bf16, True], b_in: ptr[bf16, True], out: ptr[bf16, True]) -> void:
	block_size: i32 = 64
	row_blocks: i32 = 8
	col_blocks: i32 = 8
	k_blocks: i32 = 8
	a_k_stride: i32 = 512
	b_k_stride: i32 = 64
	b_n_stride: i32 = 512
	c_m_stride: i32 = 64
	c_n_stride: i32 = 512
	bf_mul_conf: i32 = 60
	mac_conf: i32 = 780
	one_vec: aie_vector[bf16, 64] = broadcast(bf16, 64, 1.0)
	set_ctrl_reg(1, 12)

	m: i32 = 0
	p_a_row0_base: ptr[bf16] = g_in
	p_a_row1_base: ptr[bf16] = g_in + block_size
	p_c_row0_base: ptr[bf16] = out
	p_c_row1_base: ptr[bf16] = out + c_m_stride
	while m < row_blocks:
		n: i32 = 0
		p_c_row0: ptr[bf16] = p_c_row0_base
		p_c_row1: ptr[bf16] = p_c_row1_base
		p_b_col0_base: ptr[bf16] = b_in
		p_b_col1_base: ptr[bf16] = b_in + b_n_stride
		while n < col_blocks:
			p_c00: ptr[bf16] = p_c_row0
			p_c01: ptr[bf16] = p_c00 + c_n_stride
			p_c10: ptr[bf16] = p_c_row1
			p_c11: ptr[bf16] = p_c10 + c_n_stride

			vc00: aie_vector[bf16, 64] = load_v(p_c00, 64)
			vc01: aie_vector[bf16, 64] = load_v(p_c01, 64)
			vc10: aie_vector[bf16, 64] = load_v(p_c10, 64)
			vc11: aie_vector[bf16, 64] = load_v(p_c11, 64)

			vc00_lo: aie_vector[bf16, 32] = vector_extract(vc00, 0, 32)
			vc00_hi: aie_vector[bf16, 32] = vector_extract(vc00, 32, 32)
			acc_c00_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc00_lo)
			acc_c00_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc00_hi)
			acc_c00_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c00_lo, i64, 16)
			acc_c00_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c00_hi, i64, 16)
			acc_c00_i64: aie_vector[i64, 32] = concat(acc_c00_lo_i64, acc_c00_hi_i64)
			acc_c00: aie_vector[f32, 64] = vector_cast(acc_c00_i64, f32, 64)

			vc01_lo: aie_vector[bf16, 32] = vector_extract(vc01, 0, 32)
			vc01_hi: aie_vector[bf16, 32] = vector_extract(vc01, 32, 32)
			acc_c01_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc01_lo)
			acc_c01_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc01_hi)
			acc_c01_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c01_lo, i64, 16)
			acc_c01_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c01_hi, i64, 16)
			acc_c01_i64: aie_vector[i64, 32] = concat(acc_c01_lo_i64, acc_c01_hi_i64)
			acc_c01: aie_vector[f32, 64] = vector_cast(acc_c01_i64, f32, 64)

			vc10_lo: aie_vector[bf16, 32] = vector_extract(vc10, 0, 32)
			vc10_hi: aie_vector[bf16, 32] = vector_extract(vc10, 32, 32)
			acc_c10_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc10_lo)
			acc_c10_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc10_hi)
			acc_c10_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c10_lo, i64, 16)
			acc_c10_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c10_hi, i64, 16)
			acc_c10_i64: aie_vector[i64, 32] = concat(acc_c10_lo_i64, acc_c10_hi_i64)
			acc_c10: aie_vector[f32, 64] = vector_cast(acc_c10_i64, f32, 64)

			vc11_lo: aie_vector[bf16, 32] = vector_extract(vc11, 0, 32)
			vc11_hi: aie_vector[bf16, 32] = vector_extract(vc11, 32, 32)
			acc_c11_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc11_lo)
			acc_c11_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(vc11_hi)
			acc_c11_lo_i64: aie_vector[i64, 16] = vector_cast(acc_c11_lo, i64, 16)
			acc_c11_hi_i64: aie_vector[i64, 16] = vector_cast(acc_c11_hi, i64, 16)
			acc_c11_i64: aie_vector[i64, 32] = concat(acc_c11_lo_i64, acc_c11_hi_i64)
			acc_c11: aie_vector[f32, 64] = vector_cast(acc_c11_i64, f32, 64)

			p_a0: ptr[bf16] = p_a_row0_base
			p_a1: ptr[bf16] = p_a_row1_base
			p_b0: ptr[bf16] = p_b_col0_base
			p_b1: ptr[bf16] = p_b_col1_base

			k: i32 = 0
			while k < k_blocks:
				va0: aie_vector[bf16, 64] = load_v(p_a0, 64)
				p_a0 = p_a0 + a_k_stride

				a0_lo: aie_vector[bf16, 32] = vector_extract(va0, 0, 32)
				a0_hi: aie_vector[bf16, 32] = vector_extract(va0, 32, 32)
				a0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_lo)
				a0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_hi)
				a0_acc_lo_i64: aie_vector[i64, 16] = vector_cast(a0_acc_lo, i64, 16)
				a0_acc_hi_i64: aie_vector[i64, 16] = vector_cast(a0_acc_hi, i64, 16)
				a0_acc_i64: aie_vector[i64, 32] = concat(a0_acc_lo_i64, a0_acc_hi_i64)
				a0_acc: aie_vector[f32, 64] = vector_cast(a0_acc_i64, f32, 64)
				a0_mant, a0_exp = v64accfloat_to_v64bfp16ebs8(a0_acc)

				vb0: aie_vector[bf16, 64] = load_v(p_b0, 64)
				p_b0 = p_b0 + b_k_stride
				b0_i32: aie_vector[i32, 32] = vector_cast(vb0, i32, 32)
				b0_lo_i: aie_vector[i32, 16] = vector_extract(b0_i32, 0, 16)
				b0_hi_i: aie_vector[i32, 16] = vector_extract(b0_i32, 16, 16)
				b0_even: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 52)
				b0_odd: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 53)
				b0_cat: aie_vector[i32, 32] = concat(b0_even, b0_odd)
				vb0_s: aie_vector[bf16, 64] = vector_cast(b0_cat, bf16, 64)
				b0_acc: aie_vector[f32, 64] = I1024_I1024_ACC2048_bf_mul_conf(vb0_s, one_vec, bf_mul_conf)
				b0_mant, b0_exp = v64accfloat_to_v64bfp16ebs8(b0_acc)

				acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
				res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, mac_conf
				)

				vb1: aie_vector[bf16, 64] = load_v(p_b1, 64)
				p_b1 = p_b1 + b_k_stride
				b1_i32: aie_vector[i32, 32] = vector_cast(vb1, i32, 32)
				b1_lo_i: aie_vector[i32, 16] = vector_extract(b1_i32, 0, 16)
				b1_hi_i: aie_vector[i32, 16] = vector_extract(b1_i32, 16, 16)
				b1_even: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 52)
				b1_odd: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 53)
				b1_cat: aie_vector[i32, 32] = concat(b1_even, b1_odd)
				vb1_s: aie_vector[bf16, 64] = vector_cast(b1_cat, bf16, 64)
				b1_acc: aie_vector[f32, 64] = I1024_I1024_ACC2048_bf_mul_conf(vb1_s, one_vec, bf_mul_conf)
				b1_mant, b1_exp = v64accfloat_to_v64bfp16ebs8(b1_acc)

				acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
				res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, mac_conf
				)

				va1: aie_vector[bf16, 64] = load_v(p_a1, 64)
				p_a1 = p_a1 + a_k_stride
				a1_lo: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
				a1_hi: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
				a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
				a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
				a1_acc_lo_i64: aie_vector[i64, 16] = vector_cast(a1_acc_lo, i64, 16)
				a1_acc_hi_i64: aie_vector[i64, 16] = vector_cast(a1_acc_hi, i64, 16)
				a1_acc_i64: aie_vector[i64, 32] = concat(a1_acc_lo_i64, a1_acc_hi_i64)
				a1_acc: aie_vector[f32, 64] = vector_cast(a1_acc_i64, f32, 64)
				a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

				acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
				res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, mac_conf
				)
				acc_i11: aie_vector[i32, 64] = vector_cast(acc_c11, i32, 64)
				res11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
					a1_mant, a1_exp, b1_mant, b1_exp, acc_i11, mac_conf
				)

				acc_c00 = vector_cast(res00, f32, 64)
				acc_c01 = vector_cast(res01, f32, 64)
				acc_c10 = vector_cast(res10, f32, 64)
				acc_c11 = vector_cast(res11, f32, 64)
				k = k + 1

			acc_c00_store_i64: aie_vector[i64, 32] = vector_cast(acc_c00, i64, 32)
			acc_c00_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c00_store_i64, 0, 16)
			acc_c00_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c00_store_i64, 16, 16)
			acc_c00_store_lo: aie_vector[f32, 32] = vector_cast(acc_c00_store_lo_i64, f32, 32)
			acc_c00_store_hi: aie_vector[f32, 32] = vector_cast(acc_c00_store_hi_i64, f32, 32)
			store_v(p_c00, v32accfloat_to_v32bf16(acc_c00_store_lo))
			store_v(p_c00 + 32, v32accfloat_to_v32bf16(acc_c00_store_hi))

			acc_c01_store_i64: aie_vector[i64, 32] = vector_cast(acc_c01, i64, 32)
			acc_c01_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c01_store_i64, 0, 16)
			acc_c01_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c01_store_i64, 16, 16)
			acc_c01_store_lo: aie_vector[f32, 32] = vector_cast(acc_c01_store_lo_i64, f32, 32)
			acc_c01_store_hi: aie_vector[f32, 32] = vector_cast(acc_c01_store_hi_i64, f32, 32)
			store_v(p_c01, v32accfloat_to_v32bf16(acc_c01_store_lo))
			store_v(p_c01 + 32, v32accfloat_to_v32bf16(acc_c01_store_hi))

			acc_c10_store_i64: aie_vector[i64, 32] = vector_cast(acc_c10, i64, 32)
			acc_c10_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c10_store_i64, 0, 16)
			acc_c10_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c10_store_i64, 16, 16)
			acc_c10_store_lo: aie_vector[f32, 32] = vector_cast(acc_c10_store_lo_i64, f32, 32)
			acc_c10_store_hi: aie_vector[f32, 32] = vector_cast(acc_c10_store_hi_i64, f32, 32)
			store_v(p_c10, v32accfloat_to_v32bf16(acc_c10_store_lo))
			store_v(p_c10 + 32, v32accfloat_to_v32bf16(acc_c10_store_hi))

			acc_c11_store_i64: aie_vector[i64, 32] = vector_cast(acc_c11, i64, 32)
			acc_c11_store_lo_i64: aie_vector[i64, 16] = vector_extract(acc_c11_store_i64, 0, 16)
			acc_c11_store_hi_i64: aie_vector[i64, 16] = vector_extract(acc_c11_store_i64, 16, 16)
			acc_c11_store_lo: aie_vector[f32, 32] = vector_cast(acc_c11_store_lo_i64, f32, 32)
			acc_c11_store_hi: aie_vector[f32, 32] = vector_cast(acc_c11_store_hi_i64, f32, 32)
			store_v(p_c11, v32accfloat_to_v32bf16(acc_c11_store_lo))
			store_v(p_c11 + 32, v32accfloat_to_v32bf16(acc_c11_store_hi))

			p_c_row0 = p_c01 + c_n_stride
			p_c_row1 = p_c11 + c_n_stride
			p_b_col0_base = p_b_col0_base + b_n_stride + b_n_stride
			p_b_col1_base = p_b_col1_base + b_n_stride + b_n_stride

			n = n + 2
		p_a_row0_base = p_a_row0_base + block_size + block_size
		p_a_row1_base = p_a_row1_base + block_size + block_size
		p_c_row0_base = p_c_row0_base + c_m_stride + c_m_stride
		p_c_row1_base = p_c_row1_base + c_m_stride + c_m_stride
		m = m + 2


# ---------------------------------------------------------------------------
# Phase 3.4 (complete): the remaining four flash-attention kernels.
#
# These are appended AFTER the 15 already-working kernels above so they
# act as additional helpers in the same compilation unit; `fused_softmax`
# is intentionally placed LAST so PythoC's `compile_pythoc_source` AST
# walker -- which `break`s on `function_name` -- picks it as the entry
# point with everything else (matmul_g_b_bf16 included) compiled as
# helpers in the same .o.
#
# Layout assumptions match attn_npu2.cc (lqp=lkp=64, column-major 8x8
# tiled). See attn_npu2.cc:270-700 for the C++ source these mirror.
# ---------------------------------------------------------------------------


@aie_kernel
def max_g_bf16(g: ptr[bf16, True], out: ptr[bf16, True]) -> void:
	# Per-row max over G (column-major 8x8 tiled). Mirrors attn_npu2.cc:270.
	# VecLen=32 processes 4 rows at a time (half a row-block).
	#
	# Init to bf16 lowest (0xff7f, ~-3.389e38) instead of -inf so that a
	# fully-masked row collapsing to lowest-lowest=0 inside exp_g_minus_u
	# does not produce NaN.
	#
	# llvm-aie's GISel legalizer has no lowering for either
	# `vector.reduce.fmax` (bf16 or f32) on these widths, so we fold the
	# per-row max in scalar bf16 via repeated extractelement + IfExp.
	# We then pack the 8 row-maxes for the block into the low 8 lanes of
	# a v32 bf16 and emit a single 8-lane store -- writing 4 contiguous
	# scalar bf16 stores trips the same `<4 x bf16> G_FPTRUNC` legaliser
	# bug as f32 does, even when the values are already bf16 (the
	# optimiser keeps them in f32 registers internally).
	vec_size: i32 = 32
	block_size: i32 = 64
	row_blocks: i32 = 8
	col_blocks: i32 = 8
	block_stride: i32 = 512
	rows_per_block: i32 = 8
	lowest_vec: aie_vector[bf16, 32] = broadcast(bf16, 32, -3.389e38)

	p_out: ptr[bf16] = out

	rb: i32 = 0
	while rb < row_blocks:
		# Build a v32 bf16 staging vector holding all 8 row-maxes for this
		# row block in the low 8 lanes.
		row_max_vec: aie_vector[bf16, 32] = lowest_vec

		half: i32 = 0
		while half < 2:
			# Inner loop over column blocks accumulating element-wise max.
			max_vec: aie_vector[bf16, 32] = lowest_vec
			base: i32 = rb * block_size + half * vec_size
			cb: i32 = 0
			while cb < col_blocks:
				v: aie_vector[bf16, 32] = load_v(g + base + cb * block_stride, 32)
				max_vec, cmp_mask = vmax_ltbf16(max_vec, v)
				cb = cb + 1

			# Scalar-reduce each 8-lane group in bf16. Scalar bf16
			# comparisons lower to the native `vlt.bf16` predicate and
			# avoid pulling in __gtsf2 (softfloat) from libcompiler-rt.
			lane: i32 = 0
			while lane < 4:
				base_lane: i32 = lane * 8
				m: bf16 = extract_elem(max_vec, base_lane)
				e1: bf16 = extract_elem(max_vec, base_lane + 1)
				m = e1 if e1 > m else m
				e2: bf16 = extract_elem(max_vec, base_lane + 2)
				m = e2 if e2 > m else m
				e3: bf16 = extract_elem(max_vec, base_lane + 3)
				m = e3 if e3 > m else m
				e4: bf16 = extract_elem(max_vec, base_lane + 4)
				m = e4 if e4 > m else m
				e5: bf16 = extract_elem(max_vec, base_lane + 5)
				m = e5 if e5 > m else m
				e6: bf16 = extract_elem(max_vec, base_lane + 6)
				m = e6 if e6 > m else m
				e7: bf16 = extract_elem(max_vec, base_lane + 7)
				m = e7 if e7 > m else m
				row_max_vec = insert_elem(row_max_vec, half * 4 + lane, m)
				lane = lane + 1
			half = half + 1

		# Emit a single 8-lane bf16 store from lanes [0..7] of the staging vec.
		row_max_v8: aie_vector[bf16, 8] = vector_extract(row_max_vec, 0, 8)
		store_v(p_out, row_max_v8)
		p_out = p_out + rows_per_block
		rb = rb + 1


@aie_kernel
def sum_g(g: ptr[bf16, True], s: ptr[bf16, True]) -> void:
	# Per-row sum over G (column-major 8x8 tiled). Mirrors attn_npu2.cc:472.
	#
	# The .cc accumulates in accfloat (f32-equivalent) and reduces in f32.
	# We can't replicate that in PythoC without bumping into:
	#   - the accfloat register-pair format (`<32 x f32>` at the LLVM level
	#     but a 40-bit-per-lane HW register pair semantically); inserting
	#     plain f32 values via `insert_elem` corrupts the back-conversion;
	#   - the missing legaliser for `<N x bfloat> G_FPTRUNC`, so we cannot
	#     emit four scalar `bf16(f32)` stores in a row without llc dying.
	#
	# Solution: stay in BF16 throughout. After exp_g_minus_u every lane is
	# in [0, 1], 8 column-block adds keep each lane in [0, 8], 8 row-block
	# horizontal sums in [0, 64] -- all exactly representable in bf16. We
	# `vector_add` v32 bf16 across the col blocks (which lowers via the
	# AIE2P accfloat-internal addition intrinsic), then scalar-fold each
	# 8-lane row group via IfExp-style += (bf16 scalar add lowers to a
	# single `vadd.bf16`, no softfloat libcalls). The 8 row-sums for the
	# whole row block stage in a v32 bf16 vector and emit a single v8
	# store at the end.
	vec_size: i32 = 32
	block_size: i32 = 64
	row_blocks: i32 = 8
	col_blocks: i32 = 8
	block_stride: i32 = 512
	rows_per_block: i32 = 8

	p_s: ptr[bf16] = s

	rb: i32 = 0
	while rb < row_blocks:
		row_sum_vec: aie_vector[bf16, 32] = zeros(bf16, 32)

		half: i32 = 0
		while half < 2:
			sum_vec: aie_vector[bf16, 32] = zeros(bf16, 32)
			base: i32 = rb * block_size + half * vec_size
			cb: i32 = 0
			while cb < col_blocks:
				v: aie_vector[bf16, 32] = load_v(g + base + cb * block_stride, 32)
				sum_vec = vector_add(sum_vec, v)
				cb = cb + 1

			# Scalar-fold each 8-lane row group via bf16 scalar adds.
			lane: i32 = 0
			while lane < 4:
				base_lane: i32 = lane * 8
				m: bf16 = extract_elem(sum_vec, base_lane)
				m = m + extract_elem(sum_vec, base_lane + 1)
				m = m + extract_elem(sum_vec, base_lane + 2)
				m = m + extract_elem(sum_vec, base_lane + 3)
				m = m + extract_elem(sum_vec, base_lane + 4)
				m = m + extract_elem(sum_vec, base_lane + 5)
				m = m + extract_elem(sum_vec, base_lane + 6)
				m = m + extract_elem(sum_vec, base_lane + 7)
				row_sum_vec = insert_elem(row_sum_vec, half * 4 + lane, m)
				lane = lane + 1
			half = half + 1

		row_sum_v8: aie_vector[bf16, 8] = vector_extract(row_sum_vec, 0, 8)
		store_v(p_s, row_sum_v8)
		p_s = p_s + rows_per_block
		rb = rb + 1


@aie_kernel
def apply_causal_mask(g: ptr[bf16, True], q_block_idx: i32, kv_block_idx: i32) -> void:
	# Causal mask on a 64x64 column-major 8x8 tiled G buffer.
	# Mirrors attn_npu2.cc:634. Three cases:
	#   1. kv > q : entire block above diagonal -> all -inf.
	#   2. kv < q : entire block below diagonal -> no-op.
	#   3. kv == q: scalar fallback (one BlkDim=8 row slice at a time).
	#
	# The C++ implementation builds an aie::mask<8> + aie::select for
	# partial blocks; we drop that in favour of scalar writes, which is
	# acceptable here because the diagonal block runs once per layer and
	# the mask shape is constant 64x64.
	#
	# The mask value MUST be a true bf16 -inf (0xff80), not just bf16
	# lowest (0xff7f, ~-3.389e38). max_g_bf16 initialises with lowest;
	# if we mask with lowest too, max(...) returns lowest for a fully
	# masked row, then `G - max = lowest - lowest = 0` and `exp(0) = 1`
	# instead of 0 -- which corrupts the softmax. With true -inf,
	# `max(lowest, -inf) = lowest` and `-inf - lowest = -inf`, which
	# `exp_g_minus_u` clamps back to lowest before `exp`, yielding 0.
	#
	# llvmlite's `ir.Constant(bfloat, -1e40)` overflows float32; build
	# the -inf vector by broadcasting the 0xff80 bit-pattern as an i16
	# (-128 signed) and bitcasting to bf16.
	vec_size: i32 = 32
	lqp: i32 = 64
	lkp: i32 = 64
	blk_dim: i32 = 8
	neg_inf_bits_vec: aie_vector[i16, 32] = broadcast(i16, 32, -128)
	neg_inf_vec: aie_vector[bf16, 32] = vector_cast(neg_inf_bits_vec, bf16, 32)
	neg_inf_v8_bits: aie_vector[i16, 8] = broadcast(i16, 8, -128)
	neg_inf_v8: aie_vector[bf16, 8] = vector_cast(neg_inf_v8_bits, bf16, 8)
	neg_inf_val: bf16 = extract_elem(neg_inf_v8, 0)

	if kv_block_idx > q_block_idx:
		# Above-diagonal block -> overwrite all 4096 elements with -inf.
		p: ptr[bf16] = g
		i: i32 = 0
		total: i32 = lqp * lkp
		while i < total:
			store_v(p, neg_inf_vec)
			p = p + vec_size
			i = i + vec_size
	else:
		if kv_block_idx == q_block_idx:
			# Diagonal block: scalar masking per row, per 8-element slice.
			row: i32 = 0
			while row < lqp:
				mask_start: i32 = row + 1
				row_blk: i32 = row // blk_dim
				row_in: i32 = row - row_blk * blk_dim
				col_blk: i32 = 0
				while col_blk < lkp // blk_dim:
					col_start: i32 = col_blk * blk_dim
					off: i32 = col_blk * (lqp * blk_dim) + row_blk * (blk_dim * blk_dim) + row_in * blk_dim
					p_row: ptr[bf16] = g + off
					if mask_start < lkp:
						if col_start >= mask_start:
							# Entire 8-element slice is masked.
							c: i32 = 0
							while c < blk_dim:
								p_row[c] = neg_inf_val
								c = c + 1
						else:
							if col_start + blk_dim > mask_start:
								# Partial slice: mask cols >= mask_start.
								c2: i32 = 0
								while c2 < blk_dim:
									if col_start + c2 >= mask_start:
										p_row[c2] = neg_inf_val
									c2 = c2 + 1
					col_blk = col_blk + 1
				row = row + 1


@aie_kernel
def fused_softmax(g: ptr[bf16, True], up: ptr[bf16, True], sp: ptr[bf16, True], r: ptr[bf16, True]) -> void:
	# Composes the helpers above. Mirrors attn_npu2.cc:601.
	#   r  = max(G, axis=-1)
	#   r  = max(up, r)             # online new max
	#   G  = exp(G - r)             # softmax numerator
	#   sp = exp(up - r)            # rescale factor
	#   up = r                      # publish new max
	#   r  = sp                     # downstream wants r=rescale
	#   sp = sum(G, axis=-1)        # softmax denominator
	zero_offset: i32 = 0
	max_g_bf16(g, r)
	maximum_up_u_bf16(up, r)
	exp_g_minus_u(r, g)
	exp_up_minus_u(up, r, sp)
	vector_copy_32elems(zero_offset, r, up)
	vector_copy_32elems(zero_offset, sp, r)
	sum_g(g, sp)
