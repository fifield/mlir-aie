from aie.iron.pythoc import aie_kernel

from pythoc import ptr, i16, i32, i64, f32, bf16, void
from pythoc.aie import ACC2048_accfloat_add_conf, BFP576_BFP576_ACC2048_mac_conf, I1024_I1024_ACC2048_bf_mul_conf, I512_I512_ACC1024_bf_mac_conf, I512_I512_ACC1024_bf_mul_conf, I512_I512_ACC1024_bf_negmul_conf, acc_extract, acc_grow, aie_vector, broadcast, concat, getExpBf16, load_v, set_ctrl_reg, store_v, v32accfloat_to_v32bf16, v32bf16_to_v32accfloat, v64accfloat_to_v64bfp16ebs8, vector_add, vector_blend, vector_cast, vector_extract, vector_insert, vector_mul, vector_sub, vmax_ltbf16, vshuffle, zeros


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


@aie_kernel
def accum_sp_r_s_pythoc(sp: ptr[bf16, True], r: ptr[bf16, True], s: ptr[bf16, True]) -> void:
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
def div_gp_sp_pythoc(sp: ptr[bf16, True], gp: ptr[bf16, True]) -> void:
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
def matmul_a_b_bf16_pythoc(a_in: ptr[bf16, True], b_in: ptr[bf16, True], out: ptr[bf16, True]) -> void:
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
def matmul_g_b_bf16_pythoc(g_in: ptr[bf16, True], b_in: ptr[bf16, True], out: ptr[bf16, True]) -> void:
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




