"""Per-shape multicore conv config registry.

Defines the (label, n_cores, tile_h, tile_w, ic, oc_block, kernel_size,
stride, ppc) tuple for every conv shape the model dispatches. Merged-ELF
builders (build_x1_mc.py, build_pair_rn3.py, etc.) look up shapes by
label and feed them to aie2_multicore.py as CLI args.

(Previously lived in build_multicore.py, which was an xclbin builder.
The xclbin path was retired in the Phase G+ cleanup; only the shape
table from that file is still needed.)
"""

# (label, n_cores, tile_h, tile_w, ic, oc_block, kernel_size, stride, ppc)
CONFIGS = [
    # Stems (stride 2)
    ("mc_ftconv0",       32, 20, 20,   8,  32, 3, 2, 1),
    ("mc_ftconv0_p2",    32, 20, 20,   8,  32, 3, 2, 2),
    ("mc_ftconv1",       32, 12, 12,  32,  16, 3, 2, 1),
    ("mc_ftconv1_p2",    32, 12, 12,  32,  16, 3, 2, 2),

    # ELAN2 (160x160, 64ch)
    ("mc_elan_c1",       32,  8,  8,  64,  64, 1, 1, 1),
    ("mc_elan_c3",       32,  8,  8,  32,  32, 3, 1, 1),
    ("mc_elan_c3_p2",    32,  8,  8,  32,  32, 3, 1, 2),
    ("mc_elan_c3_p4",    32,  8,  8,  32,  32, 3, 1, 4),
    ("mc_elan_c4",       32,  8,  8, 128,  64, 1, 1, 1),

    # AConv (stride 2)
    ("mc_aconv3",        32,  8,  8,  64,  16, 3, 2, 1),
    ("mc_aconv3_p2",     32,  8,  8,  64,  16, 3, 2, 2),
    ("mc_aconv5",        32,  4,  4,  96,   8, 3, 2, 1),
    ("mc_aconv5_p2",     32,  4,  4,  96,   8, 3, 2, 2),
    ("mc_aconv5_p4",     32,  4,  4,  96,   8, 3, 2, 4),
    ("mc_aconv7",        32,  4,  4, 128,   8, 3, 2, 1),
    ("mc_aconv16",       32,  4,  4,  64,   8, 3, 2, 1),
    ("mc_aconv19",       32,  4,  4,  96,   8, 3, 2, 1),

    # RepNCSPELAN4 (80x80, 128ch)
    ("mc_re4_c1",        32, 10, 10, 128,  64, 1, 1, 1),
    ("mc_re4_c3",        32, 12, 12,  64,  16, 3, 1, 1),
    ("mc_re4_c3_p2",     32, 12, 12,  64,  16, 3, 1, 2),
    ("mc_re4_c4",        32,  8,  8, 256,  32, 1, 1, 1),
    ("mc_re4_rn1",       32, 16, 16,  64,  32, 1, 1, 1),
    ("mc_re4_rn3",       32,  8,  8,  32,  32, 3, 1, 1),
    ("mc_re4_rn3_p2",    32,  8,  8,  32,  32, 3, 1, 2),
    ("mc_re4_rn3_p4",    32,  8,  8,  32,  32, 3, 1, 4),

    # RepNCSPELAN6 (40x40, 192ch)
    ("mc_re6_c1",        32,  8,  8, 192,  32, 1, 1, 1),
    ("mc_re6_c3",        32,  8,  8,  96,  16, 3, 1, 1),
    ("mc_re6_c3_p2",     32,  8,  8,  96,  16, 3, 1, 2),
    ("mc_re6_c4",        32,  4,  4, 384,  32, 1, 1, 1),
    ("mc_re6_rn1",       32, 10, 10,  96,  48, 1, 1, 1),
    ("mc_re6_rn3",       32,  8,  8,  48,  16, 3, 1, 1),
    ("mc_re6_rn3_p2",    32,  8,  8,  48,  16, 3, 1, 2),
    ("mc_re6_rnm",       32,  8,  8,  96,  48, 1, 1, 1),

    # RepNCSPELAN8 (20x20, 256ch)
    ("mc_re8_c1",        32,  4,  4, 256,  32, 1, 1, 1),
    ("mc_re8_c3",        32,  4,  4, 128,  16, 3, 1, 1),
    ("mc_re8_c3_p2",     32,  4,  4, 128,  16, 3, 1, 2),
    ("mc_re8_c4",        32,  4,  4, 512,  16, 1, 1, 1),
    ("mc_re8_rn1",       32,  8,  8, 128,  64, 1, 1, 1),
    ("mc_re8_rn3",       32,  8,  8,  64,  16, 3, 1, 1),
    ("mc_re8_rn3_p2",    32,  8,  8,  64,  16, 3, 1, 2),
    ("mc_re8_rnm",       32,  4,  4, 256,  32, 1, 1, 1),

    # SPP / neck
    ("mc_spp_c1",        32,  4,  4, 256,  32, 1, 1, 1),
    ("mc_re12_c1",       32,  4,  4, 448,  32, 1, 1, 1),
    ("mc_re15_c1",       32,  6,  6, 320,  32, 1, 1, 1),
    ("mc_re15_c4",       32,  8,  8, 256,  32, 1, 1, 1),
    ("mc_re15_rnm",      32,  8,  8, 128,  64, 1, 1, 1),
    ("mc_re18_c1",       32,  4,  4, 288,  32, 1, 1, 1),
    ("mc_re21_c1",       32,  4,  4, 384,  32, 1, 1, 1),
]
