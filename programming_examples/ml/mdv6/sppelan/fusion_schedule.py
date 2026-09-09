"""Authored SPP9 storage/ordering model, NOT an executable IRON schedule.

Run from the MDV6 directory: python -m sppelan.fusion_schedule
Only ingress/egress use the external-BO TensorContract. Internal reservations
are independent metadata and do not extend DeviceBuffer's storage support.
"""
from dataclasses import dataclass
import json

from device_buffers import TensorContract


H = W = 20
C_IN, C_NECK, C_OUT = 256, 128, 256
COLUMNS, ROWS, SHARD, STRIPE = 4, 4, 8, 16
L1_LIMIT, L2_LIMIT = 64 * 1024, 512 * 1024
PHASES = ("project_pool_store", "gather_project_drain")


@dataclass(frozen=True)
class Reservation:
    name: str
    storage: str
    nbytes: int
    first: int
    last: int


def boundaries():
    def contract(name, channels, producer, consumers, last_use):
        shape = (H, W, channels)
        return TensorContract(name, shape, shape, (W * channels, channels, 1),
                              "HWC", "spp9", producer, consumers, last_use,
                              (0, 0, 0), shape)
    return (contract("spp9_input", C_IN, "rep_elan8", ("spp9",), 0),
            contract("spp9_output", C_OUT, "spp9", ("upsample10", "re21_input_concat"), 1))


def reservations():
    """Worst single-core and single-memtile budgets, with explicit lifetimes.

    'reserve' is conservative bookkeeping, not proof that stack/bank placement
    fits. Weight staging in L2 is included even when weights are also in L1.
    """
    return (
        Reservation("four_channel_shard_features", "L1", 4 * H * W * SHARD * 2, 0, 0),
        Reservation("input_ping_pong", "L1", 2 * STRIPE * C_IN * 2, 0, 0),
        Reservation("conv1_weights_bn", "L1", (C_IN * SHARD + 2 * SHARD) * 2, 0, 0),
        Reservation("gather_ping_pong", "L1", 2 * STRIPE * 4 * C_NECK * 2, 1, 1),
        Reservation("conv5_weights_bn", "L1", (4 * C_NECK * 16 + 2 * 16) * 2, 1, 1),
        Reservation("output_ping_pong", "L1", 2 * STRIPE * 16 * 2, 1, 1),
        Reservation("stack_reserve", "L1", 4096, 0, 1),
        Reservation("scratch_alignment_reserve", "L1", 4096, 0, 1),
        Reservation("resident_features", "L2", 4 * ROWS * H * W * SHARD * 2, 0, 1),
        Reservation("input_ping_pong", "L2", 2 * STRIPE * C_IN * 2, 0, 0),
        Reservation("conv1_weight_staging", "L2", ROWS * (C_IN * SHARD + 2 * SHARD) * 2, 0, 0),
        Reservation("gather_ping_pong", "L2", 2 * STRIPE * 4 * C_NECK * 2, 1, 1),
        Reservation("conv5_weight_staging", "L2", ROWS * (4 * C_NECK * 16 + 2 * 16) * 2, 1, 1),
        Reservation("output_ping_pong", "L2", 2 * STRIPE * 64 * 2, 1, 1),
        Reservation("routing_alignment_reserve", "L2", 16 * 1024, 0, 1),
    )


def peak_bytes(storage, records=None):
    records = reservations() if records is None else records
    return max(sum(r.nbytes for r in records
                   if r.storage == storage and r.first <= phase <= r.last)
               for phase in range(len(PHASES)))


def source_for_channel(channel):
    """Logical concat K -> (source column, row, pool level, local channel)."""
    if not 0 <= channel < 4 * C_NECK:
        raise ValueError("concat channel outside [0, 512)")
    level, neck_channel = divmod(channel, C_NECK)
    worker, local_channel = divmod(neck_channel, SHARD)
    column, row = divmod(worker, ROWS)
    return column, row, level, local_channel


def gather_segments(stripe):
    """Bounded microtransfer description; offsets are bf16 elements, not bytes.

    Source memtile layout [row, level, pixel, channel8]; destination stripe
    layout [pixel16, concat_channel512]. Each segment has 16 rows of 8 values.
    Descriptor encoding and routing feasibility are deliberately unproven.
    """
    if not 0 <= stripe < H * W // STRIPE:
        raise ValueError("stripe outside [0, 25)")
    for k in range(0, 4 * C_NECK, SHARD):
        column, row, level, _ = source_for_channel(k)
        yield {"source_column": column,
               "source_offset": ((row * 4 + level) * H * W + stripe * STRIPE) * SHARD,
               "destination_offset": k,
               "rows": STRIPE, "width": SHARD,
               "source_stride": SHARD, "destination_stride": 4 * C_NECK}


def validate(records=None):
    records = reservations() if records is None else tuple(records)
    if COLUMNS * ROWS * SHARD != C_NECK:
        raise ValueError("worker channel shards must cover the neck channels")
    if H * W % STRIPE:
        raise ValueError("spatial extent must be divisible by stripe size")
    if not records:
        raise ValueError("storage reservations are required")
    for record in records:
        if (record.storage not in ("L1", "L2") or record.nbytes <= 0
                or not 0 <= record.first <= record.last < len(PHASES)):
            raise ValueError(f"invalid storage reservation: {record.name}")
    for storage, limit in (("L1", L1_LIMIT), ("L2", L2_LIMIT)):
        if peak_bytes(storage, records) > limit:
            raise ValueError(f"{storage} peak exceeds per-tile capacity")
    if len(set(source_for_channel(k) for k in range(4 * C_NECK))) != 4 * C_NECK:
        raise ValueError("concat channel mapping is not one-to-one")
    return {"status": "authored; not compiled or hardware validated",
            "workers": COLUMNS * ROWS, "stripes": H * W // STRIPE,
            "l1_peak_bytes_per_worker": peak_bytes("L1", records),
            "l2_peak_bytes_per_column": peak_bytes("L2", records),
            "boundary_bytes": {c.name: c.nbytes for c in boundaries()},
            "resident_feature_bytes": 4 * H * W * C_NECK * 2,
            "logical_activation_external_bytes": 2 * H * W * C_IN * 2,
            "gather_destination_bytes": COLUMNS * H * W * 4 * C_NECK * 2,
            "gather_segments_per_destination_stripe": len(list(gather_segments(0)))}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
