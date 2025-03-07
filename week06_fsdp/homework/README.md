# Week 6 assignment

## Task 1 — `FULL_SHARD` (no overlap) (6 points)

`hw_fsdp.py` contains a draft of a simple FSDP implementation (we'll call it
hwFSDP).

- Fill in the TODO's in `hw_fsdp.py` to achieve a functioning FSDP
  implementation.
- Validate your hwFSDP against FSDP2 using `train.py` (more info in
  [Notes](#notes) section).
- Make sure your losses and grad norms match the ones of a FSDP2 run. Attach
  both runs' logs for validation.
- Make sure to free unsharded params after each `FSDPModule`'s forward and
  backward.
- Make sure to free unsharded grads after each `FSDPModule`'s reduce-scatter.
- Attach a memory snapshot of a hwFSDP run for validation.
- Make sure the memory usage is similar to the one of FSDP2 (attach a memory
snapshot of a FSDP2 run as well).
- Sharded params must me instances of `DTensor` with correct mesh and
  placements.
- Functioning forward pass gets you `3 points` and a full functioning step gets
  you another `3 points`.

No computation / communication overlap is required in this part of the
assignment.

## Task 2 — `FULL_SHARD` (implicit forward prefetch) (2 points)

- Make changes to hwFSDP to overlap forward communications (parameter gathering)
  with forward computations. Make use of multiple CUDA streams and use CUDA
  event to sync them.
- Make sure losses and grad norms still match the FSDP2 ones (or are close).
- Make sure memory usage is still fine.
- Attach a profile trace which depicts the overlap (more on traces in
  [Notes](#notes) section).

## Task 3 — `FULL_SHARD` (explicit backward prefetch) (2 points)

- Overlap backward communications (gradient reduction and parameter gathering)
  with backward computations.
- Just as before, validate losses and grad norms, make sure memory usage is
  okay.
- Attach a profile trace which depicts the overlap.

## Bonus

### Activation checkpointing support (1 point)

- Make changes to hwFSDP to support using activation checkpointing with hwFSDP.
- Validate losses, grad norms and memory, if you've achieved overlap make sure
  it's still there.

### `reshard_after_forward=False` support (1 point)

- Make changes to hwFSDP to support no resharding after forward (aka ZeRO-2).
- Validate losses, grad norms and memory, if you've achieved overlap make sure
  it's still there.
- Attach a trace which depicts shows there are to parameter all-gathers during
  backward pass

## Notes

- It is recommended to debug your code using a
  [dev-container](https://code.visualstudio.com/docs/devcontainers/containers)
  with configuration provided in `.devcontainer.json` and debug configs from
  `.vscode/launch.json`.
- Debug configs launch hwFSDP and FSDP2 runs of the `train.py` script.
- `train.py` runs a debug Llama pre-train, logs metrics, saves profiling traces
  and memory snapshots.
- Overlap can be checked using profiling traces. To visualize them use
  [perfetto.dev](perfetto.dev). `train.py` saves profiling traces to
  `train/(hw-fsdp|fsdp-2)/profile_trace`.
- Memory snapshots can be visualized using
  [pytorch.org/memory_viz](pytorch.org/memory_viz). `train.py` saves memory
  snapshots to `train/(hw-fsdp|fsdp-2)/memory_snapshot`.
- Tip: to get a clear picture of the overlap GPUs must be pretty well utilized,
  to achieve that change the model flavour from `debugmodel` to `1B` and
  increase seq-len until the utilization is high enough (by default `train.py`
  runs a small debug model).
