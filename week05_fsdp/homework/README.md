# Week 5 assignment

`fsdp.py` contains a backbone for your FSDP implementation. Some parts marked
with TODO's are missing for you to fill them in.

## Task 1 — no overlap (6 points)

- Complete all the `task1` TODO's in `fsdp.py`.
- Validate the correctness of your implementation against FSDP2 using `train.py`
  (more on that in the [Notes](#notes) section):
  - Make sure losses and grad norms match the FSDP2 ones. Your implementation
    should pass the `test_fsdp` test in `test.py` (run `pytest
    test.py::test_fsdp`).
- Validate the memory usage of your implementation against FSDP2 (by comparing
  the memory snapshots):
  - Make sure you free the unsharded params after each `FSDPModule`'s forward
    and backward.
  - Make sure you free unsharded grads after each `FSDPModule`'s gradient
    reduction.
  - Attach a memory snapshot of both your implementation and FSDP2 for
    validation.
- Sharded params must be instances of `DTensor` with correct mesh and
  placements.
- Functioning forward pass gets you `3 points` and a full functioning step gets
  you another `3 points`.

No computation / communication overlap is required in this part of the
assignment.

## Task 2 — implicit forward prefetch (2 points)

- Complete all the `task2` TODO's to overlap forward comms (parameter gathering)
  with forward compute.
- Make sure losses and grad norms still match the FSDP2 ones (`test_fsdp`
  passes).
- Make sure memory usage is still fine (it might increase slightly as there's
  now memory allocated for async prefetching).
- Attach a profile trace which depicts the overlap (more on traces in the
  [Notes](#notes) section).

## Task 3 — explicit backward prefetch and gradient reduction overlap (2 points)

- Complete all the `task3` TODO's to overlap backward comms (parameter gathering
  and gradient reduction) with backward compute.
- Make sure memory usage is still fine (again, it might increase slightly as
  there's now more memory allocated for async prefetching and gradient
  reduction).
- Attach a profile trace which depicts the overlap.

## Bonus

### `reshard_after_forward=False` (1 point)

- Make changes to support no resharding after forward (ZeRO-2).
- Validate losses, grad norms and memory usage, if you've achieved overlap make
  sure it's still there.
- Attach a trace which shows there is no parameter gathering during backward
  pass.

### `reshard_after_backward=False` + gradient accumulation (1 point)

- Make changes to support no resharding after backward with
  `gradient_accumulation_steps >= 2` (ZeRO-2 with gradient accumulation).
- Validate losses, grad norms and memory usage, if you've achieved overlap make
  sure it's still there.
- Attach a trace which shows there is no parameter gathering during both during
  backward pass and all forward passes except the first one.

### `reduce_grads=False` + gradient accumulation (1 point)

- Make changes to support no gradient reduction (before the last gradient
  accumulation step) with `gradient_accumulation_steps >= 2` (ZeRO-1 with
  gradient accumulation).
- Validate losses, grad norms and memory usage, if you've achieved overlap make
  sure it's still there.
- Attach a trace which shows there is no gradient reduction during both during
  all backward passes except the last one.

## Notes

### Setting up the environment

- Running `uv sync` should be enough to set up the environment (you might need
  to install `uv` first
  `https://docs.astral.sh/uv/getting-started/installation`).
- You can then activate the virtual env with `.venv/bin/activate` or use `uv run
  ...`.
- You will also need to download dataset and tokenizer files:
  ```bash
  mkdir -p c4_test && curl https://raw.githubusercontent.com/pytorch/torchtitan/refs/heads/main/tests/assets/c4_test/data.json --output c4_test/data.json
  mkdir -p tokenizer \
    && curl https://raw.githubusercontent.com/pytorch/torchtitan/refs/heads/main/tests/assets/tokenizer/tokenizer.json --output tokenizer/tokenizer.json \
    && curl https://raw.githubusercontent.com/pytorch/torchtitan/refs/heads/main/tests/assets/tokenizer/tokenizer_config.json --output tokenizer/tokenizer_config.json
  ```

### Debugging

- There's a debug config in `.vscode/launch.json` for you to use.
- Overlap can be verified using profiling traces. To visualize them use
  [perfetto.dev](perfetto.dev). `train.py` saves profiling traces to
  `traces/(effdl|fsdp2)`.
- Memory snapshots can be visualized using
  [pytorch.org/memory_viz](pytorch.org/memory_viz). `train.py` saves memory
  snapshots to `snapshots/(effdl|fsdp2)`.
- To get a clear picture of the overlap GPUs must be pretty well utilized. To
  achieve that change the model from `debug` to `1B` and increase `seq_len`
  until the utilization is high enough (by default `train.py` runs a small
  `debug` model with a small `seq_len`).
