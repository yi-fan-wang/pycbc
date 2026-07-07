# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this fork

This is a fork of [gwastro/pycbc](https://github.com/gwastro/pycbc) (branch `gpu`, remote `yi-fan-wang/pycbc`). PyCBC is a gravitational-wave search and parameter-estimation toolkit that traditionally runs on CPU. **The goal of this fork is to extend both major functions — the matched-filter search (`bin/pycbc_inspiral`) and Bayesian inference (`bin/pycbc_inference`) — to run on GPU using CuPy.** Upstream PR [gwastro/pycbc#5249](https://github.com/gwastro/pycbc/pull/5249) established the prototype `CUPYScheme` and the initial `*_cupy.py` backends; work here builds on that. The older `CUDAScheme` (PyCUDA-based) backends are legacy — new GPU work should target CuPy.

## Common commands

```bash
# Editable install (needs a C/C++ compiler; builds Cython extensions)
pip install -e .

# Full unit-test suite (upstream CI uses pixi; tasks defined in pyproject.toml)
pixi run -e unittest test-unittest      # equivalent to: pytest
pixi run -e search test-search          # bash tools/pycbc_test_suite.sh, PYCBC_TEST_TYPE=search
pixi run -e inference test-inference

# Without pixi, plain pytest works if dependencies are installed:
pytest test/test_array.py               # one test file
pytest test/test_array.py -k inner      # one test by name

# Many test files are scheme-parametrized via test/utils.py:
python test/test_schemes.py --scheme cuda   # choices are currently only cpu/cuda
```

Notes:
- `pytest.ini` sets `testpaths = test` and ignores `test/test_fftw_pthreads.py`.
- `test/utils.py` (`parse_args_all_schemes`) only knows `cpu` and `cuda` — it has **not** been extended to `cupy` yet. GPU-correctness testing of CuPy backends is part of the porting work.
- CuPy and a visible GPU may not be available in the current shell (e.g. login node). Check with `python -c "import cupy; cupy.cuda.runtime.getDeviceCount()"` before assuming GPU tests can run locally.

## Running on GPU

Two entry points select the CuPy backend:
- CLI flag on executables that call `scheme.insert_processing_option_group`: `--processing-scheme cupy`
- Environment variable `PYCBC_SCHEME=cupy` — sets the process-wide `DefaultScheme` at import time (see bottom of `pycbc/scheme.py`)

Or programmatically:
```python
from pycbc.scheme import CUPYScheme
with CUPYScheme():   # all schemed operations inside dispatch to *_cupy backends
    ...
```
`CUPYScheme` supports MPI multi-GPU: with MPI enabled it assigns `device = rank % device_count`.

## Architecture: the scheme dispatch system

This is the core mechanism for everything GPU-related.

1. **Schemes** (`pycbc/scheme.py`): context managers `CPUScheme`, `MKLScheme`, `NumpyScheme`, `CUDAScheme` (legacy PyCUDA), `CUPYScheme`. A global singleton `mgr` tracks the active scheme; only one non-default scheme may be active at a time. `scheme_prefix` maps scheme class → string (`"cupy"`, `"cpu"`, ...).

2. **`@schemed(BACKEND_PREFIX)` decorator**: a function decorated in a frontend module (e.g. `pycbc/types/array.py` with `BACKEND_PREFIX="pycbc.types.array_"`) is a stub. At call time the decorator imports `<prefix><scheme_name>` (e.g. `pycbc.types.array_cupy`), walks the scheme class MRO as fallback, finds the same-named function there, caches it per-scheme, and calls it. Failure raises `RuntimeError: Failed to find implementation of <fn> for cupy scheme`.

3. **Backend module convention**: each schemed frontend has sibling modules per scheme, e.g.
   - `pycbc/types/array.py` → `array_cpu.pyx`, `array_cuda.py`, `array_cupy.py`
   - `pycbc/filter/matchedfilter.py` → `matchedfilter_cpu.pyx`, `matchedfilter_numpy.py`, `matchedfilter_cupy.py`, ...

   **To port a function to GPU: implement a same-named function in the corresponding `_cupy.py` module.** CPU backends are often Cython (`.pyx`, built by `setup.py`); CuPy backends are pure Python using `cupy` (elementwise/RawKernels where needed).

4. **Existing CuPy backends** (the current porting frontier — mostly the search-side kernels):
   `types/array_cupy.py`, `fft/cupyfft.py` + `fft/backend_cupy.py`, `filter/matchedfilter_cupy.py`, `events/threshold_cupy.py`, `vetoes/chisq_cupy.py`, `waveform/spa_tmplt_cupy.py`, `waveform/decompress_cupy.py`, `waveform/utils_cupy.py`.
   Frontends using `@schemed` today: `types/array.py`, `fft/backend_support.py`, `filter/matchedfilter.py`, `vetoes/chisq.py`, `events/eventmgr.py`, `waveform/{spa_tmplt,compress,utils}.py`.

5. **Data movement**: `pycbc.types.Array` wraps `_data` (a numpy or cupy ndarray). Arrays migrate to the active scheme's memory lazily when touched inside a scheme context (see `test/test_schemes.py` for the expected semantics). `TimeSeries`/`FrequencySeries` subclass `Array`, so GPU support flows from the `Array` backend upward. Code that calls `.numpy()`, `.lal()`, or hands `_data` to LAL/scipy forces a device→host copy — these are the hotspots to hunt when profiling GPU runs.

6. **FFT layer** (`pycbc/fft/`): separate from `@schemed` — it has its own backend registry (`backend_support.py`, `func_api.py`, `class_api.py`). The CuPy FFT backend is `cupyfft.py`, registered via `backend_cupy.py`.

## The two pipelines to port

- **Search** (`bin/pycbc_inspiral`): strain conditioning (`pycbc/strain`) → template generation (`pycbc/waveform`, `pycbc/tmpltbank`) → matched filter (`pycbc/filter/matchedfilter.py`) → thresholding/clustering (`pycbc/events`) → chi-squared vetoes (`pycbc/vetoes`). **Status 2026-07: pycbc_inspiral runs end-to-end under `--processing-scheme cupy`** on the examples/search GW170814 data (`/work/yifanwang/gpu/search_gpu_test/run_inspiral.sh {cpu,cupy} out.hdf`): 99.8% of triggers match CPU 1:1, identical loudest event (GW170814, snr 9.4611), SNR rel diff ~3e-6. Bugs fixed to get there (all in `_cupy` prototype code, PR-worthy): undefined `inner_real`/`abs_arg_max_complex` in array_cupy; scalar reductions returning device arrays (now `.item()` host scalars, matching the CUDA backend's `.get()` convention); `backend='nvcc'` RawKernels (NVRTC works, no toolkit needed) and `#include <cstdint>` (NVRTC has no C++ stdlib); transposed grid/block launch config in `CUDAThresholdCluster` (shared-mem overflow → CUDA_ERROR_ILLEGAL_ADDRESS); `sinegauss.fd_sine_gaussian` copy=False on a host array. The chisq CPU-vs-GPU %-level trigger differences were investigated and are **benign**: identical inputs agree to ~1e-5 and the GPU kernel (exact integer-mod phase) is *closer* to fp64 truth than the CPU fp32 phasor recurrence; the trigger-level differences come entirely from fp32-rounding-level differences (~4e-6 median in-band) in the *data-estimated PSD*, amplified by the ill-conditioned equal-power bin-edge searchsorted near the template's high-frequency tail (edges shift by up to ~170 samples where power density → 0). Making bins bit-stable across backends would require computing the sigma_vec cumsum in float64 (cheap, optional upstream improvement). The never-exercised non-pow2 GPU shift_sum path had three latent bugs (extra `N` in signature, `*args` instead of `tuple(args)`, catastrophic fp32 `sincosf(phase*i)` at large i) — all fixed; both paths now agree with fp64 truth to ~3e-6. Remaining for "search fully on GPU": run the whole examples/search workflow with GPU inspiral jobs.
- **Inference** (`bin/pycbc_inference`): models in `pycbc/inference/models/` (`gaussian_noise.py`, `relbin.py`, `marginalized_gaussian_noise.py`, ...) compute likelihoods via `pycbc.filter` inner products and `pycbc.waveform` generators; samplers in `pycbc/inference/sampler/`. **Status 2026-07: GaussianNoise.loglr runs correctly under CUPYScheme** (agrees with CPU to ~1e-13; benchmark `/work/yifanwang/gpu/inference_gpu_test/gaussnoise_bench.py`, config via BENCH_* env vars). Required fixing `Array.__array_ufunc__` to operate on `_data` instead of round-tripping through `.numpy()` (host result + copy=False blew up under CUPYScheme; also needless transfers). Measured per-loglr: BBH 8s: cpu 1.3ms / gpu 2.1ms; BNS 256s TaylorF2: 43.6 / 31.4ms; BNS 256s PhenomD: 480 / 374ms. **The bottleneck is host-side lalsim waveform generation (85-95% of the cost), not the inner products** — single-likelihood GPU gains are Amdahl-capped at ~1.3×. The inference GPU strategy is therefore: GPU-native waveforms (phenomxpy plugin / cupy TaylorF2-XAS port) and/or batched likelihood evaluation (many samples per kernel launch), and/or relbin (no full-grid waveforms).
  **Batched prototype** (`/work/yifanwang/gpu/inference_gpu_test/batch_taylorf2.py` + `bench_batch.py`): one RawKernel launch computes the full B×Nf complex128 aligned-spin TaylorF2 matrix (PN coefficients from lalsim host-side, ~7 us/point) and loglr for all rows via two matrix ops. Validated to 1.5e-11 (waveforms, vs fp64 reference) and 1e-14 (loglr); the pycbc CPU `spa_tmplt` fp32 anchor differs by 6.6e-3 — that is the *CPU's* fp32 phase error, same story as the chisq investigation. Throughput at B=256, BNS 256 s: **0.084 ms per likelihood** (vs 43.6 ms scalar CPU GaussianNoise — ~500×). Breakdown: waveform matrix ~3 ms, batched loglr (einsum) 16.9 ms, host coeffs 1.7 ms. Caveats: single detector, no antenna response yet.
  **Full BNS PE comparison** (`pe_bns_compare.py`, emcee vectorize=True, 256 walkers, 6 params, SNR-35 zero-noise injection): identical likelihood/sampler/init on both sides. CPU scalar loop: 400 steps = 102,400 likelihoods in **3878 s**; GPU batched: **6.8 s** (571×); a 2000-step GPU chain takes 32 s. Posterior marginals statistically indistinguishable: KS(cpu,gpu) 0.12-0.19, at/below the same-sampler convergence-noise scale KS(gpu400,gpu2000) 0.11-0.36. Note emcee stretch-move mixing is slow here (acceptance ~0.10) — chains at 400 steps are not fully converged, symmetrically for both runs. Next: 2 detectors + antenna response, wrap as a pycbc inference model class, consider a better-mixing vectorized sampler.

## GPU waveforms (the inference-side strategy)

lalsimulation waveforms are host-side only. Two GPU-native reference implementations, both cloned as siblings of this repo:
- `/work/yifanwang/gpu/phenomxpy` — python IMRPhenomT family, CPU (numba) / GPU (cupy) selected per-call via `cuda=True`; parameters use `f_min`/`eta`/`total_mass` style (`phenomxpy.utils.convert_params` converts from `mass1/mass2/spin1z...`); `compute_polarizations(times)` returns `(hp, hc, times)`.
- `/work/yifanwang/gpu/BBHX-waveform-model` — the BBHx LISA plugin, the model for plugin packaging (entry_points `pycbc.waveform.fd_det`, `pycbc.waveform.length`).

Our plugin `/work/yifanwang/gpu/pycbc-phenomxpy-plugin` registers `PyIMRPhenomT{,HM,P,PHM}` (td + length entry points). It checks the active scheme and passes `cuda=True` under CUPYScheme so the returned TimeSeries wraps a cupy array directly. CPU validation: match vs lalsim `IMRPhenomT` = 1.000000 for both polarizations.

Trap fixed in `td_fd_waveform_transform` (waveform.py): registering a TD plugin plus a length estimator used to make `cpu_fd[apx]=get_fd_waveform_from_td` and then override `cpu_td[apx]=get_td_waveform_from_fd` — infinite mutual recursion. The guard (`cpu_fd[approximant] is not get_fd_waveform_from_td`) must be preserved when merging upstream.

`cupy_td`/`cupy_fd` in waveform.py are ChainMaps layered over the cpu dicts: CPU approximants (incl. late-registered plugins) are visible under CUPYScheme automatically; GPU-native overrides go in `_cupy_td/fd_approximants`.

## Environment / cluster

- Login node (hypatia*.aei.mpg.de) has NO GPU; login-node dev env is `/work/yifanwang/gpu/env` (python 3.11).
- **GPU nodes run python 3.12, the login node 3.11 — a venv built on one does not resolve on the other** (its `bin/python` symlinks the node's system interpreter, so site-packages become invisible). Use `/work/yifanwang/gpu/env-gpu` (python 3.12, same editable installs + cupy-cuda12x) on GPU nodes.
- GPU nodes: `lakshmi` (8× A100-80GB, all in the condor pool) and `saraswati` (8× A100-40GB, only 1 in the condor pool). Condor route: `requirements = regexp("A100", CUDADeviceName)`, `request_gpus = 1`. Faster route when the pool is full: ssh to saraswati, pick an idle GPU from `nvidia-smi`, run with `CUDA_VISIBLE_DEVICES=<n>` (the 7 non-pool GPUs are fair game).
- GPU smoke test: `/work/yifanwang/gpu/gpu_validation/gpu_smoke_test.py` — validates CUPYScheme waveform + matched filter against CPU. Verified 2026-07: all pass on A100, waveform GPU==CPU to 6e-13, single-waveform speedup ×13.5.

## Conventions

- CPU hot loops are Cython (`*_cython.pyx`, `*.pyx`) compiled by `setup.py`; after editing `.pyx` files re-run `pip install -e .` (or `python setup.py build_ext --inplace`).
- `pycbc.HAVE_CUDA` in `pycbc/__init__.py` refers to the legacy PyCUDA stack, not CuPy. CuPy availability is checked by importing `cupy` inside `CUPYScheme.__init__`.
- Upstream contributions go to `gwastro/pycbc` `master`; keep changes scheme-dispatched (never import `cupy` at top level of frontend modules) so CPU-only installs are unaffected.
