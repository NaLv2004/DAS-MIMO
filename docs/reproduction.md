# Reproducing the paper experiments

Run these commands from the repository root. The paper uses 16 receive antennas, eight transmit antennas, 16-QAM and rate-matched 5G LDPC codewords of length 1,024. Each codeword occupies 32 MIMO vectors.

## Environment

The reference environment uses Linux, Python 3.10, PyTorch 2.3.0 and NumPy 1.26.4. To use the CUDA 11.8 build of PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

CUDA source export runs on a CPU. To compile and time the kernels, install a CUDA toolkit with `nvcc`, cuBLASLt and a C++17 host compiler. The RTX 4090 benchmark was also run with toolkit 13.2. For a toolkit installed under `/usr/local/cuda`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Train a detector

```bash
python generate_data.py --config das_sparse
python train.py das_sparse

python generate_data.py --config das_dense
python train.py das_dense
```

Each configuration generates 156,250 training codewords across 24 fixed rate and SNR pairs. The [Sparse configuration](../configs/das_sparse.json) spans rates 0.3–0.8 and the [Dense configuration](../configs/das_dense.json) spans 0.6–0.8. Their operator sparsity coefficients are `1e-3` and `1e-5`, with relative penalties of 300 and 30 on matrix operations.

ADMM-Net, DetNet and ChannelNet use the DAS-Sparse training data:

```bash
python train.py admm_net
python train.py detnet
python train.py channelnet
```

DetNet starts from the [noiseless initialization](../checkpoints/detnet_init.pt). Each epoch saves evaluation weights to `runs/<model>/<model>.pt` and a training checkpoint to `runs/<model>/last.pt`. Use `python train.py --help` for batch size, data directory and resume options.

## Evaluate BLER

```bash
python evaluate.py --models das_sparse das_dense admm_net detnet channelnet mmse amp8 amp15 ep3 ep7 kbest8 \
    --rate 0.8 --snr 10.1 --frames 4096
python evaluate.py --models das_sparse das_dense --rate 0.6747796940963932 \
    --snr 9.566666666666666 --frames 8192
```

BLER is the fraction of codewords with an erroneous information bit after LDPC decoding. The selected detectors use the same evaluation frames. The `--snr` argument is information-bit $E_b/N_0$ in dB. Set `--nr` and `--nt` for a different antenna configuration, or `--nmse 0.01` for channel estimates at −20 dB NMSE. To evaluate newly trained weights, pass `--models das_sparse --weights runs/das_sparse`.

## Export CUDA and count FLOPs

```bash
python generate_cuda.py das_sparse --task 10 --compile
python generate_cuda.py das_dense --task 10 --compile
python flops.py --task 10 --reuse 64
```

The compiler selects the graph for the specified task, prunes operator weights below 0.01 and emits CUDA source in `build/`. Omit `--compile` to export source alone. To export your own checkpoint, add `--weights runs/das_sparse/das_sparse.pt` to the corresponding generator command.

Task 10 corresponds to rate 0.8 for both models. The paper's comparison at rate 0.675 uses Sparse task 22 and Dense task 23, with conditioning rates of 0.674780 and 0.675708, respectively. For the corresponding FLOP counts:

```bash
python flops.py --models das_sparse --task 22 --reuse 64
python flops.py --models das_dense --task 23 --reuse 64
```

Equivalent FLOPs per vector equal the detector cost per vector plus the channel preprocessing cost divided by the reuse count. A fused multiply-add counts as two operations. Scalar comparisons and elementary functions each count as one equivalent operation. A K-best compare-exchange counts as one comparison and four record moves. DetNet uses the unfolded-layer count reported in the paper.

## Measure GPU throughput

```bash
python throughput.py --models das_sparse das_dense --all-tasks --channels 128 --reuse 64 128 512
python throughput.py --models admm_net detnet channelnet mmse amp8 amp15 ep3 ep7 kbest8 \
    --channels 128 --reuse 64 128 512 --check
```

The benchmark calls the same compiler as `generate_cuda.py`. Each measurement covers channel preprocessing and detection of all receive vectors. The reuse count is the number of vectors per channel matrix. The paper reports the median across 24 code configurations for each DAS detector. To time one configuration, replace `--all-tasks` with `--tasks 10`. The optional `--check` compares CUDA and PyTorch symbol estimates before timing.
