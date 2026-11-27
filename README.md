# DAS-MIMO

**DAS-MIMO learns the structure of a MIMO detection algorithm directly from data.** It represents the detector as a directed acyclic graph of candidate algebraic operations. Gradient descent jointly optimizes which operations contribute to each update, which intermediate variables they take as inputs, and how their outputs are combined. The resulting detector can use a different update rule at each iteration and reuse estimates computed several iterations earlier.

The cost assigned to each operation during training controls the trade-off between decoding performance and detector complexity. A stronger penalty produces DAS-Sparse, which relies mainly on inexpensive vector operations. DAS-Dense retains more matrix operations to improve detection at high code rates. After training, the compiler simplifies each learned graph into explicit detector equations and generates fused CUDA kernels from those equations.

## Installation

The experiments below used Linux, Python 3.10.12, NumPy 1.26.4 and PyTorch 2.3.0 with CUDA 11.8. CUDA generation runs on a CPU. Compiling and timing kernels requires a CUDA-capable GPU, NVIDIA's CUDA toolkit with `nvcc` and cuBLASLt, and a C++17 host compiler. The throughput checks used an RTX 4090 and CUDA toolkit 13.2.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Run the commands below from the project root.

## Models and checkpoints

| Detector | Command-line name | Configuration or depth | Evaluation checkpoint |
|---|---|---|---|
| DAS-Sparse | `das_sparse` | [Sparse configuration](configs/das_sparse.json), rates 0.3–0.8 | [Weights](checkpoints/das_sparse.pt) |
| DAS-Dense | `das_dense` | [Dense configuration](configs/das_dense.json), rates 0.6–0.8 | [Weights](checkpoints/das_dense.pt) |
| ADMM-Net | `admm_net` | 10 layers | [Weights](checkpoints/admm_net.pt) |
| DetNet | `detnet` | L = 10, joint 16-QAM output | [Weights](checkpoints/detnet.pt) |
| ChannelNet | `channelnet` | 10 iterations, 10 features | [Weights](checkpoints/channelnet.pt) |
| MMSE | `mmse` | Linear MMSE | Analytical |
| AMP | `amp8`, `amp15` | 8 or 15 iterations, damping 0.5 | Analytical |
| EP | `ep3`, `ep7` | 3 or 7 iterations | Analytical |
| K-best | `kbest8` | List size 8 | Analytical |

Each learned detector is distributed as a PyTorch state dictionary. [DetNet's initialization](checkpoints/detnet_init.pt) contains the weights learned on noiseless observations before coded training. The evaluation checkpoint contains the weights after coded training.

### Baseline references

| Detector | Paper | Public implementation |
|---|---|---|
| ADMM-Net | Kim and Park, [Learnable MIMO Detection Networks Based on Inexact ADMM](https://ieeexplore.ieee.org/document/9210880), IEEE TWC, 2021 | [Authors’ repository](https://github.com/Minsik-K/ADMM-Net-for-MIMO-Detection) |
| DetNet | Samuel, Diskin, and Wiesel, [Learning to Detect](https://doi.org/10.1109/TSP.2019.2899805), IEEE TSP, 2019 | [Authors’ 16-QAM implementation](https://github.com/neevsamuel/LearningToDetect/blob/master/DetNet_HD_16QAM.py) |
| ChannelNet | Ye and Liang, [On Purely Data-Driven Massive MIMO Detectors](https://doi.org/10.1109/TSP.2025.3594197), IEEE TSP, 2025 | [Authors’ repository](https://github.com/UCSC-Edge-AI-Lab/ChannelNet) |
| Linear MMSE | McKay, Collings, and Tulino, [Achievable Sum Rate of MIMO MMSE Receivers: A General Analytic Framework](https://doi.org/10.1109/TIT.2009.2034893), IEEE TIT, 2010 | [IIP Group, unbiased MMSE](https://github.com/IIP-Group/massiveMIMOdetection/blob/main/detectors/MMSE.m) |
| AMP (LAMA) | Jeon, Ghods, Maleki, and Studer, [Optimality of Large MIMO Detection via Approximate Message Passing](https://doi.org/10.1109/ISIT.2015.7282651), IEEE ISIT, 2015 | [IIP Group, LAMA](https://github.com/IIP-Group/massiveMIMOdetection/blob/main/detectors/LAMA.m) |
| EP | Céspedes, Olmos, Sánchez-Fernández, and Pérez-Cruz, [Expectation Propagation Detection for High-Order High-Dimensional MIMO Systems](https://doi.org/10.1109/TCOMM.2014.2332349), IEEE TCOM, 2014 | [NVIDIA Sionna, EPDetector](https://github.com/NVlabs/sionna/blob/2f0a3820269b53fd022780a6f2f5f720f7ceb9b4/src/sionna/phy/mimo/detection.py#L1080) |
| K-best | Guo and Nilsson, [Algorithm and Implementation of the K-Best Sphere Decoding for MIMO Detection](https://doi.org/10.1109/JSAC.2005.862402), IEEE JSAC, 2006 | [IIP Group, K-best search](https://github.com/IIP-Group/massiveMIMOdetection/blob/main/detectors/KBEST.m) |

## Data and training

Each configuration generates 156,250 training codewords across 24 fixed code-rate and SNR pairs. Each codeword contains 1,024 coded bits and occupies 32 receive vectors with 16-QAM and eight transmit antennas. The generator stores the information bits, rate-matched bits, Rayleigh channels and noise. Training reuses these frames across epochs. The three learned baselines share the DAS-Sparse dataset.

```bash
python generate_data.py --config das_sparse
python generate_data.py --config das_dense
python train.py das_sparse
python train.py das_dense
python train.py admm_net
python train.py detnet
python train.py channelnet
```

DAS-Sparse and DAS-Dense use operator sparsity coefficients of `1e-3` and `1e-5`, respectively. The relative penalties on matrix operators are 300 and 30. Both models combine the LDPC decoder loss with an auxiliary loss on intermediate detector outputs. The weight of the auxiliary loss decays exponentially during training. Each epoch writes evaluation weights to `runs/<model>/<model>.pt` and the optimizer state with the detector to `runs/<model>/last.pt`. Use `--weights runs/<model>` in the evaluation or timing commands to select your own weights.

For a short training run on one GPU:

```bash
python generate_data.py --config das_sparse --tasks 10 22 --frames-per-task 4 --output data/example
python train.py das_sparse --data data/example --epochs 1 --batch 4 --microbatch 2 --output runs/example
```

`--batch` sets the number of codewords per GPU and optimizer step. `--microbatch` controls the number per backward pass. For a four-GPU run, use `torchrun --standalone --nproc-per-node=4 train.py das_sparse`. To continue a run, pass `--resume runs/das_sparse/last.pt` with the same dataset.

## Decoded BLER

The evaluation measures the fraction of codewords with at least one erroneous information bit after eight offset-min-sum iterations. All selected detectors receive the same information bits, channels and noise. The seed selects the evaluation frames independently of the model list and batch size.

```bash
python evaluate.py --models das_sparse das_dense admm_net detnet channelnet mmse amp8 amp15 ep3 ep7 kbest8 \
    --rate 0.8 --snr 10.1 --frames 4096 --seed 43001
python evaluate.py --models das_sparse das_dense --rate 0.6747796940963932 \
    --snr 9.566666666666666 --frames 8192 --seed 43002
```

`--snr` denotes the information-bit `Eb/N0` in dB. Use `--nr` and `--nt` to select the antenna dimensions and `--nmse` for channel-estimation NMSE in linear units. For example, `--models das_sparse das_dense channelnet --nr 24 --nt 8 --nmse 0` evaluates transfer to 24 receive antennas. With `--nr 16 --nt 8 --nmse 0.01`, the detector receives a channel estimate at −20 dB NMSE. A JSON report from `--output /path/to/result.json` contains the actual information length, block count and error count.

## CUDA generation and equivalent FLOPs

The compiler conditions the checkpoint on the selected code configuration, prunes operator weights below 0.01, removes dead expressions and shares common subexpressions. It emits a detector specialized for the paper's 16 × 8 grouped-channel layout. The standalone exporter, FLOP counter and throughput benchmark call the same compiler.

```bash
python generate_cuda.py das_sparse --task 10 --compile
python generate_cuda.py das_dense --task 10 --compile
python flops.py --task 10 --reuse 64
python flops.py --models das_sparse --task 22 --reuse 64
python flops.py --models das_dense --task 23 --reuse 64
```

Task 10 corresponds to rate 0.8 in both configurations. For the 691-bit point at rate approximately 0.675, the published complexity calculation uses Sparse task 22 and Dense task 23. Their conditioning rates are 0.674780 and 0.675708. The BLER calculation uses the same 691-bit payload for both detectors. The task IDs, rates and training SNRs are listed in the dataset's `manifest.json`.

The counter reports the per-vector detector cost plus the per-channel preprocessing cost divided by the reuse count. One fused multiply-add counts as two operations. Scalar comparisons and elementary functions each count as one equivalent operation. Each K-best compare-exchange counts as one comparison and four record moves. DetNet uses the unfolded-layer count reported in the paper. Its CUDA kernel folds the linear constellation readout into the trained affine map. K-best's symbol search is timed through symbol output. Its BLER path computes bit metrics over the survivor list.

## GPU throughput

```bash
python throughput.py --models das_sparse das_dense --tasks 10 --channels 128 --reuse 64 128 512 --check
python throughput.py --models admm_net detnet channelnet mmse amp8 amp15 ep3 ep7 kbest8 \
    --channels 128 --reuse 64 128 512 --check
python throughput.py --models das_sparse das_dense --all-tasks
```

Each timing starts with 128 raw channel matrices and covers channel preprocessing and detection of all receive vectors. The reuse count is the number of vectors per channel. The benchmark selects the fastest measured launch configuration, reports median CUDA-event time and converts it to millions of MIMO vectors per second. The paper reports the median across 24 code configurations for each DAS detector. `--check` also compares CUDA and PyTorch symbol estimates on matched inputs. Building a library for the first time incurs compilation time outside the measured interval.
