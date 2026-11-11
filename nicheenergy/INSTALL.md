# Installation

NicheEnergy requires Python 3.10. A separate environment is recommended:

```bash
conda create -n nicheenergy python=3.10 -y
conda activate nicheenergy
pip install nicheenergy
```

## NVIDIA GPU

The MERFISH tutorial uses an NVIDIA GPU. For the tested PyTorch 2.5.1 / CUDA 12.1 build, install PyTorch first:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install nicheenergy
```

The model can also run on CPU. Choose a PyTorch build compatible with your system and driver; see the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

## From source

From the downloaded NicheEnergy repository:

```bash
python -m pip install .
```

The package uses scvi-tools 0.20.3 and PyTorch Lightning 1.9.5. The pinned validation environment is listed in `config.txt`.

## Notebook kernel (optional)

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name nicheenergy --display-name "Python 3.10 (NicheEnergy)"
```
