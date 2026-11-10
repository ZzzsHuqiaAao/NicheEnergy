# Installation

## New Linux environment

Use Python 3.10 and an NVIDIA GPU/driver compatible with the [PyTorch 2.5.1 CUDA 12.1 build](https://pytorch.org/get-started/previous-versions/#v251). For example, create an environment with Conda:

```bash
conda create -n nicheenergy python=3.10 -y
conda activate nicheenergy
```

From the downloaded NicheEnergy repository, run:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r config.txt
python -m pip install --no-deps scvi-tools==0.16.1 .
```

These steps apply to both newer and older Linux servers. `--no-deps` keeps the tested scvi-tools 0.16.1 / Lightning 1.8.6 combination; scvi-tools declares Lightning <1.6, so `pip check` reports that version conflict. The toolchain and dependency versions are listed in `config.txt`.

## Notebook kernel (optional)

```bash
python -m pip install -c config.txt ipykernel==6.31.0 nbclient==0.10.4 nbformat==5.10.4
python -m ipykernel install --user --name nicheenergy --display-name "Python 3.10 (NicheEnergy)"
```
