# Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks

![abstract](figures/abstract.png)

### Abstract

We present ***cmKAN***, a versatile framework for color matching. Given an input image with colors from a source color distribution, our method effectively and accurately maps these colors to match a target color distribution in both **supervised** and **unsupervised** settings. Our framework leverages the spline capabilities of Kolmogorov-Arnold Networks (KANs) to model the color matching between source and target distributions. Specifically, we developed a hypernetwork that generates spatially varying weight maps to control the nonlinear splines of a KAN, enabling accurate color matching. As part of this work, we introduce a large-scale dataset of paired images captured by two distinct cameras to evaluate our method’s efficacy in matching colors produced by different cameras. We evaluated our approach across various color-matching tasks, including: (1) **raw-to-raw mapping**, where the source color distribution is in one camera’s raw color space and the target in another camera’s raw space; (2) **raw-to-sRGB mapping**, where the source color distribution is in a camera’s raw space and the target is in the display sRGB space, emulating the color rendering of a camera ISP; and (3) **sRGB-to-sRGB mapping**, where the goal is to transfer colors from a source sRGB space (e.g., produced by a source camera ISP) to a target sRGB space (e.g., from a different camera ISP). The results demonstrate that our method achieves state-of-the-art performance across these tasks while remaining lightweight compared to other color matching and transfer methods.

## Installation and Requirements

Create a conda (or python) environment, clone the repository and install the required packages:

```bash
# 1. Create an environment

conda create -n cmKAN python=3.13 pip
conda activate cmKAN
# or
python -m venv .venv
source .venv/bin/activate

# 2. Clone the repository
git clone https://github.com/gosha20777/cmKAN.git
cd cmKAN

# 3. Install the required packages
pip install -r requirements.txt
```

## Dataset

We introduce a large-scale [Volga2K dataset](https://huggingface.co/datasets/gosha20777/volga2k), featuring over 2000 well-aligned images captured with a Huawei P40 Pro. This device was selected for its distinct Quad-Bayer RGGB and RYYB camera sensors, which employ different image processing techniques. These sensor variations in color handling, sensitivity, and tone-mapping create a significant domain gap. Spanning four years and multiple locations, the dataset is designed to effectively evaluate color-matching methods.

You can find more details about the Volga2K dataset on [😡 Hugging Face 😡 official page](https://huggingface.co/datasets/gosha20777/volga2k). To download Volga2K, please use the following command:

```bash
huggingface-cli download gosha20777/volga2k --repo-type dataset --local-dir data/
```

## Pre-trained Models

We provide pre-trained models for the following tasks:

| Dataset | Task | Training Config | Checkpoint | 
| ------- | ---- | ---------------- | ---------- |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [unsupervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [pair-based](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Adobe 5K](https://pan.baidu.com/share/init?surl=CsQRFsEPZCSjkT3Z1X_B1w) (password: `5fyk`)| sRGB-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Samsung2Iphone](https://github.com/mahmoudnafifi/raw2raw) | raw-to-raw | [unsupervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Zurich raw-to-sRGB](https://people.ee.ethz.ch/~ihnatova/pynet.html) | raw-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |



## How To Use

Our `cmKAN` provides a command-line interface (CLI) to interact with the following tools:

```bash
python main.py -h 

Usage: cmKAN CLI [-h] {data-create,test,train,predict,unit-test} ...

Options:
  -h, --help            Show this help message and exit

Tools:
  {data-create,test,train,predict,unit-test}
    
    data-create         Create dataset
    train               Train model
    test                Test model
    predict             Run model inference
    unit-test           Run unit tests
```

For all the tools, you can use the `-h` flag to get help on how to use them (e.g. `python main.py train -h`). Here are some examples on how to use the tools:

### Train

```bash
python main.py train -c configs/config.yaml
```

### Test

```bash
python main.py test -c configs/config.yaml -w checkpoint.ckpt
```

### Predict

```bash
python main.py predict -c configs/config.yaml -i path/to/input/folder -o path/to/output/folder
```

### Additional Guides

You can find additional guides how to reproduce all our experiments in [our wiki page](https://github.com/gosha20777/cmKAN/wiki). We provide detailed instructions on how to train and test our model, as well as how to use it for inference. 

## Citation

If you find our work useful, please consider citing it:

```bibtex
@article{perevozchikov2025color,
  title={Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks},
  author={Nikonorov, Artem and Perevozchikov, Georgy and Korepanov, Andrei and Mehta, Nancy and Afifi, Mahmoud and Ershov, Egor and Timofte, Radu},
  journal={arXiv preprint arXiv:2503.11781},
  year={2025}
}
```