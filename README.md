# CompStruct
This repository records code used in the study [_**What Information is Necessary and Sufficient to Predict Materials Properties using Machine Learning?**_](https://arxiv.org/abs/2206.04968)
This repository is based on four materials property prediction machine learning (ML) framework repositories, [MEGNet](https://github.com/materialsvirtuallab/megnet), [CGCNN](https://github.com/txie-93/cgcnn), [CrabNet](https://github.com/anthony-wang/CrabNet), and [Roost](https://github.com/CompRhys/roost).


# Table of Contents
- [CompStruct](#CompStruct)
- [How to Cite](#how-to-cite)
- [Installation](#installation)
  - [Ways to create virtual enviroments with Anaconda](#ways-to-create-virtual-enviroments-with-anaconda1)
  - [MEGNet](#megnet)
  - [CGCNN](#cgcnn)
  - [CrabNet](#crabnet)
  - [Roost](#roost)
- [Usage](#usage)
  - []()
  - []()
- [Authors](#authors)

# How to Cite
Please cite the following work if you want to use CompStruct
```
@misc{https://doi.org/10.48550/arxiv.2206.04968,
  doi = {10.48550/ARXIV.2206.04968},
  url = {https://arxiv.org/abs/2206.04968},
  author = {Tian, Siyu Isaac Parker and Walsh, Aron and Ren, Zekun and Li, Qianxiao and Buonassisi, Tonio},
  keywords = {Materials Science (cond-mat.mtrl-sci), Computational Physics (physics.comp-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {What Information is Necessary and Sufficient to Predict Materials Properties using Machine Learning?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# Installation
We recommend users to build separate virtual environments for every ML framework to avoid package conflicts. We give our installation instructions based on the ones in respective framework repositories. We first introduce [ways to create virtual environments](#ways-to-create-virtual-enviroments-with-anaconda1) and then give respective intructions for installing each framework.

## Ways to create virtual enviroments with Anaconda[^1]
1. Download and install [Anaconda](https://conda.io/docs/index.html)
2. Navigate to the project directory (from above).
3. Open Anaconda prompt in this directory.
4. If there is a conda environment `.yml` file, run the following command from Anaconda prompt to automatically create an environment and install packages needed from the `*.yml` file:
    - `conda env create --file *.yml` (environment name defined in the `*.yml` file)
  
    Else, run the following command to create a new conda environment with specific python version 
    first to install packages later.
    - `conda create -n your_env_name python=3.x` 
  
    where specific environement name and python version should be inputted, *e.g.*, `conda env create megnet python=3.6`.
5. Run one of the following commands from Anaconda prompt depending on your operating system to activate the environment:
    - `conda activate crabnet`
    - `source activate crabnet`

    This environment upon activation can be used for installing required packages and then running code.

For more information about creating, managing, and working with Conda environments, please consult the [relevant help page](https://conda.io/docs/user-guide/tasks/manage-environments.html).

**Before proceeding further to installation of any following packages, clone or download the CompStruct repository, and navigate to the CompStruct repository directory in your Anaconda prompt.**

## MEGNet
1. Build a python 3.7 virtual environment
2. Once the environment is activated, run
   - `pip install -r requirements_megnet.txt`

*Avoid using `pip install megnet` as instructed by [MEGNet installation](https://github.com/materialsvirtuallab/megnet#installation) because we have modified the MEGNet source code and included the modified code in this repository. Our modified code is based on `MEGNet` version 1.2.8*

## CGCNN
1. Build a python 3.6 virtual environment
1. Once the environment is activated, run
   - `pip install -r requirements_cgcnn.txt`
  
*Included CGCNN code is modified based on the [CGCNN repository](https://github.com/txie-93/cgcnn) on Dec 3, 2021.*

## CrabNet
*First Option*

1. Run
    - `conda env create --file conda-env_crabnet.yml`
    - `conda env create --file conda-env-cpuonly_crabnet.yml` if you only have a CPU and no GPU in your system

    This creates a virtual environment named `crabnet` and also installs required packages.
2. Activate the built environment by running one of the following commands from Anaconda prompt depending on your operating system
   - `conda activate crabnet`
   - `source activate crabnet`

*Second Option*
1. Build a python 3.8 virtual environment
2. Once the environment is activated, open `conda-env_crabnet.yml` and `pip install` all the packages listed there.

### IMPORTANT - if you want to reproduce the publication Figures 1 and 2 in CrabNet:[^2]
The PyTorch-builtin function for outting the multi-headed attention operation defaults to averaging the attention matrix across all heads.
Thus, in order to obtain the per-head attention information, we have to edit a bit of PyTorch's source code so that the individual attention matrices are returned.

To properly export the attention heads from the PyTorch `nn.MultiheadAttention` implementation within the transformer encoder layer, you will need to manually modify some of the source code of the PyTorch library.
This applies to PyTorch v1.6.0, v1.7.0, and v1.7.1 (potentially to other untested versions as well).

For this, open the file:
`C:\Users\{USERNAME}\Anaconda3\envs\{ENVIRONMENT}\Lib\site-packages\torch\nn\functional.py`
(where `USERNAME` is your Windows user name and `ENVIRONMENT` is your conda environment name (if you followed the steps above, then it should be `crabnet`))

At the end of the function defition of `multi_head_attention_forward` (line numbers may differ slightly):
```python
L4011 def multi_head_attention_forward(
# ...
# ... [some lines omitted]
# ...
L4291    if need_weights:
L4292        # average attention weights over heads
L4293        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
L4294        return attn_output, attn_output_weights.sum(dim=1) / num_heads
L4295    else:
L4296        return attn_output, None
```

Change the specific line
```python
return attn_output, attn_output_weights.sum(dim=1) / num_heads
```

to:
```python
return attn_output, attn_output_weights
```

This prevents the returning of the attention values as an average value over all heads, and instead returns each head's attention matrix individually.
For more information see:
- https://github.com/pytorch/pytorch/issues/34537
- https://github.com/pytorch/pytorch/issues/32590
- https://discuss.pytorch.org/t/getting-nn-multiheadattention-attention-weights-for-each-head/72195/

*Included CrabNet code is modified based on the [CrabNet repository](https://github.com/anthony-wang/CrabNet) on Dec 3, 2021.*


## Roost
*First Option*

1. Run
    - `conda env create --file conda-env_roost.yml`

    This creates a virtual environment named `roost` and also installs required packages.
2. Activate the built environment by running one of the following commands from Anaconda prompt depending on your operating system
   - `conda activate roost`
   - `source activate roost`

    If you are not using `cudatoolkit` version 11.1 or do not have access to a GPU this setup will not work for you. If so please check the following pages [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch-Scatter](https://github.com/rusty1s/pytorch_scatter#installation) for how to install the core packages and then install the remaining requirements as detailed in requirements_roost.txt.

*Second Option*
1. Build a python 3.8 virtual environment
2. Once the environment is activated, run
   - `pip install -r requirements_roost.txt --find-links https://data.pyg.org/whl/torch-1.9.0+cu111.html`

    The extra `--find-links` is to cater for the installation of [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter#installation). This link is for `pytorch` version 1.9.0 and `cudatoolkit` version 11.1 as default in the [Roost installation](https://github.com/CompRhys/roost#environment-setup). If you do not have access to a GPU or are not using `cudatoolkit` version 11.1, you can refer to [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter#installation) to change the `cu111` to `cpu` (CPU only), or `cu102` (`cudatookit` version 10.2) etc. in the `--find-links` link.

*Included Roost code is modified based on the [Roost repository](https://github.com/CompRhys/roost) on Dec 3, 2021.*

# Usage
To get results as shown in Figure 3, S1, S2 and S3 for various datasets, run main results. Before running each main result, activate the respective environment and navigate to CompStruct repository directory.
## Run main results
<table>
    <thead>
        <tr>
            <th>Frameworks</th>
            <th>Run Options</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>MEGNet</td>
            <td rowspan=1>Run <code>python main_megnet.py</code> using default parameters</td>
        </tr>
        <tr>
            <td rowspan=1>Run <code>python main_megnet.py --help</code> to get parameters for toggling</td>
        </tr>
        <tr>
            <td rowspan=2>CGCNN</td>
            <td rowspan=1>Run <code>python main_cgcnn.py</code> using default parameters</td>
        </tr>
        <tr>
            <td rowspan=1>Run <code>python main_cgcnn.py --help</code> to get parameters for toggling</td>
        </tr>
        <tr>
            <td rowspan=2>Roost</td>
            <td rowspan=1>Run <code>python main_roost.py --train --evaluate</code> using default parameters</td>
        </tr>
        <tr>
            <td rowspan=1>Run <code>python main_roost.py --help</code> to get parameters for toggling</td>
        </tr>
        <tr>
            <td rowspan=2>CrabNet</td>
            <td rowspan=1>Run <code>python main_crabnet.py</code> using default parameters</td>
        </tr>
        <tr>
            <td rowspan=1>Run <code>python main_crabnet.py --help</code> to get parameters for toggling</td>
        </tr>
    </tbody>
</table>

## Reproduce publications figures

Run `.py` files for respective figures in [publication figures](./publication%20figures/).
1. Navigate to publication figures folder
2. Run `python figure_3.py` to generate Figure 3, and other scripts to generate other figures.

## Included scripts and folders:
<table>
    <thead>
        <tr>
            <th>Scripts</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1><code>main_megnet.py</code></td>
            <td rowspan=4>Main files to run for training various ML frameworks</td>
        </tr>
        <tr>
            <td rowspan=1><code>main_cgcnn.py</code></td>
        </tr>
        <tr>
            <td rowspan=1><code>main_roost.py</code></td>
        </tr>
        <tr>
            <td rowspan=1><code>main_crabnet.py</code></td>
        </tr>
        <tr>
            <td rowspan=1>requirements_megnet.txt</td>
            <td rowspan=1>Installation file for MEGNet</td>
        </tr>
        <tr>
            <td rowspan=1>requirements_cgcnn.txt</td>
            <td rowspan=1>Installation file for CGCNN</td>
        </tr>
        <tr>
            <td rowspan=1>requirements_roost.txt</td>
            <td rowspan=2>Installation files for Roost</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env_roost.yml</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env_crabnet.yml</td>
            <td rowspan=2>Installation files for CrabNet</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env-cpuonly_crabnet.yml</td>
        </tr>
    </tbody>
</table>
<table>
    <thead>
        <tr>
            <th>Folders</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1><a ref='https://github.com/PV-Lab/CompStruct/main/main_megnet.py'>main_megnet.py</a></td>
            <td rowspan=4>Main files to run for training various ML frameworks</td>
        </tr>
        <tr>
            <td rowspan=1><code>main_cgcnn.py</code></td>
        </tr>
        <tr>
            <td rowspan=1><code>main_roost.py</code></td>
        </tr>
        <tr>
            <td rowspan=1><code>main_crabnet.py</code></td>
        </tr>
        <tr>
            <td rowspan=1>requirements_megnet.txt</td>
            <td rowspan=1>Installation file for MEGNet</td>
        </tr>
        <tr>
            <td rowspan=1>requirements_cgcnn.txt</td>
            <td rowspan=1>Installation file for CGCNN</td>
        </tr>
        <tr>
            <td rowspan=1>requirements_roost.txt</td>
            <td rowspan=2>Installation files for Roost</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env_roost.yml</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env_crabnet.yml</td>
            <td rowspan=2>Installation files for CrabNet</td>
        </tr>
        <tr>
            <td rowspan=1>conda-env-cpuonly_crabnet.yml</td>
        </tr>
    </tbody>
</table>

| Folders | Description |
| ------------- | ------------------------------ |
| [data](./data)  | hosts saved datasets; the [utils](./data/utils/) folder within also contains scripts for Tauc-Lorentz oscillator `TaucLorentz.py` and transfer-matrix method `ScatteringMatrix.py`. |
| [pre-trained models](./pre-trained%20models/) | hosts pre-trained models; pre-trained models outputted by a running of `pre-training.py` will replace the current saved pre-trained models.

# Authors
The code was primarily written by Siyu Isaac Parker Tian, under the supervision of Tonio Buonassisi and Qianxiao Li.

[^1]: explanations of this section borrow from [CrabNet Installation](https://github.com/anthony-wang/CrabNet/blob/master/README_CrabNet.md#install-dependencies-via-anaconda)
[^2]: based on [CrabNet Installation Figure Reproduction](https://github.com/anthony-wang/CrabNet/blob/master/README_CrabNet.md#important---if-you-want-to-reproduce-the-publication-figures-1-and-2)
