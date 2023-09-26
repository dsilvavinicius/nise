# Neural Implicit Surface Evolution
[Tiago Novello [1]](https://sites.google.com/site/tiagonovellodebrito),
[Vinícius da Silva [2]](https://dsilvavinicius.github.io/),
[Guilherme Schardong [3]](https://schardong.github.io/),
[Luiz Schirmer [4]](https://www.lschirmer.com),
[Hélio Lopes [2]](http://www-di.inf.puc-rio.br/~lopes/),
[Luiz Velho [1]](https://lvelho.impa.br/)
<br>
[1] Institute for Pure and Applied Mathematics (IMPA),
[2] Pontifical Catholic University of Rio de Janeiro (PUC-Rio),
[3] University of Coimbra (UC)
[4] University of the Sinos Valley (UNISINOS)

This is the official implementation of "Neural Implicit Surface Evolution", published on [arXiv](https://arxiv.org/abs/2201.09636) and to appear at the Proceedings of ICCV 2023.

![Smoothing of the Armadillo - Curvature rendering](figs/smoothing-arm-curvatures.png)

## Getting started

### Prerequisites

1. [Anaconda](https://www.anaconda.com/products/individual#Downloads), alternativelly you can use [PyEnv](https://github.com/pyenv/pyenv) and [PyEnv-VirtualEnv](https://github.com/pyenv/pyenv-virtualenv) on UNIX based/like systems
2. [Git](https://git-scm.com/download)
3. [Integrate Git Bash with conda](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) (If on Windows)
4. [MeshLab](https://www.meshlab.net/)
5. [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin/)

If using Linux, or macOS, these programs should be available through the package manager, or Homebrew.

### Code organization
The common code is contained in the `nise` package. Inside the respective folder, there are the following files:
* `dataset.py` - contains the sampling and data classes
* `diff_operators.py` - implementation of differential operators (gradient, hessian, jacobian, curvatures)
* `loss.py` - contains loss functions for different experimental settings
* `meshing.py` - mesh creation through marching cubes
* `model.py` - networks and layers implementations
* `util.py` - miscelaneous functions and utilities

Additionally, under the `experiment_scripts` folder, there are more scripts with experiments and other auxiliary code that is generally independent of the main code.
* `discrete_smoothing.py` - experiments with discrete smoothing methods (lapliacian, cotangent, etc.)
* `enlarge_networks.py` - studying the effects of increasing network width in the training results (see Sec. 4.4 of the supplementary material)
* `lipschitz_exp.py` - experiments with lipschitz regularization
* `mean_curvature_scale.py` - experiments with different values for the mean curvature equation scale parameter (see Sec. 4.2 of the supplementary material)
* `point_sample_proportions.py` - experiments with the proportions of points drawn from the surface, off-surface and along time (see Sec. 4.3 of the supplementary material)
* `smoothing.py` - smoothing and sharpening experiments (see Sec. 6.2 in the paper)
* `training_time_intervals.py` - studies with varying time-intervals for training (see Sec. 4.1 of the supplementary material)

The main training and reconstruction scripts are in the repository's root folder:
* `meancurvature-train.py` - train a smoothing/sharpening of a single neural implicit surface (see Sec 6.2 in the paper)
* `morph-train.py` - train an interpolation between two neural implicit surfaces (see Sec 6.3 in the paper)
* `reconstruct.py` - given a trained model (pth) reconstructs the mesh using marching cubes at values `t` given by the user
* `vectorfield-train.py` - train a neural-based deformation of a neural implicit surface (see Sec 6.1 in the paper)

### Setup and sample run

1. Open a terminal (or Git Bash if using Windows)
2. Clone this repository: `git clone git@github.com:dsilvavinicius/nise.git`
3. Enter project folder: `cd nise`
4. Setup project dependencies via conda
```
conda env create -f environment.yml
conda activate nise
pip install -e .
```
or, if using pyenv (with pyenv-virtualenv):
```
pyenv virtualenv 3.9.9 nise
pyenv local nise
pip install -r requirements.txt
pip install -e .
```
5. Download the [pretrained neural implicit objects](https://drive.google.com/file/d/11PkscMHBUkkENhHfI1lpH5Dh6X9f2028/view?usp=sharing) into the `ni` folder in the repository
6. Download the [meshes]() into the `data` folder in the repository
7. Run the desired script passing the pipeline test configuration file as input
```
python meancurvature-train.py experiments/meancurvature_bunny.yaml
```
7. (Optional) Run tensorboard using the command below and access http://localhost:6006/ to see the training progress
```
tensorboard --logdir results/meancurvature_bunny/summaries
```
8. Run the reconstruction script to convert the output model to a series of meshes
```
python reconstruct.py results/meancurvature_bunny/models/best.pth results/meancurvature_bunny/reconstructions/ -t -0.2 0.0 0.2
```
9. Run MeshLab and open one the resulting mesh files `results/meancurvature_bunny/reconstructions/time_-0.2.ply`

<!-- Alternatively, on Linux and macOS systems, steps 3 (except the `activate` command) through 6 are implemented on the `Makefile` at the root of the project. -->

### End Result
If everything works, MeshLab should show the following image (or an image similar to it):


## Citation
If you find our work useful in your research, please cite:
```
@article{novello2022neural,
	title = {Neural Implicit Surface Evolution},
	author = {Novello, Tiago and da Silva, Vin\'icius and Schardong, Guilherme and Schirmer,
		Luiz and Lopes, H\'elio and Velho, Luiz},
	booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
	year = {2023}
}
```

## Contact
If you have any questions, please feel free to email the authors, or open an issue.
