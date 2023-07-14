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

This is the official implementation of "Neural Implicit Surface Evolution".

## Getting started

### Prerequisites

1. [Anaconda](https://www.anaconda.com/products/individual#Downloads), alternativelly you can use [PyEnv](https://github.com/pyenv/pyenv) and [PyEnv-VirtualEnv](https://github.com/pyenv/pyenv-virtualenv) on UNIX based/like systems
2. [Git](https://git-scm.com/download)
3. [Integrate Git Bash with conda](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) (If on Windows)
4. [MeshLab](https://www.meshlab.net/)
5. [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin/)

### Code organization
* `dataset.py` - contains the sampling and data classes
* `diff_operators.py` - implementation of differential operators (gradient, hessian, jacobian, curvatures)
* `loss.py` - contains loss functions for different experimental settings
* `main.py` - main function and point-of-entry to our code
* `meshing.py` - mesh creation through marching cubes
* `model.py` - networks and layers implementations
* `util.py` - miscelaneous functions and utilities

Additionally, under the `experiment_scripts` folder, there are more scripts with experiments and other auxiliary code that is generally independent of the main code.

* `comparison_analytic.py` - comparison experiments of RBF, SIREN and our approach for analytic models of a sphere and torus
* `comparison_ply.py` - comparison experiments of RBF, SIREN and our approach for PLY models
* `sdf_for_n_iters.py` - Experiments with intermitent querying of off-surface points. Both to speed-up the training and test the influence of generating the off-surface points at each iteration.

In the `tools` folder, there are two scripts:

* `estimate_mesh_curvatures.py`: Given a trained model (pth) and the original mesh, we estimate the vertex curvatures using the trained model.
* `reconstruct.py`: Given a trained model (pth) reconstructs the mesh using marching cubes.

If using Linux, or macOS, these programs should be available through the package manager, or Homebrew.

### Setup and sample run

1. Open a terminal (or Git Bash if using Windows)
2. Clone this repository: `git clone git@github.com:dsilvavinicius/i4d.git`
3. Enter project folder: `cd i4d`
4. Setup project dependencies via conda
```
conda env create -f environment.yml
conda activate i4d
pip install -e .
```
or, if using pyenv (with pyenv-virtualenv):
```
pyenv virtualenv 3.9.9 i4d
pyenv local i4d
pip install -r requirements.txt
pip install -e .
```
5. Download the [Double Torus Mesh](https://drive.google.com/file/d/11PkscMHBUkkENhHfI1lpH5Dh6X9f2028/view?usp=sharing) into the `data` folder in the repository
6. Run the main script passing the pipeline test configuration file as input
```
python main.py experiments/double_torus_toy.json
```
7. Run MeshLab and open the resulting mesh file `logs/double_torus_toy/final.ply`
8. (Optional) Run tensorboard using the command below and access http://localhost:6006/ to see the training progress.
```
tensorboard --logdir logs/double_torus_toy
```

Alternatively, on Linux and macOS systems, steps 3 (except the `activate` command) through 6 are implemented on the `Makefile` at the root of the project.

## Citation
If you find our work useful in your research, please cite:
```
Bibtex here!
```

## Contact
If you have any questions, please feel free to email the authors, or open an issue.
