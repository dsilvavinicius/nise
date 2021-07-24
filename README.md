# i4d
Morphing and animation of implicit surfaces.

## Getting Started

### Prerequisites

1. Anaconda
2. Git
3. MeshLab

If using Linux, or macOS, these programs should be available through the package manager, or Homebrew.

### Setup

1. Clone this repository: `git clone git@github.com:dsilvavinicius/i4d.git`
2. Enter project folder: `cd i4d`
3. Setup project dependencies via conda
```
conda env create -f environment.yml
conda activate i4d
```
4. Download the [Double Torus Mesh](https://drive.google.com/file/d/11PkscMHBUkkENhHfI1lpH5Dh6X9f2028/view?usp=sharing) into the `data` folder in the repository
5. Run the main script passing the pipeline test configuration file as input
```
python main.py experiments/double_torus_toy.json
```
6. Run MeshLab and open the resulting mesh file `logs/double_torus_toy/final.ply`
7. (Optional) Run tensorboard using the command below and access http://localhost:6006/ to see the training progress.
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
If you have any questions, please feel free to email the authors.
