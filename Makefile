all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf logs/double_torus_toy
	@rm -Rf results/*

meancurv_armadillo: results/meancurvature_armadillo_ni/reconstructions/time_0.0.ply
	@echo "Mean Curvature Equation: Armadillo trained and reconstructed"

results/meancurvature_armadillo_ni/models/best.pth: experiments/meancurvature_armadillo_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_armadillo_ni/reconstructions/time_0.0.ply: results/meancurvature_armadillo_ni/models/best.pth
	@python tools/reconstruct.py $< $@ -r 256 -t -0.9 -0.5 0.0 0.5 0.9

meancurv_max: results/meancurvature_max_ni/reconstructions/time_0.0.ply
	@echo "Mean Curvature Equation: Max trained and reconstructed"

results/meancurvature_max_ni/models/best.pth: experiments/meancurvature_max_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_max_ni/reconstructions/time_0.0.ply: results/meancurvature_max_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

# data/double_torus.ply:
#	@mkdir data
#	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply

# logs/double_torus_toy/reconstructions/final.ply: data/double_torus.ply
#	@python main.py experiments/double_torus_toy.json

# meshlab: logs/double_torus_toy/reconstructions/final.ply
#	@meshlab logs/double_torus_toy/reconstructions/final.ply

conda-env:
	@conda env create -f environment.yml

.PHONY: all clean conda-env meancurv_armadillo meancurv_max
