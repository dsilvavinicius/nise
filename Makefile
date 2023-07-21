all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf results/*

meancurv_all: meancurv_armadillo meancurv_bob meancurv_falcon meancurv_max meancurv_neptune meancurv_spot meancurv_witch

meancurv_armadillo: results/meancurvature_armadillo_ni/reconstructions/time_0.0.ply
	@echo "Mean Curvature Equation: Armadillo trained and reconstructed"

results/meancurvature_armadillo_ni/models/best.pth: experiments/meancurvature_armadillo_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_armadillo_ni/reconstructions/time_0.0.ply: results/meancurvature_armadillo_ni/models/best.pth
	@python tools/reconstruct.py $< $@ -r 256 -t -0.9 -0.5 0.0 0.5 0.9

meancurv_bob: results/meancurvature_bob_ni/reconstructions/
	@echo "Mean Curvature Equation: Bob trained and reconstructed"

results/meancurvature_bob_ni/models/best.pth: experiments/meancurvature_bob_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_bob_ni/reconstructions/: results/meancurvature_bob_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_falcon: results/meancurvature_falcon_ni/reconstructions/
	@echo "Mean Curvature Equation: Falcon trained and reconstructed"

results/meancurvature_falcon_ni/models/best.pth: experiments/meancurvature_falcon_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_falcon_ni/reconstructions/: results/meancurvature_falcon_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.05 0.0 0.5 0.9

meancurv_max: results/meancurvature_max_ni/reconstructions/
	@echo "Mean Curvature Equation: Max trained and reconstructed"

results/meancurvature_max_ni/models/best.pth: experiments/meancurvature_max_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_max_ni/reconstructions/: results/meancurvature_max_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_neptune: results/meancurvature_neptune_ni/reconstructions/
	@echo "Mean Curvature Equation: Neptune trained and reconstructed"

results/meancurvature_neptune_ni/models/best.pth: experiments/meancurvature_neptune_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_neptune_ni/reconstructions/: results/meancurvature_neptune_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t 0.0 0.5 0.9

meancurv_spot: results/meancurvature_spot_ni/reconstructions/
	@echo "Mean Curvature Equation: Spot trained and reconstructed"

results/meancurvature_spot_ni/models/best.pth: experiments/meancurvature_spot_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_spot_ni/reconstructions/: results/meancurvature_spot_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_witch: results/meancurvature_witch_ni/reconstructions/
	@echo "Mean Curvature Equation: Witch trained and reconstructed"

results/meancurvature_witch_ni/models/best.pth: experiments/meancurvature_witch_ni.yaml
	@python experiment_scripts/train_meancurvature.py $<

results/meancurvature_witch_ni/reconstructions/: results/meancurvature_witch_ni/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.9 0.5 0.0 0.5 0.9

# data/double_torus.ply:
#	@mkdir data
#	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply

# logs/double_torus_toy/reconstructions/final.ply: data/double_torus.ply
#	@python main.py experiments/double_torus_toy.json

# meshlab: logs/double_torus_toy/reconstructions/final.ply
#	@meshlab logs/double_torus_toy/reconstructions/final.ply

conda-env:
	@conda env create -f environment.yml

.PHONY: all clean conda-env meancurv_all meancurv_armadillo meancurv_bob meancurv_falcon meancurv_max meancurv_neptune meancurv_spot meancurv_witch
