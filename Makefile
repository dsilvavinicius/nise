all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf results/*

meancurv_all: meancurv_armadillo meancurv_bob meancurv_falcon meancurv_max meancurv_neptune meancurv_spot meancurv_witch

meancurv_armadillo: results/meancurvature_armadillo/reconstructions/time_0.0.ply
	@echo "Mean Curvature Equation: Armadillo trained and reconstructed"

results/meancurvature_armadillo/models/best.pth: experiments/meancurvature_armadillo.yaml
	@python meancurvature-train.py $<

results/meancurvature_armadillo/reconstructions/time_0.0.ply: results/meancurvature_armadillo/models/best.pth
	@python tools/reconstruct.py $< $@ -r 256 -t -0.9 -0.5 0.0 0.5 0.9

meancurv_bob: results/meancurvature_bob/reconstructions/
	@echo "Mean Curvature Equation: Bob trained and reconstructed"

results/meancurvature_bob/models/best.pth: experiments/meancurvature_bob.yaml
	@python meancurvature-train.py $<

results/meancurvature_bob/reconstructions/: results/meancurvature_bob/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_falcon: results/meancurvature_falcon/reconstructions/
	@echo "Mean Curvature Equation: Falcon trained and reconstructed"

results/meancurvature_falcon/models/best.pth: experiments/meancurvature_falcon.yaml
	@python meancurvature-train.py $<

results/meancurvature_falcon/reconstructions/: results/meancurvature_falcon/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.05 0.0 0.5 0.9

meancurv_max: results/meancurvature_max/reconstructions/
	@echo "Mean Curvature Equation: Max trained and reconstructed"

results/meancurvature_max/models/best.pth: experiments/meancurvature_max.yaml
	@python meancurvature-train.py $<

results/meancurvature_max/reconstructions/: results/meancurvature_max/models/best.pth
	python tools/reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_neptune: results/meancurvature_neptune/reconstructions/
	@echo "Mean Curvature Equation: Neptune trained and reconstructed"

results/meancurvature_neptune/models/best.pth: experiments/meancurvature_neptune.yaml
	@python meancurvature-train.py $<

results/meancurvature_neptune/reconstructions/: results/meancurvature_neptune/models/best.pth
	python reconstruct.py $< $@ -r 256 -t 0.0 0.5 0.9

meancurv_spot: results/meancurvature_spot/reconstructions/
	@echo "Mean Curvature Equation: Spot trained and reconstructed"

results/meancurvature_spot/models/best.pth: experiments/meancurvature_spot.yaml
	@python meancurvature-train.py $<

results/meancurvature_spot/reconstructions/: results/meancurvature_spot/models/best.pth
	python reconstruct.py $< $@ -r 256 -t -0.4 0.0 0.5 0.9

meancurv_witch: results/meancurvature_witch/reconstructions/
	@echo "Mean Curvature Equation: Witch trained and reconstructed"

results/meancurvature_witch/models/best.pth: experiments/meancurvature_witch.yaml
	@python meancurvature-train.py $<

results/meancurvature_witch/reconstructions/: results/meancurvature_witch/models/best.pth
	python reconstruct.py $< $@ -r 256 -t -0.9 0.5 0.0 0.5 0.9

morph_spot-bob: results/morph_spot-bob/reconstructions/
	@echo "Morph: Morphing of spot=>bob trained and reconstructed"

results/morph_spot-bob/models/best.pth: experiments/morph_spot-bob.yaml
	@python morph-train.py $<

results/morph_spot-bob/reconstructions/: results/morph_spot-bob/models/best.pth
	python reconstruct.py $< $@ -r 256 -t -0.9 0.5 0.0 0.5 0.9

conda-env:
	@conda env create -f environment.yml

.PHONY: all clean conda-env meancurv_all meancurv_armadillo meancurv_bob meancurv_falcon meancurv_max meancurv_neptune meancurv_spot meancurv_witch
