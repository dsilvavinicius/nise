all: data/double_torus.ply
	@python main.py --silent experiments/test_experiment.json

data/double_torus.ply:
	@mkdir data
	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply 

clean:
	@rm -Rf __pycache__
	@rm -Rf logs/pipeline_test

.venv:
	@python -m venv .venv
	@source .venv/bin/activate
	@pip install -r requirements.txt

.PHONY: all clean
