all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf logs/pipeline_test

data/double_torus.ply:
	@mkdir data
	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply

.venv:
	@python -m venv .venv
	@source .venv/bin/activate
	@pip install -r requirements.txt

logs/pipeline_test/tesh.ply: data/double_torus.ply
	@python main.py --silent experiments/test_experiment.json

meshlab: logs/pipeline_test/tesh.ply
	@meshlab logs/pipeline_test/test.ply

.PHONY: all clean meshlab
