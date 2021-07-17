all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf logs/double_torus_toy

data/double_torus.ply:
	@mkdir data
	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply

.venv:
	@python -m venv .venv
	@source .venv/bin/activate
	@pip install -r requirements.txt

logs/double_torus_toy/final.ply: data/double_torus.ply
	@python main.py experiments/double_torus_toy.json

meshlab: logs/double_torus_toy/final.ply
	@meshlab logs/double_torus_toy/final.ply

.PHONY: all clean meshlab
