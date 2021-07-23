all: meshlab

clean:
	@rm -Rf __pycache__
	@rm -Rf logs/double_torus_toy

data/double_torus.ply:
	@mkdir data
	@curl -s -o data/double_torus.ply http://www.inf.puc-rio.br/~gschardong/i4d/double_torus.ply

logs/double_torus_toy/reconstructions/final.ply: data/double_torus.ply .venv
	@python main.py experiments/double_torus_toy.json

meshlab: logs/double_torus_toy/reconstructions/final.ply
	@meshlab logs/double_torus_toy/reconstructions/final.ply

.PHONY: all clean meshlab
