all:
	latexmk -shell-escape Main.tex -interaction=draftmode -pdf -bibtex -jobname=dissertation
	pdflatex -interaction=batchmode -jobname=dissertation --shell-escape Main.tex

# build a specific chapter
% : %.tex
	latexmk -shell-escape $< -interaction=draftmode -pdf -bibtex
	pdflatex -interaction=batchmode --shell-escape $<
