# conda deactivate
# conda deactivate
# conda activate cv

course := '02504'
handin_file := './excercises/work.ipynb'
output_name := '${course}-s204131-exam'


all: webpdf html pdf

pdf:
	jupyter nbconvert --to pdf ${handin_file} --output '${output_name}.pdf' --LatexPreprocessor.title="${course} Exam"

webpdf:
	jupyter nbconvert --to webpdf --theme light --template classic ${handin_file} --output '${output_name}-webpdf'

html:
	jupyter nbconvert --to html --theme light --template classic ${handin_file} --output '${output_name}.html'




