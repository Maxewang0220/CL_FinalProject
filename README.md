# README

## Directory structure

.
│ corpus_reader.py
│ exhibit.py
│ model.py
│ README.md
│ test.py
│ test.log
│ train.py
│ requirements.txt

## Versions

Python: 3.9
torch: 2.6.0
datasets: 3.1.0
transformers: 4.24.0

## Instructions for running

A pre-trained `BERT_BASE` model is provided as an example.

To download the model, you can visit this
link: https://drive.google.com/file/d/1U6B_4gp8RJ4-G1uuKcoUr3UEtzv5fpKp/view?usp=sharing

To train the model, you should initialize the corresponding model and modify is_CRF attribute and the model path in
`train.py` to the path of the downloaded model, then run the `train.py` script.

To test the model, you should modify `is_CRF` and `model_name` in `test.py` to the path of the downloaded model, then
run the `test.py` script.

To see the predictions, you should modify `is_CRF` and `model_name` in `exhibit.py` to the path of the downloaded model,
then run the `exhibit.py` script.

`corpus_reader.py` is used to do the data preprocessing.

`test.log` is the log file for the test script which records the recall, precision and F1 score results of a certain
model.