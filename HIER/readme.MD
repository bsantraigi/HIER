# HIER Project

## Error Analysis Steps

1. First run `main_acts.py` or `main.py` with test_split uncommented (inside run function). This will save the prediction json files in `running` folder.
2. Then run `new_eval/new_eval.ipynb` to obtain the stats file with inform and success rates.
3. Finally, the error analysis is done in `error_reader.ipynb` notebook.

```
main_acts.py (eval only) -> error_analysis -> new_eval -> error_reader.ipynb
```

#### DO NOT TRY TO MERGE MAIN_ACTS AND ERROR_ANALYSIS. In ErrorAnalysis.ipydb, batch_size needs to be 1, so it's slow.

### DATASET: (From https://github.com/wenhuchen/HDSA-Dialog)

generated using create_delex_data.py in original multiwoz repository for multiwoz 2.1 version. <br>
add preprocessed data - train.json, val.json and test.json into hdsa_data/hdsa_data/ folder <br>
Add delex.json file into data folder, large file(~87MB)

```bash
wget --directory-prefix=data/  https://hdsa-dialog.s3-us-west-2.amazonaws.com/delex.json
```

### MODELS:
set, hier, mat, set++, hier++

Number of layers: e1,e2,d

#### SET/SET++:
initially, positional embeddings wrt individual utterances are added. After e1 layers, positional embedding wrt whole dialog are addded.  
For all e1+e2 layers, attention in between individual utterances.

#### HIER/HIER++:
For e1 layers, attention in between individual utterances.
For e2 layers, cross attention between current and last utterance.
For d layers in decoder, lower triangular mask to prevent from attending to future positions of response.

#### MAT:
For e2 layers, cross attention between current and last utterance.
For d layers in decoder, lower triangular mask to prevent from attending to future positions of response.


### HYPERPARAMETERS:
Number of layers: e1,e2,d
Number of heads in each encoder/decoder


### Reproducability:

For SET/HIER/MAT:
python main.py

For SET++/HIER++:
python main_acts.py





