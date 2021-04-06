#!/home/bishal/miniconda3/envs/py2/bin/python
# coding: utf-8

# In[1]:


import json
import os

from evaluate import *
import pandas as pd


# ## Evaluating model predictions

mode = "test"
evaluator = MultiWozEvaluator(mode)

# GT
# with open("./data/test_dials.json", "r") as f:
#     human_raw_data = json.load(f)

# HIER
model_folder = "../running/bpe_hier++"
for x in ['greedy', 'beam_2', 'beam_3', 'beam_5']:
    prediction_json = "{}/model_turns_{}_test.json".format(model_folder, x)
    if os.path.isfile(prediction_json):
        print "\nDecoding Method:", x.upper(), '\n-------------------'
        with open(prediction_json, "r") as f:
            _temp_gen = json.load(f)
        generated_data = {}
        for key, value in _temp_gen.items():
            generated_data[key + '.json'] = value

        # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
        # generated_data = human_proc_data
        # for key, value in human_raw_data.items():
        #     human_proc_data[key] = value['sys'] # Array of system utterances

        _, _, _, all_match_success = evaluator.evaluateModel(generated_data, mode=mode)

        # Match and Success stats
        pred_file = prediction_json.replace("model_turns", "stats").replace("json", "tsv")
        all_match_success = pd.DataFrame.from_records(all_match_success)
        all_match_success.to_csv(pred_file, sep="\t", index=False)
    else:
        print "skip", x.upper(), '\n'
