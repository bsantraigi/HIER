import json, numpy

PAD = 0
UNK=1
SOS=2
EOS=3

max_sent_len = 50

def append_or_add(dictionary, name, key):
    if name in dictionary:
        dictionary[name].append(key)
    else:
        dictionary[name] = [key]

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police', 'bus', 'booking', 'general']
functions = ['inform', 'request', 'recommend', 'book', 'select', 'sorry', 'none']
arguments = ['pricerange', 'id', 'address', 'postcode', 'type', 'food', 'phone', 'name', 'area', 'choice', 
             'price', 'time', 'reference', 'none', 'parking', 'stars', 'internet', 'day', 'arriveby', 'departure', 
             'destination', 'leaveat', 'duration', 'trainid', 'people', 'department', 'stay']

used_levels = domains + functions + arguments
#used_levels = functions + arguments
act_len = len(used_levels)
def act_to_vectors(acts):
    r = numpy.zeros((act_len, ), 'float32')
    for act in acts:
        p1, p2, p3 = act.split('-')
        if len(used_levels) == len(domains + functions + arguments):
            r[domains.index(p1)]
            r[len(domains) + functions.index(p2)] += 1
            r[len(domains) + len(functions) + arguments.index(p3)] += 1
        else:
            r[functions.index(p2)] += 1
            r[len(functions) + arguments.index(p3)] += 1            
    return (r > 0).astype('float32')

id_to_acts = {}
for i, name in enumerate(used_levels):
    id_to_acts[i] = name 

with open('data/act_ontology.json', 'r') as f:
    act_ontology = json.load(f)

# with open('data/belief_state.json', 'r') as f:
#     belief_state = json.load(f)


