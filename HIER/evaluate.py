import random
import sys
import sqlite3
import numpy as np
import os
sys.path.append('..')
# from preprocessing.utils.dbPointer import queryResultVenues
import json

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']

# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
dbs = {}
for domain in domains:
    db = 'preprocessing/db/{}-dbase.db'.format(domain)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c


# from github multiwoz budiowaski - removed

# from hdsa
def clean(string):
    return string.lower().replace("'", "''").strip()

def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = turn.items()
    elif real_belief == 'tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = clean(val)
                    if key == 'leaveAt':
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = clean(val)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = clean(val)
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = clean(val)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:
        result = dbs[domain].execute(sql_query).fetchall()
        return result
    except:
        return []  # TODO test it


def issubset(A, B):
    A = set(A)
    B = set(B)
    return A.issubset(B)


def parseGoal(goal, d, domain):
    """Parses user goal into dictionary format."""
    goal[domain] = {}
    goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
    if 'info' in d['goal'][domain]:
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append('reference')
            if 'reqt' in d['goal'][domain]:
                if 'trainID' in d['goal'][domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if 'reqt' in d['goal'][domain]:
                for s in d['goal'][domain]['reqt']:  # addtional requests:
                    if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append("reference")

        goal[domain]["informable"] = d['goal'][domain]['info']
        if 'book' in d['goal'][domain]:
            goal[domain]["booking"] = d['goal'][domain]['book']

    return goal


def evaluateModel(dialogues, mode='valid'):
    """Gathers statistics for the whole sets."""
    fin1 = open('data/delex.json')
    delex_dialogues = json.load(fin1)

    # print('*** No of delex_dialogues: ',len(delex_dialogues))
    # print('*** No of dialogues: ', len(dialogues), '\n')

    successes, matches = 0, 0
    real_sucesses, real_matches = 0, 0
    total = 0

    gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0], 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
    sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    filenames = sorted(dialogues.keys())
    for filename in filenames:
        dial = dialogues[filename]
        if filename not in delex_dialogues:
            filename += ".json"

        data = delex_dialogues[filename]

        # print(dial, '\n',  data)
        success, match, stats = evaluateDialogue(dial, data)

        successes += success
        matches += match
        total += 1

    # Print results
    matches = matches / float(total) * 100
    successes = successes / float(total) * 100

    # print('Corpus Entity Matches : %2.2f%%' % (matches))
    # print('Corpus Requestable Success : %2.2f%%' % (successes))
    
    # return "{}_{}".format("%2.2f"%bleu, matches, successes)
    return matches, successes

def evaluateModel_Slow(dialogues, mode='valid'):
    """Gathers statistics for the whole sets."""
    fin1 = open('data/delex.json')
    delex_dialogues = json.load(fin1)

    # print('*** No of delex_dialogues: ',len(delex_dialogues))
    # print('*** No of dialogues: ', len(dialogues), '\n')

    successes, matches = 0, 0
    real_sucesses, real_matches = 0, 0
    total = 0

    gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0], 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
    sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    filenames = sorted(dialogues.keys())
    all_stats = {}
    for filename in filenames:
        dial = dialogues[filename]
        if filename not in delex_dialogues:
            filename += ".json"

        data = delex_dialogues[filename]

        # print(dial, '\n',  data)
        success, match, stats = evaluateDialogue(dial, data)
        all_stats[filename] = (match, success)

        successes += success
        matches += match
        total += 1

    # Print results
    matches = matches / float(total) * 100
    successes = successes / float(total) * 100

    # print('Corpus Entity Matches : %2.2f%%' % (matches))
    # print('Corpus Requestable Success : %2.2f%%' % (successes))
    
    # return "{}_{}".format("%2.2f"%bleu, matches, successes)
    return matches, successes, all_stats


def evaluateDialogue(dialog, realDialogue):
    # get the list of domains in the goal
    goal = {}
    for domain in domains:
        if realDialogue['goal'][domain]:
            goal = parseGoal(goal, realDialogue, domain)

    real_requestables = {}
    for domain in goal.keys():
        real_requestables[domain] = goal[domain]['requestable']

    # CHECK IF MATCH HAPPENED
    provided_requestables = {}
    venue_offered = {}

    for domain in goal.keys():
        venue_offered[domain] = []
        provided_requestables[domain] = []

    for t, sent_t in enumerate(dialog):
        #sent_t = sent_t.replace("colleges", "[attaraction_name]")
        #sent_t = sent_t.replace("college", "[attaraction_name]")
        for domain in goal.keys():
            # Search for the only restaurant, hotel, attraction or train with an ID
            if '[' + domain + '_name]' in sent_t or 'trainid]' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(domain, realDialogue['log'][t * 2 + 1])
                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = venues  # random.sample(venues, 1)
                    else:
                        flag = True
                        for ven in venue_offered[domain]:
                            if ven not in venues:
                                flag = False
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            venue_offered[domain] = venues
                else:
                    venue_offered[domain] = '[' + domain + '_name]'

            # ATTENTION: assumption here - we didn't provide phone or address twice! etc
            for requestable in requestables:
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:

                        if 'restaurant_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'train_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        # if name was provided for the user, the match is being done automatically
        if 'name' in goal[domain]['informable']:
            venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'

        if domain == 'train':
            if not venue_offered[domain]:
                if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[' + domain + '_name]'
    """
    Given all inform and requestable slots
    we go through each domain from the user goal
    and check whether right entity was provided and
    all requestable slots were given to the user.
    The dialogue is successful if that's the case for all domains.
    """
    # HARD EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0
    success = 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif venue_offered[domain]:
                groundtruth = queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                if issubset(venue_offered[domain], groundtruth):
                    match += 1
                    match_stat = 1
        else:
            if '[' + domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in goal.keys():
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    # rint requests, 'DIFF', requests_real, 'SUCC', success
    return success, match, stats
