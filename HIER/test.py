from main import run

import argparse

parser = argparse.ArgumentParser() 

parser.add_argument("-embed", "--embedding_size", default=100, type=int, help = "Give embedding size")
parser.add_argument("-heads", "--nhead", default=4, type=int,  help = "Give number of heads")
parser.add_argument("-hid", "--nhid", default=100, type=int,  help = "Give hidden size")

parser.add_argument("-l_e1", "--nlayers_e1", default=3, type=int,  help = "Give number of layers for Encoder 1")
parser.add_argument("-l_e2", "--nlayers_e2", default=3, type=int,  help = "Give number of layers for Encoder 2")
parser.add_argument("-l_d", "--nlayers_d", default=3, type=int,  help = "Give number of layers for Decoder")

parser.add_argument("-d", "--dropout",default=0.2, type=float, help = "Give dropout")
parser.add_argument("-bs", "--batch_size", default=16, type=int, help = "Give batch size")
parser.add_argument("-e", "--epochs", default=3, type=int, help = "Give number of epochs")
parser.add_argument("-model", "--model_type", default="SET", help="Give model name one of [SET, HIER, MAT]")

args = parser.parse_args() 

test_loss, test_bleu, test_f1entity, matches, successes = run(args)
print(test_loss, test_bleu, test_f1entity, matches, successes)