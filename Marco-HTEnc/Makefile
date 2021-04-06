gpu?=0
test:
	echo GPU=$(gpu)
run4:
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model modelAct/ --batch_size 512 --max_seq_length 50 --act_source pred --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model modelAct/ --batch_size 512 --max_seq_length 50 --act_source pred --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model modelAct/ --batch_size 512 --max_seq_length 50 --act_source pred --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model modelAct/ --batch_size 512 --max_seq_length 50 --act_source pred --learning_rate 1e-4

run4bert:
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model model/ --batch_size 512 --max_seq_length 50 --act_source bert --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model model/ --batch_size 512 --max_seq_length 50 --act_source bert --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model model/ --batch_size 512 --max_seq_length 50 --act_source bert --learning_rate 1e-4
	CUDA_VISIBLE_DEVICES=$(gpu) python train_generator.py --option train --model model/ --batch_size 512 --max_seq_length 50 --act_source bert --learning_rate 1e-4