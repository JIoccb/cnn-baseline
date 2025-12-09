train:
	PYTHONPATH=. python src/train.py

test:
	PYTHONPATH=. python src/test.py

tensorboard:
	tensorboard --logdir /home/student03/work/Artyom/work_dir/experiments/artyom_docs/log_dir