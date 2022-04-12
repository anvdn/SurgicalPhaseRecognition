python train-validate.py --model='MobileNet' --lr=.003 --epochs=4 --gamma=.5 --step_size=1 --os=0
python train-validate.py --model='MobileNet' --lr=.002 --epochs=4 --gamma=.4 --step_size=1 --os=1
python train-validate.py --model='MobileNet' --lr=.001 --epochs=8 --gamma=.8 --step_size=1 --os=0
python train-validate.py --model='MobileNet' --lr=.001 --epochs=8 --gamma=.7 --step_size=1 --os=1
python train-validate.py --model='MobileNetStage' --lr=.003 --epochs=4 --gamma=.5 --step_size=1 --os=0
python train-validate.py --model='MobileNetStage' --lr=.002 --epochs=4 --gamma=.4 --step_size=1 --os=1
python train-validate.py --model='MobileNetStage' --lr=.001 --epochs=8 --gamma=.8 --step_size=1 --os=0
python train-validate.py --model='MobileNetStage' --lr=.001 --epochs=8 --gamma=.7 --step_size=1 --os=1