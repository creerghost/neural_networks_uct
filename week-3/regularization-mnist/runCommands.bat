python mnist_regularization.py
python mnist_regularization.py --no_dataset_restriction
python mnist_regularization.py --hidden_layer_sizes="400,400"
python mnist_regularization.py --weight_decay=0.01
python mnist_regularization.py --weight_decay=0.1
python mnist_regularization.py --weight_decay=0.3
python mnist_regularization.py --weight_decay=0.5
python mnist_regularization.py --dropout=0.1
python mnist_regularization.py --dropout=0.3
python mnist_regularization.py --dropout=0.5
python mnist_regularization.py --label_smoothing=0.1
python mnist_regularization.py --label_smoothing=0.3
python mnist_regularization.py --label_smoothing=0.5
python mnist_regularization.py --dropout=0.3 --label_smoothing=0.3
python mnist_regularization.py --dropout=0.1 --label_smoothing=0.1
python mnist_regularization.py --dropout=0.1 --label_smoothing=0.3