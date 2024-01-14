# how to use

# 1. Install requirements

```
pip3 install -r requirements.txt
```
# 2. Train on MNIST

## 2.1 Plain Mode
```
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Plain
```

## 2.2 Encrypt with Paillier
```
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 128

python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 256
...

python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 2048
```

## 2.2 Encrypt with CKKS

### with different poly_modulus_degree
```
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 1024
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 2048
...
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 32768

```

### with different  multiplication depth
```
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9--mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 1024
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9--mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 2048
...
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 32768
```

### with different security level
```
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 2048
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 192 --ckks_mul_depth 0 --ckks_key_len 2048
...
python3 main.py --gpu -1 --dataset mnist --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 256 --ckks_mul_depth 0 --ckks_key_len 2048
```

# 3. Train on Cifar-10

## 3.1 Plain Mode

with LetNet:
```
python3 main.py --gpu -1 --dataset mnist --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Plain
```

with AlexNet:
```
python3 main.py --gpu -1 --dataset mnist --model AlexNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Plain
```
## 3.2 Encrypt with Paillier

```
python3 main.py --gpu -1 --dataset mnist --model LeNet  --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 128

python3 main.py --gpu -1 --dataset mnist --model LeNet  --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 256

...

python3 main.py --gpu -1 --dataset mnist --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode Paillier --phe_key_len 2048

```

## 3.2 Encrypt with CKKS

### with different poly_modulus_degree
```
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 1024

python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 2048

...
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 32768
```

### with different  multiplication depth
```
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 1024

python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 1 --ckks_key_len 1024
...
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 5 --ckks_key_len 1024

```

### with different security level
```
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 128 --ckks_mul_depth 0 --ckks_key_len 2048
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 192 --ckks_mul_depth 0 --ckks_key_len 2048
...
python3 main.py --gpu -1 --dataset cifar --model LeNet --num_channels 1  --epochs 10 --local_ep 10 --lr 0.015 --momentum 0.9 --mode CKKS --ckks_sec_level 256 --ckks_mul_depth 0 --ckks_key_len 2048
```