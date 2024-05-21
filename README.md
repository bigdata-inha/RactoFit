# RactoFit
(CoLLAs 2024) Replaying with Realistic Latent Vectors in Generative Continual Learning

<img src="https://github.com/hyemin-Jeong/RactoFit/blob/main/RactoFit.jpg" width="900">

## Running the code
### 1. Preliminary
- Prepare dataset for training  
  ```cd Data```  
  ```python main_data.py --task disjoint --dataset cifar10 --n_tasks 10 --dir ../Archives```  
  ```python main_data.py --task disjoint --upperbound True --dataset cifar10 --n_tasks ```  
- Prepare pre-trained model for FID (expert)  
  download from [here](https://github.com/huyvnphan/PyTorch_CIFAR10)

### 2. Training
- MerGAN
  ```
  python main.py --method Generative_Replay --dataset cifar10 --train_G True
  ```

- (0.8%) Rehearsal
  ```
  python main.py  --method Rehearsal --dataset cifar10 --train_G True  --nb_samples_rehearsal 50
  ```

- (0%) RactoFit
  ```
  python main.py  --method Ractofit_0 --dataset cifar10 --train_G True
  ```

- (0.8%) RactoFit
  ```
  python main.py --method Ractofit --dataset cifar10 --train_G True --num_z 1200
  ```


### 3. Evaluation
- FID (expert) ```python main.py --method Generative_Replay --dataset cifar10 --FID True```
- fitting capacity ```python main.py --method Generative_Replay --dataset cifar10 --Fitting_capacity True```

### Requirements
python=3.8.8  
pytorch=1.13.1  
scipy  
matplotlib  
tqdm  
imageio  
scikit-learn  
lpips

### Acknowledgment
Our code is based on the implementations of [Generative_Continual_Learning](https://github.com/TLESORT/Generative_Continual_Learning)
