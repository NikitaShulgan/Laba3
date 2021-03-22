# Лабораторная работа #3
## Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning
### Задание 1. С использованием и техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife с использованием фиксированных темпов обучения 0.1, 0.01, 0.001, 0.0001
#### owl-1616081750.4760325, lr = 0.1
#### owl-1616093079.8669348, lr = 0.01
#### owl-1616095020.9706807, lr = 0.001
#### owl-1616096200.7460656, lr = 0.0001, нижний оранжевый owl-1616096200.7460656/validation
![image](https://user-images.githubusercontent.com/80168174/111886334-10a62100-89de-11eb-8a00-231c14ecca2a.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/1_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/1_epoch_loss.svg">

#### Только валидация:
![image](https://user-images.githubusercontent.com/80168174/111938058-31ec3780-8ada-11eb-8924-06ac3f681654.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/1_valid_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/1_valid_epoch_loss.svg">

### Задание 2. Реализовать и применить в обучении следующие политики изменения темпа обучения, а также определить оптимальные параметры для каждой политики:
### 2а. Пошаговое затухание (Step Decay)
```
BATCH_SIZE = 64

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 5.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)
```
#### owl-1616282276.646658, drop = 0.5, epochs_drop = 10.0
#### owl-1616284313.4728131, drop = 0.6, epochs_drop = 10.0
#### owl-1616339973.6213276, drop = 0.4, epochs_drop = 10.0
#### owl-1616341240.7965152, drop = 0.5, epochs_drop = 5.0
![image](https://user-images.githubusercontent.com/80168174/111912113-40eecd80-8a79-11eb-8551-a5e713ebf7d5.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_epoch_loss.svg">
#### Только валидация:

![image](https://user-images.githubusercontent.com/80168174/111938433-fa31bf80-8ada-11eb-8fa2-4c0045bcb787.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_valid_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_valid_epoch_loss.svg">

### 2b. Экспоненциальное затухание (Exponential Decay)
```
BATCH_SIZE = 64

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate
   
lrate = LearningRateScheduler(exp_decay)
```
####  owl-1616285996.999752, k=0.1, серый owl-1616285996.999752/valid снизу
####  owl-1616287216.463951, k=0.2, оранжевый сверху в epoch_categorical_accuracy owl-1616287216.463951/train
####  owl-1616288058.7027614, k=0.3
####  owl-1616288907.0635068, k=0.4, зеленый owl-1616288907.0635068/valid снизу
####  owl-1616337614.888935, k=0.5
![image](https://user-images.githubusercontent.com/80168174/111909940-4bf13000-8a70-11eb-8d10-c1e00016f9f0.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_loss.svg">
### Анализ результатов
   Нами было исследовано 2 алгоритма затухания: Пошаговое затухание (Step Decay) и Экспоненциальное затухание (Exponential Decay). По результатам экспериментов видно, что во всех случаях epoch_loss на валидации 0.2 и epoch_categorical_accuracy на валидации 89%. Но используя пошаговое затухание, мы смогли раньше приблизиться к итоговым результатам. 
