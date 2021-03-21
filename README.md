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

### Задание 2. Реализовать и применить в обучении следующие политики изменения темпа обучения, а также определить оптимальные параметры для каждой политики:
### 2а. Пошаговое затухание (Step Decay)

### 2а. Экспоненциальное затухание (Exponential Decay)
####  owl-1616285996.999752, k=0.1
####  owl-1616287216.463951, k=0.2
####  owl-1616288058.7027614, k=0.3
####  owl-1616288907.0635068, k=0.4
####  owl-1616337614.888935, k=0.5
![image](https://user-images.githubusercontent.com/80168174/111909940-4bf13000-8a70-11eb-8d10-c1e00016f9f0.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_loss.svg">
