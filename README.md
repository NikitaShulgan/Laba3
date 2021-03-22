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

#### По 2 графикам наилучший результат при ```lr=0.001```. Найменьшая функция ошибки 0,1909 на 32 эпохе. Метрика качества 89,43% на 32 эпохе.

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
#### owl-1616282276.646658 ```drop = 0.5, epochs_drop = 10.0```
#### owl-1616284313.4728131 ```drop = 0.6, epochs_drop = 10.0```
#### owl-1616339973.6213276, ```drop = 0.4, epochs_drop = 10.0```
#### owl-1616341240.7965152, ```drop = 0.5, epochs_drop = 5.0```

#### owl-1616436470.0354586 ```drop = 0.4    epochs_drop = 5.0```
#### owl-1616437242.9938867 ```drop = 0.4    epochs_drop = 3.0```
#### owl-1616438159.4302144  ```drop = 0.5   epochs_drop = 3.0```
#### owl-1616438913.9406679 ```drop = 0.6   epochs_drop = 3.0```
#### owl-1616439741.1608436 ```drop = 0.6   epochs_drop = 5.0```

![image](https://user-images.githubusercontent.com/80168174/112048167-2db62d80-8b5f-11eb-9248-964abe359a13.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/full_step_decay_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/full_step_decay_epoch_loss.svg">

#### На 16й эпохе функция потерь с ```drop = 0.6, epochs_drop = 5.0``` (owl-1616439741.1608436) достигла своего мин значения 0,2001. Метрика качества на 16й эпохе 88,88%. 
#### На 26й эпохе функция потерь с ```drop = 0.5, epochs_drop = 10.0``` (owl-1616282276.646658) достигла своего мин значения 0,1922. Метрика качества на 24 эпохе 89,30%.
#### При уменьшении параметра ```epochs_drop``` у нас увеличивается скорость сходимости алгоритма, но мы теряем в точности.
### Дополнительные графики:

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

![image](https://user-images.githubusercontent.com/80168174/112046863-9ac8c380-8b5d-11eb-8325-38a15e256695.png)


#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_decay_2_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/step_dacay_2_epoch_loss.svg">

### 2b. Экспоненциальное затухание (Exponential Decay)
[Tensorboard](https://tensorboard.dev/experiment/kJJ9XASQR5CQk3DOAFmJow/#scalars&runSelectionState=eyJvd2wtMTYxNjA4MTc1MC40NzYwMzI1L3RyYWluIjpmYWxzZSwib3dsLTE2MTYwODE3NTAuNDc2MDMyNS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTYwOTMwNzkuODY2OTM0OC90cmFpbiI6ZmFsc2UsIm93bC0xNjE2MDkzMDc5Ljg2NjkzNDgvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2MDk1MDIwLjk3MDY4MDcvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjA5NTAyMC45NzA2ODA3L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjA5NjIwMC43NDYwNjU2L3RyYWluIjpmYWxzZSwib3dsLTE2MTYwOTYyMDAuNzQ2MDY1Ni92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTYyODIyNzYuNjQ2NjU4L3RyYWluIjpmYWxzZSwib3dsLTE2MTYyODIyNzYuNjQ2NjU4L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjI4NDMxMy40NzI4MTMxL3RyYWluIjpmYWxzZSwib3dsLTE2MTYyODQzMTMuNDcyODEzMS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTYyODU5OTYuOTk5NzUyL3RyYWluIjpmYWxzZSwib3dsLTE2MTYyODU5OTYuOTk5NzUyL3ZhbGlkYXRpb24iOnRydWUsIm93bC0xNjE2Mjg3MjE2LjQ2Mzk1MS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2Mjg3MjE2LjQ2Mzk1MS92YWxpZGF0aW9uIjp0cnVlLCJvd2wtMTYxNjI4ODA1OC43MDI3NjE0L3RyYWluIjpmYWxzZSwib3dsLTE2MTYyODgwNTguNzAyNzYxNC92YWxpZGF0aW9uIjp0cnVlLCJvd2wtMTYxNjI4ODkwNy4wNjM1MDY4L3RyYWluIjpmYWxzZSwib3dsLTE2MTYyODg5MDcuMDYzNTA2OC92YWxpZGF0aW9uIjp0cnVlLCJvd2wtMTYxNjMzNzYxNC44ODg5MzUvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjMzNzYxNC44ODg5MzUvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTYzMzk5NzMuNjIxMzI3Ni90cmFpbiI6ZmFsc2UsIm93bC0xNjE2MzM5OTczLjYyMTMyNzYvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2MzQxMjQwLjc5NjUxNTIvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjM0MTI0MC43OTY1MTUyL3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjQwMjk0OS4zMzQxNTYzL3RyYWluIjpmYWxzZSwib3dsLTE2MTY0MDI5NDkuMzM0MTU2My92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY0MDcxODIuMjM1MjYyNC90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDA3MTgyLjIzNTI2MjQvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MTQ3NTIuMjAzMDkwMi90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDE0NzUyLjIwMzA5MDIvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MTU2ODkuMjc2NDY2MS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDE1Njg5LjI3NjQ2NjEvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MTcyNzkuNDYyMTU1My90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDE3Mjc5LjQ2MjE1NTMvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2NDE4MjAyLjA3MjUxMi90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDE4MjAyLjA3MjUxMi92YWxpZGF0aW9uIjp0cnVlLCJvd2wtMTYxNjQyNzcyMi4yMDE4MTIvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjQyNzcyMi4yMDE4MTIvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2NDMwNjY2LjQyNzExMTkvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjQzMDY2Ni40MjcxMTE5L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjQzMzYwNS41NTg1NjEzL3RyYWluIjpmYWxzZSwib3dsLTE2MTY0MzM2MDUuNTU4NTYxMy92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY0MzQwMTQuMzk4NDM1NC90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDM0MDE0LjM5ODQzNTQvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MzQ4MTYuMzgwMjQ1Mi90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDM0ODE2LjM4MDI0NTIvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MzU1NzcuNDQwMjMwNi90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDM1NTc3LjQ0MDIzMDYvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTY0MzY0NzAuMDM1NDU4Ni90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDM2NDcwLjAzNTQ1ODYvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2NDM3MjQyLjk5Mzg4NjcvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjQzNzI0Mi45OTM4ODY3L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjQzODE1OS40MzAyMTQ0L3RyYWluIjpmYWxzZSwib3dsLTE2MTY0MzgxNTkuNDMwMjE0NC92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY0Mzg5MTMuOTQwNjY3OS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NDM4OTEzLjk0MDY2NzkvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2NDM5NzQxLjE2MDg0MzYvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjQzOTc0MS4xNjA4NDM2L3ZhbGlkYXRpb24iOmZhbHNlfQ%3D%3D&_smoothingWeight=0&run=owl-1616081750.4760325%2Ftrain)
```
BATCH_SIZE = 64

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate
   
lrate = LearningRateScheduler(exp_decay)
```
####  owl-1616285996.999752, ``` k=0.1 ```, серый owl-1616285996.999752/valid снизу
####  owl-1616287216.463951, ```k=0.2```, оранжевый сверху в epoch_categorical_accuracy owl-1616287216.463951/train
####  owl-1616288058.7027614, ```k=0.3```
####  owl-1616288907.0635068, ```k=0.4```, зеленый owl-1616288907.0635068/valid снизу
####  owl-1616337614.888935, ```k=0.5```
#### owl-1616434014.3984354 ```k=0.4 l=0.01```
#### owl-1616434816.3802452 ```k=0.5 l=0.01```
#### owl-1616435577.4402306 ```k=0.6 l=0.01```
#### owl-1616407182.2352624 ```k=0.6```
#### owl-1616414752.2030902 ```k=0.7```
#### owl-1616415689.2764661 ```k=0.8```
#### owl-1616417279.4621553 ```k=0.9```
#### owl-1616418202.072512 ```k=1.0```
![image](https://user-images.githubusercontent.com/80168174/111909940-4bf13000-8a70-11eb-8d10-c1e00016f9f0.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_epoch_loss.svg">

#### Только валидация

![image](https://user-images.githubusercontent.com/80168174/111938945-1da93a00-8adc-11eb-97e5-7b3c5aa48458.png)
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_valid_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/exp_valid_epoch_loss.svg">

#### На 25й эпохе график с ```k=0.1``` достиг минимальной ошибки 0,1868. Метрика качества 89,07%.

### Общее сравнение
![image](https://user-images.githubusercontent.com/80168174/112022103-02bde080-8b43-11eb-8588-3939dfdb8f52.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/best_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba3/main/For_Readmi/best_epoch_loss.svg">

###  Анализ результатов
   Я исследовал 2 алгоритма затухания: Пошаговое затухание (Step Decay) и Экспоненциальное затухание (Exponential Decay). По результатам экспериментов видно, что в первом задании наилучший результат на валидации 89,43% (мы использовали ```lr=0.001```), в пошаговом затухании 89,13% (```drop = 0.6, epochs_drop = 10.0```), в экспоненциальном затухании 89,07% (```k=0.1```). Наилучший результат у меня с фиксированным темпом обучения ```lr=0.001```. Если сравнивать с экспоненциальным и пошаговым затуханием, то разница 0,36% и 0,30% соответственно. Лучше всего себя показало фиксированное затухание с  ```lr=0.001```.
