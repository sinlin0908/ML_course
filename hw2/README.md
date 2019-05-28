# HW2

404410101 林政賢

## 題目

比較 resnet pretrain 與 unpretrain 的差異

## Data

- Goal: Car type classification
- Task: Image classification (196 classes of cars)
- Training / Testing: 8,144 / 8,041
- Ref: <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>

## Eviroment

- OS : Ubuntu 18.04LTS
- CPU : Intel(R) Xeon(R) CPU E5-1620 v3 @ 3.50GHz
- Ram: 50G
- GPU : GTX 1080TI 11G


## Result

### Pretrained

- epoch : 100
- batch size : 32
- learning rate : 0.0001
- loss function : cross entropy
- optimizer: Adam
- train dataset : 8144
- test dataset : 8041
- time : 
![](https://i.imgur.com/oF5lgID.png)
- accuracy: 
![](https://i.imgur.com/vQGvIt5.png)
- loss:
![](https://i.imgur.com/U6hMY1K.png)


### Unpretrained

#### 實驗一
- epoch : 100
- batch size : 32
- learning rate : 0.0001
- loss function : cross entropy
- optimizer: Adam
- train dataset : 8144
- test dataset : 8041
- time : 
![](https://i.imgur.com/AqeHH0F.png)

- accuracy: 
![](https://i.imgur.com/cGufSUp.png)

- loss:
![](https://i.imgur.com/fljDRwf.png)

#### 實驗二

這次試驗為把training dataset 左右反轉增加資料量，accuracy 增加 10%

- epoch : 100
- batch size : 32
- learning rate : 0.0001
- loss function : cross entropy
- optimizer: Adam
- train dataset : 16288
- test dataset : 8041
- time: 
![](https://i.imgur.com/gqZfgmj.jpg)
- accuracy:
![](https://i.imgur.com/IvZ7uaZ.jpg)

- loss
![](https://i.imgur.com/AvAOWJB.png)

### 實驗三
已實驗二資料再做RGB轉成BGR，Accuracy 又增加 10%

- epoch : 100
- batch size : 32
- learning rate : 0.0001
- loss function : cross entropy
- optimizer: Adam
- train dataset : 32576
- test dataset : 8041
- accuracy:
![](https://i.imgur.com/S8powCa.png)


## 心得

1. training 時 pretrain 比 unpretrain 更快到達 90以上
![](https://i.imgur.com/4fMt4U6.png)

2. 增加 training set 有助於增加準確率




## Error
### Loss 部分出現 RuntimeError: cuda runtime error (59) : device-side assert triggered

原因 :
![](https://i.imgur.com/n98PEz1.png)
好像是 num_classes 部分設定 196 是錯的要設定 197
