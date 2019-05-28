# Machine Learning Hw 1 report

資工四 404410101 林政賢

## 題目

練習使用 Pytorch 做參數調整訓練 VGG16 模型進行圖片分類
並和題目給定的參數做比較

## Data

- Goal: Classify image data of natural scenes around the world
- Task: image classification (6 classes)
- Training / Testing: around 14k / 3k
- Refernce: <https://www.kaggle.com/puneet6060/intel-image-classification>

## 實驗環境

OS: Ubuntu 18.04 LTS
GPU: GTX 1080Ti


## 範例 Model
Τrain loss
![](https://i.imgur.com/DWIFWvN.png)


Train accuracy
![](https://i.imgur.com/0r2GVUj.png)

Test accuracy
![](https://i.imgur.com/HbOix3K.png)

## 調整參數 Model

### Method description

1. Parmeter: 
    - batch size: 32
    - learning rate: 0.0001
    - epoch: 100
2. Bactch Normalize:
    由於參數無法突破範例 model，因此我在 model 新增了 Batch Normalize 層，上升 1~3％

### Experimental results

![](https://i.imgur.com/Wv5IPom.png)

![](https://i.imgur.com/unDfbzv.png)


## Discussion

1. 使用 Adam Loss 下降比 SGD 較快，大概 第10 epoch accuracy 就到 90％
2. 由於試了很多參數都 test accuracy 無法突破 86 大關，於是我就增加了 batch normalize layer 結果到 87% 至 88% 左右


