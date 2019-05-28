# Hw5 股票預測

404410101 資工四 林政賢

## Enviroment

- Os : Ubuntu 18.04 LTS
- GPU : GTX 1070 8G
- CPU : i7-4770
- Ram : 16G

## Requirement

- pytorch == 1.0.1
- numpy == 1.16.2
- pandas == 0.24.2
- matplotlib == 3.0.3
- scikit-learn == 0.21.1

## Data

Ref : <https://www.kaggle.com/dgawlik/nyse>

只用 STT 這家公司的資料做訓練

## Preprocess

把資料切 0.85 當作 train data，剩下當作 test data


## Model

```bash=
Model(
  (gru): GRU(5, 64, num_layers=2, dropout=0.5)
  (output_layer): Linear(in_features=64, out_features=1, bias=True)
)

```

## 參數
- epoch: 50
- batch size: 10
- learning rate: 0.0001
- model
    - input_dim=5,
    - hidden_dim=64, 
    - num_layers=2, 
    - output_dim=1
    - dropout=0.0

## 結果
- Loss
![](https://i.imgur.com/NsBLmBV.png)


- Test 預測圖：
至少漲跌部份有預測到
![](https://i.imgur.com/6tGDYZM.png)

- 估測誤差: 420.03254 (公式 $\sum_{k=1}^{N}(Predict_k-y_k)^2$)


## Usage

preprocess

```bash=
python prepro.py
```

train

```bash=
python train.py
```

test
```bash=
python test.py
```