# Diamonds Price Prediction

The dataset is provided by Kaggle: https://www.kaggle.com/datasets/shivam2503/diamonds.

This project is to predict whether the price of diamonds.

## Methodologies
This project is done under Spyder IDE.
The model is trained under 5 dense layers (128, 64, 32, 16, 16) with ReLU activation. 

```sh
dense = tf.keras.layers.Dense(128, activation = 'relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(64, activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(32, activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation = 'relu')
x = dense(x)
```

This part of coding can be found inside the `diamonds.py`.

## Results
The model is then trained under a 32 batch size and 30 epochs. The image shown in below is the graph of Actual vs Prediction.

✨Graph of Actual vs Prediction✨

![Scatter plot](https://user-images.githubusercontent.com/34246703/163230343-675bb958-be8d-473a-92e5-66b98a2b225f.png)
