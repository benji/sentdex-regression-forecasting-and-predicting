# sentdex-regression-forecasting-and-predicting

Based on [Sentdex linear regression tutorials series](https://www.youtube.com/watch?v=OGxgnH8y2NM&index=1&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v).

# Prep

Download the working dataset with:

```
python download_dataset.py
````

# Lesson 5

Modifications:

- We changed the we train on all the rows except last 60
- we use the next 30 rows to make predictions of the last 30 rows' stock prices to be able to visually compare
- we use the last 30 rows to make predictions 30 days into the future
- changes to simplify the datetime manipulations
![](https://raw.github.com/benji/sentdex-regression-forecasting-and-predicting/master/images/lesson5_predictions.png)

The red line show actual stock data, the blue line shows the prediction of the model that carries out 30 days into the future (at the time).