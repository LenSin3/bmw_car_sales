# BMW Used Car Sales - Price Predictor

[Click here](https://bmw-car-price-predictor.herokuapp.com/) to take you to the Price Calculator App in Heroku.

App user iterface development repo is found [here.](https://github.com/LenSin3/bmw_car_sales_deployed)

![BMW Car logo](https://github.com/LenSin3/bmw_car_sales/blob/main/images/bmw_logo.jpg?raw=true)

Source: [WallpaperUse](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwc.wallpaperuse.com%2Fwallp%2F72-728441_s.jpg&imgrefurl=https%3A%2F%2Fwww.wallpaperuse.com%2Fvien%2FbJxRhR%2F&tbnid=xPVRsC9TFm6teM&vet=12ahUKEwij2ff30ab0AhUH6J4KHebACHgQMyg4egQIARBU..i&docid=A4f5eNgUWKF7SM&w=621&h=380&itg=1&q=bmw%20logo&ved=2ahUKEwij2ff30ab0AhUH6J4KHebACHgQMyg4egQIARBU)


## Background

BMW is one of the best luxury car brands and it is considered a status symbol around the world. A brand new BMW, for example a **740i/750i xDrive** Model will cost a minimum of $86,000 USD. As you can see, not everyone is cut out to spend that much on a car. Second hand or used vehicles are the go to choice for most people who cannot afford the cost of a new car.

In the light of the above, this project will use Supervised Machine Learning to predict the price of a used BMW car. 

## Method

The project will follow the below:

- Read data.
- Data validation and profiling.
- Data transformation.
- exploratory data analysis.
- Machine Learning model developmenmt.
- Feature importances.

## Data

The data used in this project is obtained from [Datacamp's Career Hub repository](https://github.com/datacamp/careerhub-data) on GitHub.

## Example plots from Exploratory Data Analysis

![Time Series of BMW Used car price](https://github.com/LenSin3/bmw_car_sales/blob/main/images/tmseriesn1.png?raw=true)

Initially, we can see prices plummet from around 1996 to 2000. The gradual increase is obeserved from 2000 to 2005 when they slightly dropped again and plateaued until 2006. This trend changed right after with a yearly increase on to 2020.

We would also like to see how the models rank over the years. We will rank models with respect to Median Price as seen below.

![Model Rank by Mean Price](https://github.com/LenSin3/bmw_car_sales/blob/main/images/price_model1.png?raw=true)

**X7** is the most expensive car when comparing average price over the years.

The histogram below shows a right skew wherein bulk of the data lie on the left.

![Histogram of Price](https://github.com/LenSin3/bmw_car_sales/blob/main/images/price_distribution1.png?raw=true)

## Supervised Machine Learning - Regression

The following approach is employed to accomplish this:

- Data Preprocessing
- Develop and train multiple Regression models
- Perform hyperparameter tuning for the best model
- Extract feature importances

Six regression models were trained including **SGDRegressor**, **Ridge**, **Lasso**, **ElasticNet**, **DecisionTreeRegressor** and **RandomForestRegressor**.

![R Squared Ranking of Regressors](https://github.com/LenSin3/bmw_car_sales/blob/main/images/Regressor_R_Squared1.png?raw=true)

The best regressor from the above plot is **RandomForestRegressor** with an approximate accuracy (R Squared) of 0.95, signifying our model is able to account for 95% of the variations in the data. Pretty good, huh!

## Feature Importances

![Feature Importances](https://github.com/LenSin3/bmw_car_sales/blob/main/images/Feature_Coefficient1.png?raw=true)

We were able to extract the feature importances after the model was fit to the training data as seen in the chart above. The output aligns with our assumption that age of the car, engine size and gas consumption will be among the top ranked features to influence the price of a used bmw car.

## Dependencies

- Numpy 1.19.2
- Matplotlib 3.3.2
- Pandas 1.1.5
- Seaborn 0.11.1
- Scikit -Learn 0.23.2

## Scripts

- funcs.py
- mlfuncs.py
- plots.py
- utils.py

## Notebook

- bmw_ml.ipynb

## Environment

- python 3.6.12