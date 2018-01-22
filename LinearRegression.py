# Experiments with several regression against Olympics data
# using scikit-learn and various python libraries

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

# Experiment 1 - Load olympics.mat
# Reference - https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

olympics_data = sio.loadmat('olympics.mat')

# Experiment 2
# Use matplotlib to plot male100
# Reference - http://matplotlib.org/users/pyplot_tutorial.html

# Get the male100 data
male100_data = olympics_data['male100']
# Get the year and time column values
train_year_male100 = male100_data[:, :1]
train_time_male100 = male100_data[:, 1:2]


# Plot the graph with read circle markers
plt.plot(train_year_male100, train_time_male100, 'ro')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 2 - Use matplotlib to plot male100')
plt.show()

# Experiment 3
# 1) Use sklearn.linear_model.LinearRegression to fit male100
# 2) List the coefficients
# 3) Predict for values x = 2012 and x = 2016
# 4) Compare results with ones from textbook section 1.2
# Reference - http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# Create the linear regression object
regr = linear_model.LinearRegression()

# Train the model using the data
regr.fit(train_year_male100, train_time_male100)

# List the coefficients -
print('\nExperiment 3 - Use sklearn.linear_model.LinearRegression to fit male100')
print('Coefficients : ', np.around(regr.coef_, decimals=4))
print('Intercept : ', np.around(regr.intercept_, decimals=3))

# Make predictions
pred_time_male100_all = regr.predict(train_year_male100)
test_year_male100 = np.array([[2012], [2016]])
pred_time_male100 = regr.predict(test_year_male100)
for x, y in zip(test_year_male100, pred_time_male100):
    print('Winning times of year ', x[0], ':', np.around(y, decimals=3)[0])

# Compare results from textbook section 1.2
# |-----------------------------------------|
# |                | Textbook  |  Program   |
# |----------------|-----------|------------|
# | Predicted 2012 | 9.585     | 9.595      |
# | Predicted 2016 | 9.541     | 9.541      |
# | Coefficients   | -0.0133   | -0.0133    |
# | Intercept      | 36.416    | 36.416     |
# |-----------------------------------------|

# Experiment 4
# Plot the male100 and the linear regression model used to fit the data
plt.scatter(train_year_male100, train_time_male100, c='r')
plt.plot(train_year_male100, pred_time_male100_all, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 4 - Plot the male100 and the linear regression model')
plt.show()

# Experiment 5
# 1) Use linear regression to fit a line to female400
# 2) Compare error of the two models used to fit male100 and female400

# Get the female400 data
female400_data = olympics_data['female400']
# Get the year and time column data
train_year_female400 = female400_data[:, :1]
train_time_female400 = female400_data[:, 1:2]
# Train the model using the data
regr.fit(train_year_female400, train_time_female400)
pred_time_female400_all = regr.predict(train_year_female400)
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_female400_all, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 5 - Use linear regression to fit a line to female400')
plt.show()
print('\nExperiment 5 - Use linear regression to fit a line to female400')
print("Mean squared Error comparison for Male100 and Female400 models")
print("Mean square Error - Male100 : %.3f" % mean_squared_error(train_time_male100, pred_time_male100_all))
print("Mean square Error - Female400 : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all))
print("Since the error is less in Male100 than Female400, the model fits male100 data better than female400")

# Experiment 6
# Fit a 3rd order polynomial to female400
# Reference - http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html

# Perform non-linear regression with linear model, using a pipeline to add non-linear features
# Create a degree three polynomial model
poly_model_3 = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
poly_model_3.fit(train_year_female400, train_time_female400)
pred_time_female400_all_3 = poly_model_3.predict(train_year_female400)
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_female400_all_3, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 6 - Fit a 3rd order polynomial to female400')
plt.show()
# Compare error to lower degree model
print('\nExperiment 6 - Fit a 3rd order polynomial to female400')
print("Mean square Error - Female400 - Linear : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all))
print("Mean square Error - Female400 - Degree 3 : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all_3))
print("Yes the error improves,"
      "since the error is lower in Degree 3 model than linear model, it fits female400 data better")

# Experiment 7
# Fit a 5th order polynomial to female400
# Reference - http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html

# Perform non-linear regression with linear model, using a pipeline to add non-linear features
# Create a degree three polynomial model
poly_model_5 = make_pipeline(PolynomialFeatures(degree=5), LinearRegression(fit_intercept=False))
poly_model_5.fit(train_year_female400, train_time_female400)
pred_time_female400_all_5 = poly_model_5.predict(train_year_female400)
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_female400_all_5, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 7 - Fit a 5th order polynomial to female400')
plt.show()
# Compare error to lower degree model
print('\nExperiment 7 - Fit a 5th order polynomial to female400')
print("Mean square Error - Female400 - Linear : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all))
print("Mean square Error - Female400 - Degree 3 : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all_3))
print("Mean square Error - Female400 - Degree 5 : %.3f" % mean_squared_error(train_time_female400, pred_time_female400_all_5))
print("The error does not improve,"
      "And slightly increases from degree 3 to degree 5")

# Experiment 8
# Use LOOCV for both 3rd and 5th order polynomials
# Reference - http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html

print('\nExperiment 8 - Use LOOCV for both 3rd and 5th order polynomials')
poly_model_3_loo = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
results_3 = []
loo = LeaveOneOut()
for train_index, test_index in loo.split(train_year_female400):
    train_year_f400_loo, test_year_f400_loo = train_year_female400[train_index], train_year_female400[test_index]
    train_time_f400_loo, test_time_f400_loo = train_time_female400[train_index], train_time_female400[test_index]
    poly_model_3_loo.fit(train_year_f400_loo, train_time_f400_loo)
    results_3.append(mean_squared_error(test_time_f400_loo, poly_model_3_loo.predict(test_year_f400_loo)))

formatted_result_3 = ["%.3f" % item for item in results_3]
pred_time_f400_all_3_loo = poly_model_3_loo.predict(train_year_female400)
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_f400_all_3_loo, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 8 - Use LOOCV for 3rd order polynomial')
plt.show()
print("Mean square Error - Female400 - Degree 3 - LOOCV : %.3f" % mean_squared_error(train_time_female400, pred_time_f400_all_3_loo))

poly_model_5_loo = make_pipeline(PolynomialFeatures(degree=5), LinearRegression(fit_intercept=False))
results_5 = []
loo = LeaveOneOut()
for train_index, test_index in loo.split(train_year_female400):
    train_year_f400_loo, test_year_f400_loo = train_year_female400[train_index], train_year_female400[test_index]
    train_time_f400_loo, test_time_f400_loo = train_time_female400[train_index], train_time_female400[test_index]
    poly_model_5_loo.fit(train_year_f400_loo, train_time_f400_loo)
    results_5.append(mean_squared_error(test_time_f400_loo, poly_model_5_loo.predict(test_year_f400_loo)))

formatted_result_5 = ["%.3f" % item for item in results_5]
pred_time_f400_all_5_loo = poly_model_5_loo.predict(train_year_female400)
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_f400_all_5_loo, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 8 - Use LOOCV for 5th order polynomial')
plt.show()
print("Mean square Error - Female400 - Degree 5 - LOOCV : %.3f" % mean_squared_error(train_time_female400, pred_time_f400_all_5_loo))
print("LOOCV - Test split - MSD - Degree 3 :", formatted_result_3)
print("LOOCV - Test split - MSD - Degree 5 :", formatted_result_5)
print("As seen from the error values, "
      "degree 3 polynomial fits the female400 slightly better than the degree 5")

# Experiment 9 - Use sklearn.linear_model.Ridge with alpha - 0.1
# to fit a 5th order polynomial to female400
# Reference - https://stackoverflow.com/questions/34373606/scikit-learn-coefficients-polynomialfeatures

print('\nExperiment 9 - Use sklearn.linear_model.Ridge with alpha - 0.1')
poly_model_5_ridge = make_pipeline(PolynomialFeatures(degree=5), Ridge(alpha=.1))
poly_model_5_ridge.fit(train_year_female400, train_time_female400)
pred_time_f400_all_5_r = poly_model_5_ridge.predict(train_year_female400)
print('Linear Coefficients : ', np.around(regr.coef_, decimals=4))
print('Ridge Coefficients : ', np.around(poly_model_5_ridge.steps[1][1].coef_, decimals=4))
print('Linear Intercept :', np.around(regr.intercept_, decimals=4))
print('Ridge Intercept :', np.around(poly_model_5_ridge.steps[1][1].intercept_, decimals=4))
# Plot to check fit
plt.scatter(train_year_female400, train_time_female400, c='r')
#plt.plot(train_year_female400, pred_time_female400_all_5, c='g')
plt.plot(train_year_female400, pred_time_f400_all_5_r, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 9 - Use sklearn.linear_model.Ridge with alpha - 0.1')
plt.show()
print("Mean square Error - Female400 - Degree 5 - Ridge : %.3f" % mean_squared_error(train_time_female400, pred_time_f400_all_5_r))

# Experiment 10 - Use sklearn.linear_model.RidgeCV to find the best
# value of alpha across the range - 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0
# Reference - http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

print('\nExperiment 10 - Use sklearn.linear_model.RidgeCV to find the best alpha across a range')
ridge = RidgeCV(alphas=[0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0])
ridge.fit(train_year_female400, train_time_female400)
pred_time_f400_all_r_cv = ridge.predict(train_year_female400)
print(ridge)
print('Estimated Regularization parameter :', ridge.alpha_)
plt.scatter(train_year_female400, train_time_female400, c='r')
plt.plot(train_year_female400, pred_time_f400_all_r_cv, c='b')
plt.xlabel('Year')
plt.ylabel('Time(seconds)')
plt.title('Exp 10 - Use sklearn.linear_model.RidgeCV\n to find the best alpha across a range')
plt.show()
