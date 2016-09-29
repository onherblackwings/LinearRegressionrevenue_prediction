#python linear regression tutorial reference from:
#http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html


from numpy import loadtxt, zeros, ones, array, linspace, logspace
#from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np
import os

os.system('cls')
#Evaluate the linear regression
def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

def user_input(percentage):
        predict1 = array([1, percentage]).dot(theta).flatten()
        return format(float(predict1*100),',.2f')



#Load the dataset
data = loadtxt('maribago.txt', delimiter=',')

#Plot the data
'''scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Revenue distribution per occupancy %')
xlabel('Occupancy Percentage')
ylabel('Revenue')
#show()'''

X = data[:, 0]
y = data[:, 1]


#number of training samples
m = y.size

#Add a column of ones to X (interception data)
it = ones(shape=(m, 2))
it[:, 1] = X

#Initialize theta parameters
theta = zeros(shape=(2, 1))

#Some gradient descent settings
iterations = 1500
alpha = 0.01

#compute and display initial cost
print "Initial cost with theta (0,0): ", compute_cost(it, y, theta)

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print "theta found by gradient descent: \n ", theta
print "\n"
#Predict values for percentages of 85 percent and 70 percent
print "sample predictions: "
predict1 = (array([1, .85]).dot(theta).flatten())*100
print 'For occupancy = 85 percent, we predict a revenue of PHP', format(float(predict1),',.2f')
predict2 = (array([1, .70]).dot(theta).flatten())*100
print 'For occupancy = 70 percent, we predict a revenue of PHP', format(float(predict2),',.2f')
print "\n",
value_percentage=input("Enter a value from 1-100: ")
if value_percentage<1 or value_percentage>100:
        print "The value is out of range."
        os.system('pause')
        exit()
else:
        percentage=float(value_percentage)/100
print "occupancy percentage: ",format(float(percentage*100),',.2f'),'percent'
print 'revenue= ',user_input(percentage)
os.system('pause')
'''#Plot the results
result = it.dot(theta).flatten()
plot(data[:, 0], result)
show()


#Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))'''





#Fill out J_vals
'''for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(it, y, thetaT)'''

#Contour plot
'''J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()'''
