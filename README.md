# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.
    
 
 

## Program:
```

Program to implement the linear regression using gradient descent.

Developed by: Divyashree B S

RegisterNumber:  212221040044

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (1).txt",header=None)

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning ] 
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

print("Compute cost value:")
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent graph:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit prediction graph:")
def predict(x,theta):
  """
  Take in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the population 35,000:")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the population 70,000:")
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

Profit prediction graph:

<img width="435" alt="mlex3op1" src="https://user-images.githubusercontent.com/127508123/229680659-0cdcfce5-6a96-4468-a2b7-58fc3247ff75.png">

Compute cost value:

<img width="132" alt="mlex32op" src="https://user-images.githubusercontent.com/127508123/229681197-d31815ab-6bb6-406b-bc4e-48850f733eac.png">

h(x) value:

<img width="165" alt="mlex33op" src="https://user-images.githubusercontent.com/127508123/229681253-6dbc7055-200c-4cee-97b8-a55b204f08fb.png">

Cost function using Gradient Descent graph:

<img width="431" alt="mlex34op" src="https://user-images.githubusercontent.com/127508123/229681279-0f940979-0119-4801-a17c-4ba659023ca9.png">

Profit prediction graph:

<img width="626" alt="mlex35op" src="https://user-images.githubusercontent.com/127508123/229681298-3e33cdfe-8851-4750-a6c3-d3787b90c661.png">

Profit for the population 35,000:

<img width="346" alt="mlex36op" src="https://user-images.githubusercontent.com/127508123/229681325-0a996b2c-8ee5-4692-9184-5b3cc3956ff0.png">

Profit for the population 70,000:

<img width="339" alt="mlex37op" src="https://user-images.githubusercontent.com/127508123/229681353-0b5bb232-07f5-44c4-ae93-1a2d810d27fd.png">

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
