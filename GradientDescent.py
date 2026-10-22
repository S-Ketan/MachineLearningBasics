import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000
    n=len(x)
    learning_rate=0.013
    for i in range(iterations):
        y_predicted=m_curr*x + b_curr
        cost=1/n*np.sum((y-y_predicted)**2)
        md=-(2/n)*np.sum(x*(y-y_predicted))
        bd=-(2/n)*np.sum(y-y_predicted)
        m_curr=m_curr-(learning_rate*md)
        b_curr=b_curr-(learning_rate*bd)
        print("M {}, B {}, cost{} , iteration{}".format(m_curr,b_curr,cost,i))


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x,y, marker='+', color='red')
plt.show()

gradient_descent(x,y)
