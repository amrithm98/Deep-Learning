from numpy import *
import matplotlib.pyplot as plt
def compute_error_for_given_points(b,m,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError+=(y-(m*x+b))**2  #SSR
    return totalError/float(len(points))
def step_gradient(b_current,m_current,points,learning_rate):
    b_gradient=0
    m_gradient=0
    N=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=-(2/N)*(y-((m_current*x)+b_current))
        m_gradient+=-(2/N)*x*(y-((m_current*x)+b_current))
    new_b=b_current-(learning_rate*b_gradient)
    new_m=m_current-(learning_rate*m_gradient)
    return [new_b,new_m]
def gradeient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b=starting_b
    m=starting_m
    x_values=points[:,0]
    y_values=points[:,1]
    for i in range(num_iterations):
        b,m=step_gradient(b,m,array(points),learning_rate)
        abline_values = [m * i + b for i in x_values]
        ln,=plt.plot(x_values, abline_values)
        print m,b
        plt.pause(0.05)
        ln.remove()
    return [b,m]
def run():
    points=genfromtxt('data.csv',delimiter=',')
    learning_rate=0.0001    #hyper param
    #y=mx+b
    plt.ion()
    plt.axis([0,100,0,100])
    x_values=points[:,0]
    y_values=points[:,1]
    plt.scatter(x_values,y_values)
    initial_b=0
    initial_m=0
    num_iterations=1000
    [b,m]=gradeient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)
    print(b)
    print(m)
if __name__=='__main__':
    run()