# -*- coding: utf-8 -*-
"""

"""

from _perceptron import (
    Perceptron, 
    plot_plane, 
    plot_scatter, 
    draw_line)

from matplotlib import pyplot as plt
import numpy

numpy.random.seed(0)
# ### Defining our dataset

features = numpy.array([
     [0,2],
     [1,0],[1,1],[1,2],[1,3],
     [2,2],[2,3],[3,2]
    ])
labels = numpy.array([0,0,0,0,1,1,1,1])

data = numpy.column_stack((features, labels))

# # Plotting the points

# We only need the full array becasue matplotlib doesn't support passing in multiple markers
# We're also giving our guess for a splitting line.
plot_scatter(data[labels == 0,0], data[labels == 0,1], marker = '^')
plot_scatter(data[labels == 1,0], data[labels == 1,1], marker = 's')
plt.legend(["Happy", "Sad"])
draw_line(-1,3.5, ending = 3)
plt.show()

instance_perceptron = Perceptron()
weights, bias, errors = instance_perceptron.perceptron_algorithm(
    features, labels, learning_rate=0.01, num_epochs = 200)

# The weights we get are the coefficients of the line (plane)
# In our case, the weights and bias describe the line Ax + By + C = 0

plot_scatter(data[labels == 0,0], data[labels == 0,1], marker = '^')
plot_scatter(data[labels == 1,0], data[labels == 1,1], marker = 's')
draw_line(
    -weights[0]/weights[1],
    -bias/weights[1],
    ending = 3,
    color='grey', linewidth=1.0, linestyle='dotted')
plt.legend(["Best fit line", "Happy", "Sad"])
plt.show()

x = [numpy.random.randint(0,10, size = 3) for i in range(5)]
x = numpy.array(x)
y = [numpy.random.randint(0,2) for i in range(5)]
y = numpy.array(y)
w, b, e = instance_perceptron.perceptron_algorithm(x,y, learning_rate = 0.01, num_epochs=200)
ax = plot_plane(w,b)
ax.scatter(x[:,0], x[:,1], x[:,2], c = y)
plt.show()

# import turicreate as tc

# datadict = {'aack': features[:,0], 'beep':features[:,1], 'prediction': labels}
# data = tc.SFrame(datadict)
# data

# perceptron = tc.logistic_classifier.create(data, target='prediction')


# perceptron.coefficients