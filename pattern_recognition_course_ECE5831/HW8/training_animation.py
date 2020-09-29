#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cost_functions import approximate_gradient
from IPython.display import display, clear_output
import keyboard

# Names for legends
shortnames = {
  'BatchGradientDescendOptimizer': "GD",
  'AdagradOptimizer': "Adagrad",
  'MomentumOptimizer': "GD w/ Momentum",
  'RMSPropOptimizer': "RMSProp",
  'AdamOptimizer': 'Adam'
}


def maximum_steps(data):
  return max([len(data[optimizer]) for optimizer in data])


def maximum_cost(data):
  return max([entry[1] for optimizer in data for entry in data[optimizer]])


def minimum_cost(data):
  return min([entry[1] for optimizer in data for entry in data[optimizer]])


def minimum_x(data):
  return min([entry[0][0] for optimizer in data for entry in data[optimizer]])


def maximum_x(data):
  return max([entry[0][0] for optimizer in data for entry in data[optimizer]])


def minimum_y(data):
  return min([entry[0][1] for optimizer in data for entry in data[optimizer]])


def maximum_y(data):
  return max([entry[0][1] for optimizer in data for entry in data[optimizer]])


class TrainingAnimation:
  """
  Helper class to generate training animation provided you pass proper training data.
  """
  def __init__(self, data: dict, cost_func: Callable, grad_func: Callable):
    """
    Creates object

    :param data: Dictionary where key is an optimizer and the value is the training data (format shown in the training
    loop on the notebook)
    :param cost_func: Callable to compute the cost function
    :param grad_func: Callable to compute the gradient of the cost function
    """
    self.data = data
    self.cost_func = cost_func
    self.grad_func = grad_func
    self.costs = None
    self.paths = None

  # Plots the cost graph and returns a dictionary of artists where keys are optimizers
  def _plot_cost(self, ax):
    artists = {}
    for optimizer in self.data:
      line, = ax.plot([], [], '-', label=shortnames[type(optimizer).__name__])
      artists[optimizer] = line
    ax.set_title('Cost function $J(x,y)$')
    ax.set_xlabel('Step')
    ax.grid()
    ax.set_xlim([0, maximum_steps(self.data)])
    ax.set_ylim([minimum_cost(self.data), maximum_cost(self.data)])
    return artists

  # Plots the contour graph and returns a dictionary of artists where keys are optimizers
  def _plot_contour(self, ax):
    pts = 50
    expansion = 1
    x = np.linspace(minimum_x(self.data) - expansion, maximum_x(self.data) + expansion, pts)
    y = np.linspace(minimum_y(self.data) - expansion, maximum_y(self.data) + expansion, pts)
    [x, y] = np.meshgrid(x, y)
    spacing = (x.max() - x.min()) / pts
    z = self.cost_func([x, y])
    dx, dy = approximate_gradient(z, spacing)
    ax.contour(x, y, z, levels=np.logspace(0, 5, 10), norm=LogNorm(), alpha=0.4)
    gstep = 2
    ax.quiver(x[::gstep, ::gstep], y[::gstep, ::gstep], -dx[::gstep, ::gstep], -dy[::gstep, ::gstep], alpha=0.3)
    artists = {}
    for optimizer in self.data:
      line, = ax.plot([], [], '-')
      artists[optimizer] = line
    x = self.data[optimizer][0][0][0]
    y = self.data[optimizer][0][0][1]
    ax.plot(x, y, 'k.', label='Initial Params', markersize=15)
    ax.set_title('Cost function $J(x,y)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.grid()
    return artists

  # Updates data of artists from data available
  def _plot_frame(self, step):
    for optimizer in self.data:
      history = self.data[optimizer]
      cost = self.costs[optimizer]
      path = self.paths[optimizer]
      index = min(step + 1, len(history))
      # Cost
      xdata = range(0, index)
      ydata = [entry[1] for entry in history[:index]]
      cost.set_data(xdata, ydata)
      # 2D Path
      xdata = [entry[0][0] for entry in history[:index]]
      ydata = [entry[0][1] for entry in history[:index]]
      path.set_data(xdata, ydata)

  def start(self):
    """
    Starts animation. Shows a frame per training step.
    """
    # Creates figure
    fig = plt.figure(figsize=(15.2, 4))
    fig.subplots_adjust(top=1.3, bottom=0.4)
    # Creates axes
    ax_cost = fig.add_subplot(121)
    ax_contour = fig.add_subplot(122)
    # Plot static data and gets artists references for updating data
    self.costs = self._plot_cost(ax_cost)
    self.paths = self._plot_contour(ax_contour)
    # Performs animation by updating data
    self._plot_frame(0)
    fig.legend(loc=8)
    for i in range(1, maximum_steps(self.data)):
      self._plot_frame(i)
      display(fig)
      clear_output(wait=True)
      plt.pause(0.01)
      if keyboard.is_pressed('Esc'):
        self._plot_frame(maximum_steps(self.data) - 1)
        display(fig)
        break
