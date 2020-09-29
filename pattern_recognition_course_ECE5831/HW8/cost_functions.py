#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author
from typing import Callable, Tuple
import numpy as np
import autograd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


def beale(inputs):
  """
  Computes Beale's function
  :param inputs: 2D iterable containing inputs
  """
  x = inputs[0]
  y = inputs[1]
  return (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2


def booth(inputs):
  """
  Computes Booth's function
  :param inputs: 2D iterable containing inputs
  """
  x = inputs[0]
  y = inputs[1]
  return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def goldstein(inputs):
  """
  Computes Goldstein's function
  :param inputs: 2D iterable containing inputs
  """
  x = inputs[0]
  y = inputs[1]
  return (1 + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
      30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


def himmelblau(inputs):
  """
  Computes Himmelblau's function
  :param inputs: 2D iterable containing inputs
  """
  x = inputs[0]
  y = inputs[1]
  return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def matyas(inputs):
  """
  Computes Matyas' function
  :param inputs: 2D iterable containing inputs
  """
  x = inputs[0]
  y = inputs[1]
  return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


# Helper functions to compute the gradient of the functions defined above
gradient_beale = autograd.grad(beale)
gradient_booth = autograd.grad(booth)
gradient_goldstein = autograd.grad(goldstein)
gradient_himmelblau = autograd.grad(himmelblau)
gradient_matyas = autograd.grad(matyas)


def approximate_gradient(f: np.ndarray, spacing: float) -> Tuple:
  """
  Computes the numerical gradient for a 2D function using the central difference for interior data points (Finite
  differences).
  :param f: Function values
  :param spacing: Time step
  :return: 2D Gradient vector as a tuple
  """
  nx = f.shape[0]
  ny = f.shape[1]
  dx = np.zeros((nx, ny))
  dy = np.zeros((nx, ny))

  if nx < 2 or ny < 2:
    return dx, dy

  for i in range(1, nx - 1):
    dy[i, :] = (f[i + 1, :] - f[i - 1, :]) / (2 * spacing)
  dy[0, :] = (f[1, :] - f[0, :]) / (2 * spacing)
  dy[nx - 1, :] = (f[nx - 1, :] - f[nx - 2, :]) / (2 * spacing)

  for j in range(1, ny - 1):
    dx[:, j] = (f[:, j + 1] - f[:, j - 1]) / (2 * spacing)
  dx[:, 0] = (f[:, 1] - f[:, 0]) / (2 * spacing)
  dx[:, ny - 1] = (f[:, ny - 1] - f[:, ny - 2]) / (2 * spacing)

  return dx, dy


def graph_function(function: Callable, xrange=None, yrange=None, gstep=2, title=None) -> None:
  """
  Graph a 2D function using a 3D surface and contour plot that includes its gradient depiction (quiver plot)
  :param function: Callable function that receives an iterable of two inputs
  :param xrange: Range in the x axis to be evaluate
  :param yrange: Range in the y axis to be evaluate
  :param gstep: Step to plot gradient vectors
  :param title: Title for this function
  """
  if yrange is None:
    yrange = [-8, 8]
  if xrange is None:
    xrange = [-8, 8]
  # Building mesh
  pts = 50
  x = np.linspace(xrange[0], xrange[1], pts)
  y = np.linspace(yrange[0], yrange[1], pts)
  [x, y] = np.meshgrid(x, y)
  z = function([x, y])
  spacing = (x.max() - x.min()) / pts
  # Creating figure
  fig = plt.figure(figsize=(16, 5))
  # Surface plot
  ax = fig.add_subplot(121, projection='3d')
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  dx, dy = approximate_gradient(z, spacing)
  ax.plot_surface(x, y, z, cmap='viridis', norm=LogNorm())
  if title is not None:
    ax.set_title(title + ': Function')
  # Contour plot
  ax = fig.add_subplot(122)
  ax.contour(x, y, z, levels=np.logspace(0, 5, 10), norm=LogNorm(), alpha=0.4)
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  if title is not None:
    ax.set_title(title + ': Gradient')
  ax.quiver(x[::gstep, ::gstep], y[::gstep, ::gstep], -dx[::gstep, ::gstep], -dy[::gstep, ::gstep])
  ax.grid()
