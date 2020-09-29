#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author
import numpy as np


# Do not modify this class
class Param:
  """
  Simple class to hold parameters
  """

  def __init__(self, shape):
    super().__init__()
    self.value = np.zeros(shape)


# Do not modify this class
class Optimizer:
  """
  Abstract optimizer implementation
  """

  def reset(self, params):
    raise NotImplementedError("Must override reset method")

  def update(self, params, gradients):
    raise NotImplementedError("Must override update method")


# Do not modify this class
class BatchGradientDescendOptimizer(Optimizer):
  """
  Basic gradient descend optimizer (better known as batch gradient descend).
  """

  def __init__(self, **opts):
    self.alpha = 1e-3 if 'alpha' not in opts else opts['alpha']

  def reset(self, params):
    pass

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      step = self.alpha * grad
      param.value -= step


# Do not modify this class
class AdagradOptimizer(Optimizer):
  """
  Adagrad optimizer. It uses an adaptative learning rate per component of the gradient. The learning rate is
  adjusted based on the cumulative gradient from previous iterations (each component treated independently).
  """

  def __init__(self, **opts):
    self.alpha = 1 if 'alpha' not in opts else opts['alpha']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.cumulative = {}

  def reset(self, params):
    for param in params:
      self.cumulative[param] = np.zeros((param.value.shape))

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      self.cumulative[param] += (grad ** 2)
      step = self.alpha / (np.sqrt(self.cumulative[param]) + self.epsilon) * grad
      param.value -= step


# You must complete this implementation
class MomentumOptimizer(Optimizer):
  """
  Gradient descend with momentum (using an exponential weighted average of past gradients).
  """

  def __init__(self, **opts):
    self.alpha = 1e-3 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.bias = False if 'bias' not in opts else opts['bias']
    self.Vdw = {}
    self.t = {}

  def reset(self, params):
    for param in params:
      self.Vdw[param] = np.zeros((param.value.shape))
      self.t = np.zeros((param.value.shape))

  def update(self, params, gradients):
    self.t += 1
    for param, grad in zip(params, gradients):
      self.Vdw[param] *= self.beta
      self.Vdw[param] += (1 - self.beta) * grad
      if self.bias is True:
        param.value -= self.alpha * (self.Vdw[param] / (1 - pow(self.beta, self.t)))        
      else:
        param.value -= self.alpha * self.Vdw[param]


# You must complete this implementation
class RMSPropOptimizer(Optimizer):
  """
  RMSProp optimizer taking advantage of the idea in Adagrad but using an exponentially
  weighted average of the squared of past gradient instead.
  """

  def __init__(self, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.bias = False if 'bias' not in opts else opts['bias']
    self.decay = 0. if 'decay' not in opts else opts['decay']
    self.Sdw = {}
    self.t = {}
    self.alphaprime = {}

  def reset(self, params):
    for param in params:
      self.Sdw[param] = np.zeros((param.value.shape))
      self.t = np.zeros((param.value.shape))
      self.alphaprime = np.zeros((param.value.shape))

  def update(self, params, gradients):
    self.t += 1
    self.alphaprime = self.alpha/(1 + (self.decay*self.t))
    for param, grad in zip(params, gradients):
      self.Sdw[param] *= self.beta
      self.Sdw[param] += (1-self.beta) * (grad**2)
      if self.bias is True:
        param.value -= (self.alphaprime/(np.sqrt(self.Sdw[param]/(1-pow(self.beta, self.t)))+self.epsilon))*grad
      else:
        param.value -= (self.alphaprime/(np.sqrt(self.Sdw[param])+self.epsilon))*grad


# You must complete this implementation
class AdamOptimizer(Optimizer):
  """
  Adam optimizer that uses momentum and the idea in Adagrad but using an exponentially
  weighted average of the squared of past gradient instead (as in RMSProp)
  """

  def __init__(self, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta1 = 0.9 if 'beta1' not in opts else opts['beta1']
    self.beta2 = 0.999 if 'beta2' not in opts else opts['beta2']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.bias = True if 'bias' not in opts else opts['bias']
    self.decay = 0 if 'decay' not in opts else opts['decay']
    self.Vdw = {}
    self.Sdw = {}
    self.t = {}
    self.alphaprime = {}

  def reset(self, params):
    for param in params:
      self.Vdw[param] = np.zeros((param.value.shape))
      self.Sdw[param] = np.zeros((param.value.shape))
      self.t = np.zeros((param.value.shape))
      self.alphaprime = np.zeros((param.value.shape))

  def update(self, params, gradients):
    self.t += 1
    self.alphaprime = self.alpha/(1 + (self.decay*self.t))
    for param, grad in zip(params, gradients):
      self.Vdw[param] *= self.beta1
      self.Vdw[param] += (1-self.beta1) * grad
      self.Sdw[param] *= self.beta2
      self.Sdw[param] += (1-self.beta2) * (grad**2)
      if self.bias is True:
       param.value -= self.alphaprime*((self.Vdw[param]/(1-pow(self.beta1, self.t)))/(np.sqrt(self.Sdw[param]/(1-pow(self.beta2, self.t)))+self.epsilon))
      else:
       param.value -= self.alphaprime*(self.Vdw[param]/(np.sqrt(self.Sdw[param])+self.epsilon))


# Do not modify this function
def train(starting_point, optimizers, cost_func, grad_func, epochs=200, epsilon=1e-7):
  # Creates x and y params
  x = Param((1,))
  y = Param((1,))
  # We store the results of the trainings here
  results = {}
  # Loops through each optimizer
  for optimizer in optimizers:
    # Reset params to initial values
    x.value = np.array(starting_point[0], dtype=np.float32)
    y.value = np.array(starting_point[1], dtype=np.float32)
    optimizer.reset([x, y])
    # Computes and stores the cost for the initial params
    history = []
    J = cost_func([x.value, y.value])
    history.append(((x.value.copy(), y.value.copy()), J))
    # Training loop
    delta = epsilon
    for epoch in range(epochs):
      # Computes gradients and updates params
      gradient = grad_func([x.value, y.value])
      optimizer.update([x, y], gradient)
      # Computes new cost after params update
      J = cost_func([x.value, y.value])
      if np.isinf(J) or np.isnan(J):
        break
      history.append(((x.value.copy(), y.value.copy()), J))
      delta = history[-2][1] - history[-1][1]
      # Check whether cost function changed enough
      if abs(delta) < epsilon:
        break
    # Logs message after training with optimizer
    print('Cost {} after {} steps (with {})'.format(
      history[-1][1],
      len(history) - 1,
      type(optimizer).__name__))
    # Stores results
    results[optimizer] = history

  # Outputs results for all optimizers
  return results
