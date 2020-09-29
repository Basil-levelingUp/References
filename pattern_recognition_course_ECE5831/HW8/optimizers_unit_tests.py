#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author
import numpy as np

# Do not modify anything from this file!
IMPORT_SOLUTION = False
try:
  if IMPORT_SOLUTION:
    optimizers = __import__('optimizers_solution')
  else:
    optimizers = __import__('optimizers_starter')
except ImportError:
  optimizers = __import__('optimizers_starter')

# Importing optimizers
Param = optimizers.Param
MomentumOptimizer = optimizers.MomentumOptimizer
RMSPropOptimizer = optimizers.RMSPropOptimizer
AdamOptimizer = optimizers.AdamOptimizer


def momentum_update_default_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = MomentumOptimizer(alpha=1e-1, beta=0.95, bias=False)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-2.8667494, -0.8556245]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-7.11124901, -2.91124901]) < [1e-7, 1e-7]).all()
  return correct


def momentum_update_with_bias_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = MomentumOptimizer(alpha=1e-1, beta=0.92, bias=True)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-4., -1.8]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-9., -4.8]) < [1e-7, 1e-7]).all()
  return correct


def momentum_reset_unit_test():
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = MomentumOptimizer(alpha=1e-1, beta=0.85, bias=True)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Stores results for comparison
  result1 = param1.value.copy()
  result2 = param2.value.copy()
  # Resets params to their initial value
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  # Resets optimizer
  optimizer.reset([param1, param2])
  # This should repeat the same process as before
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies reset method
  correct = True
  correct = correct & (param1.value != [2., 3.2]).all()
  correct = correct & (param2.value != [1., 5.2]).all()
  correct = correct & (abs(result1 - param1.value) < [1e-7, 1e-7]).all()
  correct = correct & (abs(result2 - param2.value) < [1e-7, 1e-7]).all()
  return correct


def rmsprop_update_default_unit_test():
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = RMSPropOptimizer(alpha=1e-1, beta=0.95, bias=False, decay=0)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-10.10106298, -8.90106298]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-11.10106298, -6.90106298]) < [1e-7, 1e-7]).all()
  return correct


def rmsprop_update_with_bias_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = RMSPropOptimizer(alpha=1e-1, beta=0.92, bias=True, decay=0)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-8., -6.8]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-9., -4.8]) < [1e-7, 1e-7]).all()
  return correct


def rmsprop_update_with_bias_decay_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = RMSPropOptimizer(alpha=1e-1, beta=0.92, bias=True, decay=0.1)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-0.35326634, 0.84673366]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-1.35326634, 2.84673366]) < [1e-7, 1e-7]).all()
  return correct


def rmsprop_reset_unit_test():
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = RMSPropOptimizer(alpha=1e-1, beta=0.85, bias=True, decay=0.1)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Stores results for comparison
  result1 = param1.value.copy()
  result2 = param2.value.copy()
  # Resets params to their initial value
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  # Resets optimizer
  optimizer.reset([param1, param2])
  # This should repeat the same process as before
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies reset method
  correct = True
  correct = correct & (param1.value != [2., 3.2]).all()
  correct = correct & (param2.value != [1., 5.2]).all()
  correct = correct & (abs(result1 - param1.value) < [1e-7, 1e-7]).all()
  correct = correct & (abs(result2 - param2.value) < [1e-7, 1e-7]).all()
  return correct


def adam_update_default_unit_test():
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = AdamOptimizer(alpha=1e-1, beta1=0.95, beta2=0.80, bias=False, decay=0)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-6.16878622, -4.96878622]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-7.16878622, -2.96878622]) < [1e-7, 1e-7]).all()
  return correct


def adam_update_with_bias_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = AdamOptimizer(alpha=1e-1, beta1=0.92, beta2=0.999, bias=True, decay=0)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-8., -6.8]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-9., -4.8]) < [1e-7, 1e-7]).all()
  return correct


def adam_update_with_bias_decay_unit_test() -> bool:
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = AdamOptimizer(alpha=1e-1, beta1=0.92, beta2=0.89, bias=True, decay=0.1)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies update method
  correct = True
  correct = correct & (abs(param1.value - [-0.35326634, 0.84673366]) < [1e-7, 1e-7]).all()
  correct = correct & (abs(param2.value - [-1.35326634, 2.84673366]) < [1e-7, 1e-7]).all()
  return correct


def adam_reset_unit_test():
  # Creates params
  param1 = Param((2,))
  param2 = Param((2,))
  # Gradients (we won't change this)
  grad1 = np.array([0.6, 0.5])
  grad2 = np.array([1., 1.])
  # Creates optimizer
  optimizer = AdamOptimizer(alpha=1e-1, beta1=0.85, beta2=0.99, bias=True, decay=0.1)
  # Sets values and inits optimizer
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  optimizer.reset([param1, param2])
  # Updates params using optimizer
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Stores results for comparison
  result1 = param1.value.copy()
  result2 = param2.value.copy()
  # Resets params to their initial value
  param1.value = np.array([2., 3.2])
  param2.value = np.array([1., 5.2])
  # Resets optimizer
  optimizer.reset([param1, param2])
  # This should repeat the same process as before
  for i in range(100):
    optimizer.update([param1, param2], [grad1, grad2])
  # Verifies reset method
  correct = True
  correct = correct & (param1.value != [2., 3.2]).all()
  correct = correct & (param2.value != [1., 5.2]).all()
  correct = correct & (abs(result1 - param1.value) < [1e-7, 1e-7]).all()
  correct = correct & (abs(result2 - param2.value) < [1e-7, 1e-7]).all()
  return correct


if __name__ == '__main__':
  # Momentum Optimizer
  print('---- Momentum -----')
  print('Verifying update method (without bias correction): ', end='')
  print('Passed') if momentum_update_default_unit_test() else print('Failed')
  print("Verifying update method (with bias correction): ", end='')
  print('Passed') if momentum_update_with_bias_unit_test() else print('Failed')
  print("Verifying reset method: ", end='')
  print('Passed') if momentum_reset_unit_test() else print('Failed')
  # RMSProp Optimizer
  print('---- RMSProp -----')
  print("Verifying update method (without bias correction or decay): ", end='')
  print('Passed') if rmsprop_update_default_unit_test() else print('Failed')
  print('Verifying update method (with bias correction): ', end='')
  print('Passed') if rmsprop_update_with_bias_unit_test() else print('Failed')
  print('Verifying update method (with bias correction and decay): ', end='')
  print('Passed') if rmsprop_update_with_bias_decay_unit_test() else print('Failed')
  print("Verifying reset method: ", end='')
  print('Passed') if rmsprop_reset_unit_test() else print('Failed')
  # Adam Optimizer
  print('----- Adam -----')
  print("Verifying update method (without bias correction or decay): ", end='')
  print('Passed') if adam_update_default_unit_test() else print('Failed')
  print('Verifying update method (with bias correction): ', end='')
  print('Passed') if adam_update_with_bias_unit_test() else print('Failed')
  print('Verifying update method (with bias correction and decay): ', end='')
  print('Passed') if adam_update_with_bias_decay_unit_test() else print('Failed')
  print("Verifying reset method: ", end='')
  print('Passed') if adam_reset_unit_test() else print('Failed')
