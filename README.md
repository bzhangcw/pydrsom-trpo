# DRSOM-for-RL


One need to add 

'''python
def compute_alpha(self, **closure):
    return self._optimizer.compute_alpha(**closure)
'''
in the optimizer_wrapper.py of the package garage