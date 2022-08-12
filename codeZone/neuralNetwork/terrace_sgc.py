# Chandler King
# terrace sgc

import torch
from torch.nn import Module, Linear

from simple_gcn import mat_power

class terrace_sgc(Module):

    def __init__(self, in_feat, out_feat, k=2):
        super().__init__()

        # Applies a linear transformation to y = W(theta)A^k + b
        # in_features – size of each input sample
        # out_features – size of each output sample
        # We assign the bias here as well??
        self.support = Linear(in_feat, out_feat, bias=True)
        self.k = k
        self.gather_type = gather_type

    def terrace(self, X, A=None):
        # TODO: mat_power_stack needs anti-bounce features.
        # TODO: if X and A are not batches, this will give unexpected results
        # TODO: X may be a dictionary

        Y = mat_power_stack(A, self.k) @ X.unsqueeze(1) @ self.W.unsqueeze(0)
        return Y
        # note: self.W needs to be initialized as a Parameter

        # LINEAR implementation
        self.layer = torch.nn.Linear(d, f)
        for layer in self.layers:
            Y.append(mat_power_stack(A, self.k) @ layer(X.unsqueeze(1)))
        return Y.sum(1)

    # For each of k there will be A^k XO vk
    # b is added at the end I dont think I neccesarily have to worry about that

    def gather(self, x):
        if self.gather_type == "mean":
            # returns the mean value of all the elements in input tensor
            return torch.mean(x, dim=0)
        if self.gather_type == 'sum':
            # Returns the sum of all elements in the input tensor
            return torch.sum(x, dim=0)
        else:
            raise ValueError(f'{self.gather_type} is not a valid gather_type.')

    def forward(self):
        # sets x to equal the linear transformation and removes
        # single dimensional entries
        x = self.support(x).squeeze()
        # Matrix product of two tensors between x and A^k

        # Is this where I would change the coding in order to go up to k with
        # the summation

        x = torch.matmul(mat_power(A, self.k), x)
        # runs def gather(with x)
        return self.gather(x)


def mat_power_stack(M, k):
    """Computes all powers of M from 0 to k. Return these results as a tensor
    stack along dim -3."""
    stack = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    if len(stack.shape) < len(M.shape):
        stack.unsqueeze(-1)
        stack = stack.repeat(M.shape[0], 1, 1)
    stack = [stack]

    A = M
    while k > 0:
        stack.append(A)
        k -= 1
        A = torch.matmul(A, M)
    return torch.stack(stack, dim=-3)

function = '''

'''Notes for implementing the terrace layer:'''
def terrace(self, X, A=None):
    # TODO: mat_power_stack needs anti-bounce features.
    # TODO: if X and A are not batches, this will give unexpected results
    # TODO: X may be a dictionary

    Y = mat_power_stack(A, self.k) @ X.unsqueeze(1) @ self.W.unsqueeze(0)
    # note: self.W needs to be initialized as a Parameter

    # LINEAR implementation
    self.layer = torch.nn.Linear(d, f)
    for layer in self.layers:
        Y.append(mat_power_stack(A, self.k) @ layer(X.unsqueeze(1)))
    return Y.sum(1)

def mat_power(M, K):
    """Computes M^k"""
    if K == 0:
        # This is a tensor with this pattern. Would be equal to just finding the
        # dimensionality and multipling a matrix by 1
        # | 1 0 0 |
        # | 0 1 0 |
        # | 0 0 1 |
        # Can specify the rows and columns as well as datatype

        return torch.eye(len(M), dtype=M.dtype)
    result = M
    while K > 1:
        K -= 1
        result = torch.matmul(result, M)
    return result

'''

explanation = '''
Heres what I need to do:
Train
save
load
etc
'''

linear_alg = '''

# Multiply two arrays
x = [1,2,3]
y = [2,3,4]
product = []
for i in range(len(x)):
    product.append(x[i]*y[i])

# Linear algebra version
x = numpy.array([1,2,3])
y = numpy.array([2,3,4])
x * y

Vectors are 1-dimensional arrays of numbers or terms.
In geometry, vectors store the magnitude and direction of a potential change
to a point. The vector [3, -2] says go right 3 and down 2. A vector with more
than one dimension is called a matrix.

y = np.array([1,2,3])
x = np.array([2,3,4])
y + x = [3, 5, 7]
y - x = [-1, -1, -1]
y / x = [.5, .67, .75]

y = np.array([1,2,3])
x = np.array([2,3,4])
np.dot(y,x) = 20

y = np.array([1,2,3])
x = np.array([2,3,4])
y * x = [2, 6, 12]
'''
