
import torch
import numpy as np

class Rosenbrock:
    def torch_loss(self, x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def test_convergence(opt_class, name, lr, max_iter=20000, **kwargs):
    start_point = (-1.5, 2.0)
    rosenbrock = Rosenbrock()
    
    x = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)
    optimizer = opt_class([x], lr=lr, **kwargs)
    
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = rosenbrock.torch_loss(x)
        loss.backward()
        optimizer.step()
        
        if loss.item() < 1e-4:
            print(f"{name} (lr={lr}, {kwargs}) converged at iteration {i} with loss {loss.item()}")
            return True
            
    print(f"{name} (lr={lr}, {kwargs}) did not converge in {max_iter} iterations. Final loss: {loss.item()}")
    return False

if __name__ == "__main__":
    # Test Adam with new LR
    test_convergence(torch.optim.Adam, "Adam", 0.05)
    # Test SGD with different LRs
    test_convergence(torch.optim.SGD, "SGD", 0.01)
    test_convergence(torch.optim.SGD, "SGD", 0.001)
    test_convergence(torch.optim.SGD, "SGD_Momentum", 0.01, momentum=0.9)
    test_convergence(torch.optim.SGD, "SGD_Momentum", 0.001, momentum=0.9)
