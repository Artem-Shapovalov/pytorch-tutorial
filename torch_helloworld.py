import torch # Attaching the torch library

x = torch.tensor([1.0, 2.0, 3.0]) # Input tensor
y_target = torch.tensor(14.0)     # Desirable values of the output
lr = 0.01                         # Learning rate

# Weights tensor.
# requires_grad means this tensor would have inner stuff to calculate and
# store the gradient.
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

for step in range(20):
    # Forward pass
    y = (x * w).sum()          # Outputs tensor
    loss = (y - y_target) ** 2 # Mean squared error loss
    loss.backward()            # Compute the gradient or error, creates w.grad

    # Learning
    with torch.no_grad():      # Temporary disable autograd tracking
        w -= lr * w.grad       # Update weights

    w.grad.zero_()             # Clear error gradients for next step

    print(
        f"step={step:02d} "
        f"loss={loss.item():.4f} "
        f"w={w.tolist()}"
    )
