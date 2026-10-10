import torch
import torch.nn as nn
import math

# 1. Setup Synthetic Data and Model
torch.manual_seed(42)
n_train, n_test, d_in, d_hidden, d_out = 100, 1000, 10, 20, 1
X_train, y_train = torch.randn(n_train, d_in), torch.randn(n_train, d_out)
X_test, y_test = torch.randn(n_test, d_in), torch.randn(n_test, d_out)

model = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))
loss_fn = nn.MSELoss()

# 2. SGLD Hyperparameters
lr, n_epochs, temperature = 0.01, 150, 1.0  # temperature = 1 / beta
accumulated_sq_grad = 0.0

# 3. Training Loop with SGLD Update
for epoch in range(n_epochs):
    model.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()

    with torch.no_grad():
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.norm().item() ** 2
                
                # SGLD Step: w_{t+1} = w_t - lr * grad + sqrt(2 * lr * temperature) * noise
                p.sub_(p.grad, alpha=lr)
                p.add_(torch.randn_like(p), alpha=math.sqrt(2 * lr * temperature))
                
        accumulated_sq_grad += grad_norm_sq

# 4. Empirical Generalization Gap
with torch.no_grad():
    train_loss = loss_fn(model(X_train), y_train).item()
    test_loss = loss_fn(model(X_test), y_test).item()
    empirical_gap = abs(test_loss - train_loss)

# 5. Information-Theoretic Bound (Trajectory Mutual Information)
# Bound assumes 1-subgaussian loss distribution.
mi_bound = (1.0 / (4.0 * temperature)) * lr * accumulated_sq_grad
info_bound = math.sqrt((2.0 * mi_bound) / n_train)

print(f"Train Loss:     {train_loss:.4f}")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Empirical Gap:  {empirical_gap:.4f}")
print(f"Expected Bound: {info_bound:.4f}")