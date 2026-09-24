# cn_estimator.py
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNEstimator(nn.Module):
    def __init__(self, num_genes, region_mask, max_cn=3.0):
        super().__init__()
        self.max_cn = max_cn
        self.register_buffer('region_mask', region_mask)
        
        init_logit = torch.logit(torch.tensor(2.0 / max_cn))
        self.cn_logits = nn.Parameter(torch.full((num_genes,), init_logit.item()))
    
    def forward(self):
        return torch.sigmoid(self.cn_logits) * self.max_cn
    
    def loss(self, region_cov, delta=0.5):

        expected = self.forward() @ self.region_mask
        error = region_cov - expected
        
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = delta * (abs_error - 0.5 * delta)
        
        loss = torch.where(abs_error <= delta, quadratic, linear)
        return torch.mean(loss)


def run_cn_estimator(region_cov, region_mask, gene_names, max_cn=3.0,
                     num_iterations=500, lr=0.1, print_every=100, delta=0.5):
    
    region_cov = region_cov.to(device)
    region_mask = region_mask.to(device)
    
    model = CNEstimator(len(gene_names), region_mask, max_cn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss = model.loss(region_cov, delta=delta)
        loss.backward()
        optimizer.step()
        
        if step % print_every == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, CN={model.forward().detach().cpu().numpy()}")
    
    with torch.no_grad():
        CN = model.forward()
        expected = CN @ region_mask
        
        print(f"\n=== Fit ===")
        for i, (obs, exp) in enumerate(zip(region_cov.cpu(), expected.cpu())):
            diff = obs - exp
            outlier = "*" if abs(diff) > delta else ""
            print(f"Region {i}: obs={obs:.3f}, exp={exp:.3f}, diff={diff:+.3f} {outlier}")
        
        CN_np = CN.cpu().numpy()
    
    return {gene_names[i]: CN_np[i] for i in range(len(gene_names))}
