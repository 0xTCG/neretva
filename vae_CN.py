
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#
##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from vae_helper  import *
from coverage_CN import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Encoder(nn.Module):
    """
    Balanced encoder with improved allele network to match base network complexity
    """
    def __init__(self, num_mutations, num_alleles, num_sparse_entries, hidden, dropout):
        super().__init__()
        self.num_alleles = num_alleles
        self.num_sparse_entries = num_sparse_entries
        
        self.drop = nn.Dropout(dropout)
        
        self.fc_shared1 = nn.Linear(num_mutations, hidden)
        self.fc_shared2 = nn.Linear(hidden, hidden)
        self.fc_shared3 = nn.Linear(hidden, hidden)  
        
        self.fc_allele1 = nn.Linear(hidden, hidden)
        self.fc_allele2 = nn.Linear(hidden, hidden // 2)
        
        self.fcmu_allele = nn.Linear(hidden // 2, num_alleles)
        self.fclv_allele = nn.Linear(hidden // 2, num_alleles)
        
        self.fc_base1 = nn.Linear(hidden, hidden)
        self.fc_base2 = nn.Linear(hidden, hidden // 2)
        self.fc_base3 = nn.Linear(hidden // 2, hidden // 2)
        

        self.fcmu_base = nn.Linear(hidden // 2, num_sparse_entries * 6)
        self.fclv_base = nn.Linear(hidden // 2, num_sparse_entries * 6)
        
        self.fc_r = nn.Linear(hidden, num_mutations)

    def initialize_base_params(self, sparse_prior_mus, sparse_prior_logvars):
        with torch.no_grad():
            noise_scale_mu = 1e-2
            noise_scale_logvar = 1e-3
            
            self.fcmu_base.bias.data = sparse_prior_mus.clone() + torch.randn_like(sparse_prior_mus) * noise_scale_mu
            self.fclv_base.bias.data = sparse_prior_logvars.clone() + torch.randn_like(sparse_prior_logvars) * noise_scale_logvar
            self.fcmu_allele.bias.data.fill_(-3.0)
            self.fclv_allele.bias.data.fill_(-1.0)
            self.fcmu_base.weight.data.normal_(0, 1e-3)
            self.fclv_base.weight.data.normal_(0, 1e-4)
            
            # Initialize all layers properly
            for module in [self.fc_shared1, self.fc_shared2, self.fc_shared3,
                          self.fc_allele1, self.fc_allele2,
                          self.fc_base1, self.fc_base2, self.fc_base3]:
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        h = F.relu(self.fc_shared1(inputs))
        h = self.drop(h)
        h = F.relu(self.fc_shared2(h))
        h = self.drop(h)
        h = F.relu(self.fc_shared3(h))
        h_shared = self.drop(h)
        
        h_allele = F.relu(self.fc_allele1(h_shared))
  
        h_allele = self.drop(h_allele)
        
        h_allele = F.relu(self.fc_allele2(h_allele))

        h_allele = self.drop(h_allele)
        
        logtheta_allele_loc = self.fcmu_allele(h_allele)
        logtheta_allele_logvar = torch.clamp(self.fclv_allele(h_allele), min=-5.0, max=2.0)
        
        h_base = F.relu(self.fc_base1(h_shared))
     
        h_base = self.drop(h_base)
        
        h_base = F.relu(self.fc_base2(h_base))

        h_base = self.drop(h_base)
        
        h_base = F.relu(self.fc_base3(h_base))
        
        r_logits = self.fc_r(h_shared)

        logtheta_base_loc_flat = self.fcmu_base(h_base)
        logtheta_base_logvar_flat = torch.clamp(self.fclv_base(h_base), min=-80.0, max=-1.0)
        
        logtheta_base_loc = logtheta_base_loc_flat.view(batch_size, self.num_sparse_entries, 6)
        logtheta_base_logvar = logtheta_base_logvar_flat.view(batch_size, self.num_sparse_entries, 6)
        
        return (logtheta_allele_loc, logtheta_allele_logvar,
                logtheta_base_loc, logtheta_base_logvar, r_logits)

class VAE(nn.Module):
    def __init__(self, num_mutations, num_alleles, num_sparse_entries, 
                 total_mut_counts, sparse_prior_mu, sparse_prior_logvar, 
                 mapping_indices, sample, db, hidden=256, dropout=0, 
                 initial_strength=None, strength_lb=None, strength_ub=None, 
                  functional_observation_counts = None,
                 functional_indices = None,
                region_mask=None,        
                 region_normalized=None,
                 gene_masks = None,
                   gene_names = None
                 ):
        
        super().__init__()
        self.num_mutations = num_mutations 
        self.num_alleles = num_alleles 
        self.num_sparse_entries = num_sparse_entries  

        self.total_mut_counts = total_mut_counts 
        self.sample = sample
        self.db = db

        self.register_buffer('allele_indices', mapping_indices[0])
        self.register_buffer('mut_indices', mapping_indices[1]) 
        self.register_buffer('sparse_flat_indices', mapping_indices[2])
        
        self.encoder = Encoder(num_mutations, num_alleles, num_sparse_entries, hidden, dropout)
        
        self.register_buffer('sparse_prior_mu', sparse_prior_mu.to(device))
        self.register_buffer('sparse_prior_logvar', sparse_prior_logvar.to(device))

        self.register_buffer('strength_lb', strength_lb)
        self.register_buffer('strength_ub', strength_ub)
        normalized = (initial_strength - strength_lb) / (strength_ub - strength_lb + 1e-10) # variance wrt LB
        
        self.register_buffer('functional_observation_counts', functional_observation_counts)
        self.register_buffer('functional_indices', functional_indices)
        initial_logits = torch.logit(normalized)
        self.strength_logits = nn.Parameter(initial_logits)

        self.dispersion_logit = nn.Parameter(torch.tensor(2.0))         
        
        self.register_buffer('region_mask', region_mask)
        self.register_buffer('region_normalized', region_normalized)
        self.log_sigma_region = nn.Parameter(torch.tensor(0.0))
  
        self.register_buffer('gene_masks', gene_masks)
        self.gene_names = gene_names
        self.num_genes = len(gene_names)


    def compute_functional_Hellinger_divergence(self, theta, beta, debug=False,
                                        gene_specific_list=['KIR2DL4','KIR3DL2', 'KIR3DL3'],
                                        alpha=0.5): 
   
        eps = 1e-10
        gene_specific_div = torch.tensor(0.0, device=beta.device)
        
        for i in range(self.num_genes):
            gene_name = self.gene_names[i]
            
            if gene_name not in gene_specific_list:
                continue
            
            gene_mask = self.gene_masks[i]
            beta_gene = beta[:, gene_mask]
            obs_gene = self.functional_observation_counts[gene_mask]
            expected_gene = torch.matmul(theta.unsqueeze(0), beta_gene).squeeze(0)
            
            P = obs_gene / (obs_gene.sum() + eps)
            Q = expected_gene / (expected_gene.sum() + eps)
            
            div = 0.5 * torch.sum((torch.sqrt(P) - torch.sqrt(Q))**2)
            
            gene_specific_div += div
            
            if debug:
                print(f"  {gene_name}: Hellinger={div.item():.6f}")
        
        return gene_specific_div

    def compute_functional_JS_divergence(self, theta, beta, debug=False,
                                        gene_specific_list=['KIR2DL4', 'KIR3DL2', 'KIR3DL3'],
                                        temperature=0.5):
       
        # T < 1: Sharper (emphasizes differences)
        # T > 1: Softer (smooths differences)
        # T = 1: Standard JS
       
        eps = 1e-10
        gene_specific_js = torch.tensor(0.0, device=beta.device)
        
        for i in range(self.num_genes):
            gene_name = self.gene_names[i]
            
            if gene_name not in gene_specific_list:
                continue
            
            gene_mask = self.gene_masks[i]
            beta_gene = beta[:, gene_mask]
            obs_gene = self.functional_observation_counts[gene_mask]
            expected_gene = torch.matmul(theta.unsqueeze(0), beta_gene).squeeze(0)
            
            obs_temp = torch.pow(obs_gene + eps, 1/temperature)
            exp_temp = torch.pow(expected_gene + eps, 1/temperature)
            
            P = obs_temp / obs_temp.sum()
            Q = exp_temp / exp_temp.sum()
            
            M = 0.5 * (P + Q)
            js = 0.5 * (
                torch.sum(P * torch.log((P + eps) / (M + eps))) +
                torch.sum(Q * torch.log((Q + eps) / (M + eps)))
            )
            
            gene_specific_js += js
            
            if debug:
                print(f"  {gene_name}: JS(T={temperature})={js.item():.6f}")
        
        return gene_specific_js
    def construct_beta_sparse(self, base_probs):
        batch_size = base_probs.size(0)
        
        sparse_flat = base_probs.view(batch_size, -1)
        
        beta_batch = []
        for b in range(batch_size):
            beta = torch.zeros(self.num_alleles, self.num_mutations, device=base_probs.device)
            beta[self.allele_indices, self.mut_indices] = sparse_flat[b, self.sparse_flat_indices]
            
         
            beta_batch.append(beta)
        
        return torch.stack(beta_batch)
    
    def get_strength_mask(self):
        # Map logits to [0, 1] via sigmoid
        normalized = torch.sigmoid(self.strength_logits)
        
        # Map [0, 1] to [lb, ub]
        strength = self.strength_lb + normalized * (self.strength_ub - self.strength_lb)
        
        return strength
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, mut_counts):
        max_cn = 3
        cn_mu, cn_logvar, base_mu, base_logvar, r_logits = self.encoder(mut_counts)
        cn_z = self.reparameterize(cn_mu, cn_logvar)
        base_z = self.reparameterize(base_mu, base_logvar)

        # CN = torch.exp(torch.clamp(cn_z, min=-10, max=1.6))
        CN = torch.sigmoid(cn_z)* max_cn
     
        base_probs = F.softmax(base_z, dim=-1)
        beta = self.construct_beta_sparse(base_probs)
            
        return (beta, cn_mu, cn_logvar, base_mu, base_logvar, CN, base_probs, r_logits)
    
    def compute_major_entropy_loss(self, CN):
        eps = 1e-10
        total_entropy = 0
        
        major_to_indices = {}
        for idx, allele in enumerate(self.sample.valid_alleles):
            major = allele.name.split('.')[0] 
            if major not in major_to_indices:
                major_to_indices[major] = []
            major_to_indices[major].append(idx)
        
        for major, indices in major_to_indices.items():
            if len(indices) > 1:
                cn_major = CN[indices]
                cn_probs = cn_major / (cn_major.sum() + eps)
                entropy = -(cn_probs * torch.log(cn_probs + eps)).sum()
                total_entropy += entropy
        
        return total_entropy
    
    def compute_integer_loss(self, CN):
        integer_loss = (1 - torch.cos(2 * np.pi * CN)).sum()
        return integer_loss

    def print_variant_reconstruction(self, mut_counts, CN, beta, top_k=None, min_obs=0):
        eps = 1e-10
        
        # Get strength-adjusted beta
        strength_mask = self.get_strength_mask()
        beta_adjusted = beta * strength_mask.unsqueeze(0)
        
        # Compute expected counts: CN @ beta_adjusted
        expected_counts = torch.bmm(CN.unsqueeze(1), beta_adjusted).squeeze()  # [num_mutations]
        
        # Collect results for each variant
        results = []
        for mut_idx in range(self.num_mutations):
            variant_idx = self.sample.valid_indices[mut_idx]
            variant = self.db.variants()[variant_idx]
            
            obs = mut_counts[mut_idx].item()
            exp = expected_counts[mut_idx].item()
            
            if obs < min_obs and exp < min_obs:
                continue
            
            diff = obs - exp
            rel_error = abs(diff) / (obs + eps) * 100 if obs > 0 else 0
            
            variant_str = f"{variant.gene.name}:{variant.pos}→{variant.variant}"
            results.append({
                'variant': variant_str,
                'observed': obs,
                'expected': exp,
                'diff': diff,
                'rel_error': rel_error,
                'is_functional': variant.is_major
            })
        
        # Sort by absolute difference (largest errors first)
        results.sort(key=lambda x: abs(x['diff']), reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        # Print header
        print(f"\n{'='*80}")
        print(f"{'Variant':<35} {'Obs':>10} {'Exp':>10} {'Diff':>10} {'Err%':>8} {'Func':>5}")
        print(f"{'-'*80}")
        
        # Print each variant
        for r in results:
            func_marker = '*' if r['is_functional'] else ''
            print(f"{r['variant']:<35} {r['observed']:>10.1f} {r['expected']:>10.1f} "
                  f"{r['diff']:>+10.1f} {r['rel_error']:>7.1f}% {func_marker:>5}")
        
        print(f"{'='*80}")
        
        # Print summary statistics
        all_obs = mut_counts.cpu().numpy()
        all_exp = expected_counts.detach().cpu().numpy()
        total_obs = np.sum(all_obs)
        total_exp = np.sum(all_exp)
        mse = np.mean((all_obs - all_exp)**2)
        mae = np.mean(np.abs(all_obs - all_exp))
        
        print(f"Total observed: {total_obs:.1f}, Total expected: {total_exp:.1f}")
        print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        return results


    def loss_function(self, mut_counts, 
                    cn_mu, cn_logvar,
                    base_mu, base_logvar, beta, 
                    CN, base_probs, r_logits,
                    cn_kld_weight=1.0, base_kld_weight=0.001, 
                    functional_kld_weight=0.6, region_ll_weight = 1.0, debug_functional=False):
        
        eps = 1e-10
        
        strength_mask = self.get_strength_mask()


        beta_adjusted = beta * strength_mask.unsqueeze(0)
        expected_counts = torch.bmm(
            CN.unsqueeze(1), 
            beta_adjusted
        ).squeeze(1)
        mu = expected_counts.clamp(min=eps).squeeze(0) 

        # r_reg = torch.log(10.0 / r) 
        r_min = 50 * mu  # [num_mutations]
        r = F.softplus(r_logits.squeeze(0)) + r_min  
        log_likelihood = torch.sum(
            torch.lgamma(mut_counts + r) - torch.lgamma(r) - torch.lgamma(mut_counts + 1)
            + r * torch.log(r / (r + mu))
            + mut_counts * torch.log(mu / (r + mu))
        )
  
        expected_region = CN @ self.region_mask 
        sigma = torch.exp(self.log_sigma_region).clamp(min=0.1)
        region_ll = -0.5 * torch.sum(
            ((self.region_normalized - expected_region) / sigma) ** 2
        ) - len(self.region_normalized) * torch.log(sigma)
        
        # prior_mu = -1.0
        prior_mu = 0

        cn_kld = -0.5 * torch.sum(
            1 + cn_logvar - (cn_mu - prior_mu).pow(2) - cn_logvar.exp()
        )
 

        base_mu_flat = base_mu.view(-1)
        base_logvar_flat = base_logvar.view(-1)
        base_kld = -0.5 * torch.sum(
            1 + base_logvar_flat - self.sparse_prior_logvar
            - (base_logvar_flat.exp() + (base_mu_flat - self.sparse_prior_mu).pow(2)) / self.sparse_prior_logvar.exp()
        )
        
        beta_normalized = beta_adjusted.clone()
        row_sums = beta_normalized.sum(dim=2, keepdim=True)
        beta_normalized = beta_normalized / (row_sums + 1e-10)
        CN_props = CN / (CN.sum(dim=-1, keepdim=True) + 1e-10)
        # functional_kl = self.compute_functional_Hellinger_divergence(
        functional_kl = self.compute_functional_JS_divergence(
            # CN.squeeze(0),
            CN_props.squeeze(0),
            beta_normalized.squeeze(0),
            debug=debug_functional,
            temperature=2.0,
            gene_specific_list=['KIR2DL4', 'KIR3DL2', 'KIR3DL3']

            # threshold=self.sample.expected_coverage*0.4
        )
        
        base_probs_reshaped = base_probs.view(base_probs.size(0), -1, 6)
        base_entropy = -(base_probs_reshaped * torch.log(base_probs_reshaped + eps)).sum(dim=-1).mean()
        
        major_entropy = self.compute_major_entropy_loss(CN.squeeze(0))

        integer_loss = self.compute_integer_loss(CN.squeeze(0))
        cn_entropy = -(CN * torch.log(CN + 1e-10)).sum()
        loss = (-log_likelihood -region_ll_weight*region_ll + 
                cn_kld_weight * cn_kld + 
                base_kld_weight * base_kld + 
                functional_kld_weight * functional_kl + 
                0 * base_entropy + 1*major_entropy + 0*integer_loss
                +1.0*cn_entropy
                )
        

        return loss, -log_likelihood, -region_ll, cn_kld, base_kld, functional_kl
        
def create_sparse_to_beta_indices(sample, db):
    
    allele_indices = []
    mut_indices = []
    sparse_flat_indices = []
    
    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            for base_idx, base in enumerate(BASES):
                if base in pos_obj.variants:
                    variant = pos_obj.variants[base]
                    if variant.index in sample.valid_indices:
                        mut_idx = sample.valid_indices.index(variant.index)
                        
                        allele_indices.append(allele_idx)
                        mut_indices.append(mut_idx)
                        sparse_flat_indices.append(sparse_idx * len(BASES) + base_idx)  
            
            sparse_idx += 1
    
    return (torch.tensor(allele_indices, dtype=torch.long),
            torch.tensor(mut_indices, dtype=torch.long),
            torch.tensor(sparse_flat_indices, dtype=torch.long))


def run_vae_single_seed(total_mut_counts, valid_alleles, mut_counts, num_sparse_entries, 
                        sparse_prior_mu, sparse_prior_logvar, mapping_indices, sample, db,
                        seed=0, num_iterations=3000, lr=0.005, print_every=500, 
                        initial_strength=None, strength_lb=None, strength_ub=None, 
                        functional_observation_counts=None,
                        functional_indices=None,
                        bam_id=None,
                        region_mask = None,
                        region_normalized = None,
                        gene_masks = None,
                        gene_names = None,
                      ):
   
    torch.manual_seed(seed)

    model = VAE(
        mut_counts.shape[0], len(valid_alleles), num_sparse_entries,
        total_mut_counts, sparse_prior_mu, sparse_prior_logvar,
        mapping_indices, sample, db,
        hidden=512,
        dropout=0,
        initial_strength=initial_strength,
        strength_lb=strength_lb,
        strength_ub=strength_ub,
        functional_observation_counts=functional_observation_counts,
        functional_indices=functional_indices,
        region_mask=region_mask,
        region_normalized=region_normalized,
        gene_masks = gene_masks,
        gene_names = gene_names

    ).to(device)
    model.encoder.initialize_base_params(sparse_prior_mu, sparse_prior_logvar)
    
    cn_params = [p for name, p in model.named_parameters() if 'allele' in name]
    base_params = [p for name, p in model.named_parameters() if 'base' in name]
    other_params = [p for name, p in model.named_parameters() 
                    if 'allele' not in name and 'base' not in name]

    optimizer = optim.Adam([
        {'params': cn_params, 'lr': lr},
        {'params': base_params, 'lr': lr * 0.02},
        {'params': other_params, 'lr': lr}
    ])
    
    model.train()
    loss_history = []
    
    for step in range(num_iterations):
        optimizer.zero_grad()
        
        (beta, cn_mu, cn_logvar,
        base_mu, base_logvar,
        CN, base_probs, r_logits) = model(mut_counts.unsqueeze(0))
        
        debug_functional = (step % print_every == 0)
        
               
        loss, recon_loss, region_loss, cn_kl, base_kl, functional_kl  = model.loss_function(
            mut_counts,
            cn_mu.squeeze(0),
            cn_logvar.squeeze(0),
            base_mu.squeeze(0),
            base_logvar.squeeze(0),
            beta,
            CN,
            base_probs,
            r_logits,
            cn_kld_weight = 1,
            base_kld_weight=1e-4,
            functional_kld_weight=0,
            region_ll_weight = 0,
            debug_functional=debug_functional
        )
        
        current_loss = loss.item()
        loss_history.append((current_loss, recon_loss.item(), region_loss.item(), cn_kl.item(), base_kl.item(), functional_kl.item()))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % print_every == 0:
            print(f"Seed {seed}, Step {step:4d}: Loss = {current_loss:.2f} "
                f"(Recon: {recon_loss.item():.2f}, Region: {region_loss.item():.2f}, "
                f"CN KL: {cn_kl.item():.2f}, Base KL: {base_kl.item():.2f}, "
                f"Functional KL: {functional_kl.item():.4f})")

    # Use final model directly
    model.eval()
    with torch.no_grad():
        (beta, cn_mu, cn_logvar,
        base_mu, base_logvar,
        CN, base_probs, r_logits) = model(mut_counts.unsqueeze(0))
        
        # print(f"\n=== Final Variant Reconstruction (Seed {seed}) ===")
        model.print_variant_reconstruction(mut_counts, CN, beta, top_k=50, min_obs=1)
        strength_mask = model.get_strength_mask()  # Shape: (num_alleles, num_mutations)
        beta_squeezed = beta.squeeze(0)  # Remove batch dimension first!
        beta_adjusted = beta_squeezed * strength_mask  # Now shapes match
        row_sums = beta_adjusted.sum(dim=1, keepdim=True)
        beta_adjusted = beta_adjusted / (row_sums + 1e-10)

    final_loss = loss_history[-1][0]
    return CN, loss_history, final_loss, num_iterations - 1, beta_adjusted

def run_vae_multi_seed(total_mut_counts, valid_alleles, mut_counts,  
                       num_sparse_entries, sparse_prior_mu, sparse_prior_logvar, 
                       mapping_indices, sample, db,
                       seeds=[42, 123, 456], num_iterations=3000, lr=0.005, 
                       print_every=500, plot_loss=False, save_plot=False,  
                       initial_strength=None, strength_lb=None, strength_ub=None, 
                       functional_observation_counts=None,
                       functional_indices=None,
                       bam_id=None,
                       region_mask = None,
                       region_normalized = None,
                        gene_masks = None,
                        gene_names = None,
                      ):

    results = {}
    best_seed = None
    best_final_loss = float('inf')
    
    for seed in seeds:
        print(f"\n[VAE seed {seed}]")
        cn, loss_history, seed_best_loss, seed_best_step, beta = run_vae_single_seed(
            total_mut_counts, valid_alleles, mut_counts, 
            num_sparse_entries, sparse_prior_mu, sparse_prior_logvar, 
            mapping_indices, sample, db,
            seed=seed, num_iterations=num_iterations, lr=lr, print_every=print_every,
            initial_strength=initial_strength, strength_lb=strength_lb, strength_ub=strength_ub,
            functional_observation_counts=functional_observation_counts,
            functional_indices=functional_indices,
            bam_id=bam_id,
            region_mask=region_mask,
            region_normalized=region_normalized,
            gene_masks = gene_masks,
            gene_names = gene_names,
        )
        
        # Use best loss from training, not final loss
        final_loss = seed_best_loss
        final_recon = loss_history[seed_best_step][1]
        final_allele_kl = loss_history[seed_best_step][2]
        final_base_kl = loss_history[seed_best_step][3]
        
        results[seed] = {
            'cn': cn,
            'loss_history': loss_history,
            'final_loss': final_loss,
            'best_step': seed_best_step,
            'final_recon': final_recon,
            'final_allele_kl': final_allele_kl,
            'final_base_kl': final_base_kl,
            'beta': beta
        }
        
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_seed = seed
            print(f"New best model with loss: {best_final_loss:.2f} (from step {seed_best_step})")

    print("\n=== Multi-Seed VAE Results ===")
    print(f"Best model had seed {best_seed} with loss: {best_final_loss:.2f}")
    print("All seeds best losses:")
    for seed in seeds:
        r = results[seed]
        print(f"  Seed {seed}: Total={r['final_loss']:.2f} @ step {r['best_step']} "
              f"(Recon={r['final_recon']:.2f}, A_KL={r['final_allele_kl']:.2f}, "
              f"B_KL={r['final_base_kl']:.2f})" + 
              (" <- Best" if seed == best_seed else ""))
    
    best_result = results[best_seed]
    return best_result['cn'], best_result['beta']



def create_strength_parameters(sample, db, lb_factor=0.8, ub_factor=1.2):
    num_alleles = len(sample.valid_alleles)
    num_mutations = len(sample.valid_indices)
    
    initial_strength = torch.ones(num_alleles, num_mutations)
    strength_lb = torch.ones(num_alleles, num_mutations) * lb_factor  
    strength_ub = torch.ones(num_alleles, num_mutations) * ub_factor 
    
    for allele_idx, allele in enumerate(sample.valid_alleles):
        if not allele.enabled:
            continue
        
        allele_name = f"{allele.gene.name}*{allele.name}"
        has_bounds = False
        
        for pos_obj in allele.generatable_positions:
            position = pos_obj.position
            
            for variant in pos_obj.variants.values():
                # if allele.extended_allele_vector[variant.index] != 1:
                #     continue
                
                lb, ub = allele.get_position_strength(
                    pos_obj, 
                    variant.variant
                )

                # lb_normalized = lb / sample.expected_coverage
                # ub_normalized = ub / sample.expected_coverage
                lb_normalized = lb 
                ub_normalized = ub
                
 
                # Assert valid bounds
                if variant.index in sample.valid_indices:
                    mut_idx = sample.valid_indices.index(variant.index)
                    
                    strength_lb[allele_idx, mut_idx] = lb_normalized
                    strength_ub[allele_idx, mut_idx] = ub_normalized
                    
                    initial_strength[allele_idx, mut_idx] = (lb_normalized + ub_normalized) / 2.0
                    
                    if not has_bounds:
                        has_bounds = True
                    

    return initial_strength, strength_lb, strength_ub


def run_vae(total_mut_counts, valid_alleles, mut_counts,
            region_mask = None, region_normalized = None,
            sample=None, db=None, num_iterations=3000, bam_id = None):
    mut_counts = mut_counts.to(device)

    lb_factor = 1.00 # uninformed, lowerbound and upperbound of the coverage wrt 1x expected coverage.
    ub_factor = 1.00

    set_expected_position_strength(sample, db, lb_factor=lb_factor, ub_factor=ub_factor)
    initial_strength, strength_lb, strength_ub = create_strength_parameters(sample, db, lb_factor=lb_factor, ub_factor=ub_factor)
    
    # Create sparse priors
    sparse_prior_mu, sparse_prior_logvar = create_sparse_priors_V2(sample, db)
    num_sparse_entries = len(sparse_prior_mu) // len(BASES)  # 6 bases
    sparse_prior_mu = sparse_prior_mu.to(device)
    sparse_prior_logvar = sparse_prior_logvar.to(device)
    



    mapping_indices = create_sparse_to_beta_indices(sample, db)
    mapping_indices = (
        mapping_indices[0].to(device),
        mapping_indices[1].to(device), 
        mapping_indices[2].to(device)
    )

    functional_indices = create_functional_indices(sample, db)
    gene_masks, gene_names = create_gene_masks(sample, db, functional_indices)

    functional_indices = functional_indices.to(device)
    gene_masks = gene_masks.to(device)
    
    # 
    functional_observation_counts = create_functional_observation_counts(sample, db)
    functional_observation_counts = functional_observation_counts.to(device)
    
    # seeds = [88]
    # seeds = [42,123,456]
    seeds = [42]
    cn,  beta = run_vae_multi_seed(
        total_mut_counts, 
        valid_alleles, 
        mut_counts, 
        num_sparse_entries,
        sparse_prior_mu,
        sparse_prior_logvar,
        mapping_indices,
        sample,db,
        seeds=seeds,
        num_iterations=num_iterations,
        print_every=500,
        plot_loss=False,
        initial_strength=initial_strength,
        strength_lb=strength_lb, 
        strength_ub=strength_ub,
        functional_observation_counts = functional_observation_counts,
        functional_indices = functional_indices,
        bam_id = bam_id,
        region_mask = region_mask,
        region_normalized = region_normalized,
        gene_masks = gene_masks,
        gene_names = gene_names,

    )
    
    return cn.cpu().numpy(), beta
