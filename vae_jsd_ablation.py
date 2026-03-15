
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from vae_helper import *
#
##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from vae_helper  import *
from coverage import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Encoder(nn.Module):
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
        
        # self.bn_allele1 = nn.BatchNorm1d(hidden)
        # self.bn_allele2 = nn.BatchNorm1d(hidden // 2)
        
        self.fcmu_allele = nn.Linear(hidden // 2, num_alleles)
        self.fclv_allele = nn.Linear(hidden // 2, num_alleles)
        
        self.fc_base1 = nn.Linear(hidden, hidden)
        self.fc_base2 = nn.Linear(hidden, hidden // 2)
        self.fc_base3 = nn.Linear(hidden // 2, hidden // 2)
        
        # self.bn_base1 = nn.BatchNorm1d(hidden)
        # self.bn_base2 = nn.BatchNorm1d(hidden // 2)
        
        # Final base parameter layers
        self.fcmu_base = nn.Linear(hidden // 2, num_sparse_entries * 6)
        self.fclv_base = nn.Linear(hidden // 2, num_sparse_entries * 6)
        
    def initialize_base_params(self, sparse_prior_mus, sparse_prior_logvars):
        with torch.no_grad():
            noise_scale_mu = 1e-2
            noise_scale_logvar = 1e-3
            
            self.fcmu_base.bias.data = sparse_prior_mus.clone() + torch.randn_like(sparse_prior_mus) * noise_scale_mu
            self.fclv_base.bias.data = sparse_prior_logvars.clone() + torch.randn_like(sparse_prior_logvars) * noise_scale_logvar
            
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
        # if batch_size > 1:
        #     h_allele = self.bn_allele1(h_allele)
        h_allele = self.drop(h_allele)
        
        h_allele = F.relu(self.fc_allele2(h_allele))
        # if batch_size > 1:
        #     h_allele = self.bn_allele2(h_allele)
        h_allele = self.drop(h_allele)
        
        logtheta_allele_loc = self.fcmu_allele(h_allele)
        logtheta_allele_logvar = torch.clamp(self.fclv_allele(h_allele), min=-5.0, max=2.0)
        
        h_base = F.relu(self.fc_base1(h_shared))
        # if batch_size > 1:
        #     h_base = self.bn_base1(h_base)
        h_base = self.drop(h_base)
        
        h_base = F.relu(self.fc_base2(h_base))
        # if batch_size > 1:
        #     h_base = self.bn_base2(h_base)
        h_base = self.drop(h_base)
        
        h_base = F.relu(self.fc_base3(h_base))
        
        logtheta_base_loc_flat = self.fcmu_base(h_base)
        logtheta_base_logvar_flat = torch.clamp(self.fclv_base(h_base), min=-80.0, max=-1.0)
        
        logtheta_base_loc = logtheta_base_loc_flat.view(batch_size, self.num_sparse_entries, 6)
        logtheta_base_logvar = logtheta_base_logvar_flat.view(batch_size, self.num_sparse_entries, 6)
        
        return (logtheta_allele_loc, logtheta_allele_logvar,
                logtheta_base_loc, logtheta_base_logvar)

class VAE(nn.Module):
    def __init__(self, num_mutations, num_alleles, num_sparse_entries, 
                 total_mut_counts, sparse_prior_mu, sparse_prior_logvar, 
                 mapping_indices, sample, db, hidden=256, dropout=0, 
                 initial_strength=None, strength_lb=None, strength_ub=None, 
                  functional_observation_counts = None,
                 functional_indices = None, gene_masks = None, gene_names = None):
        
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

        self.register_buffer('gene_masks', gene_masks)
        self.gene_names = gene_names
        self.num_genes = len(gene_names)
      
        # self._create_gene_masks()
        self._calculate_normalization_weights()
    
    # def _create_gene_masks(self):
    #     gene_names = []
    #     gene_masks_list = []
        
    #     for gene in self.db.genes.values():
    #         gene_mask = torch.zeros_like(self.functional_indices, dtype=torch.bool)
            
    #         for mut_idx in range(len(self.functional_indices)):
    #             if self.functional_indices[mut_idx]:
    #                 variant = self.db.variants()[self.sample.valid_indices[mut_idx]]
    #                 if variant.gene.name == gene.name:
    #                     gene_mask[mut_idx] = True
            
    #         if gene_mask.sum() > 0:
    #             gene_names.append(gene.name)
    #             gene_masks_list.append(gene_mask)
        
    #     # Stack all masks into a single tensor for fast access
    #     self.register_buffer('gene_masks', torch.stack(gene_masks_list))
    #     self.gene_names = gene_names
    #     self.num_genes = len(gene_names)

    def _calculate_normalization_weights(self):
        generatable_counts = torch.zeros(self.num_alleles)
        
        for allele_idx, allele in enumerate(self.sample.valid_alleles):
            generatable_counts[allele_idx] = len(allele.generatable_positions)
        
        self.register_buffer("generatable_counts", generatable_counts.to(device))
    def get_functional_coverage_mismatches(self, theta, beta, use_js=False, use_reverse_kl=False, top_k=None):

        eps = 1e-10
        strength_mask = self.get_strength_mask()
        beta_adjusted = beta * strength_mask
        row_sums = beta_adjusted.sum(dim=1, keepdim=True)
        beta_adjusted = beta_adjusted / (row_sums + 1e-10)

        beta_functional = beta_adjusted[:, self.functional_indices]
        obs_functional = self.functional_observation_counts[self.functional_indices]
        expected_unnorm = torch.matmul(theta.unsqueeze(0), beta_functional).squeeze(0)
        
        P_observed = obs_functional / (obs_functional.sum() + eps)
        P_expected = expected_unnorm / (expected_unnorm.sum() + eps)
        
        # Compute divergence contributions based on method
        if use_js:
            M = 0.5 * (P_observed + P_expected)
            divergence_contributions = 0.5 * (
                torch.where(
                    P_observed > eps,
                    P_observed * torch.log(P_observed / (M + eps)),
                    torch.zeros_like(P_observed)
                ) +
                torch.where(
                    P_expected > eps,
                    P_expected * torch.log(P_expected / (M + eps)),
                    torch.zeros_like(P_expected)
                )
            )
        elif use_reverse_kl:
            forward_contributions = torch.where(
                P_observed > eps,
                P_observed * torch.log(P_observed / (P_expected + eps)),
                torch.zeros_like(P_observed)
            )
            reverse_contributions = torch.where(
                P_expected > eps,
                P_expected * torch.log(P_expected / (P_observed + eps)),
                torch.zeros_like(P_expected)
            )
            divergence_contributions = forward_contributions + reverse_contributions
        else:
            divergence_contributions = torch.where(
                P_observed > eps,
                P_observed * torch.log(P_observed / (P_expected + eps)),
                torch.zeros_like(P_observed)
            )
        
        # Get absolute contributions for ranking
        abs_contributions = torch.abs(divergence_contributions)
        
        functional_mut_indices = torch.where(self.functional_indices)[0]
        
        results = []
        for i, func_mut_idx in enumerate(functional_mut_indices):
            func_mut_idx_val = func_mut_idx.item()
            variant_idx = self.sample.valid_indices[func_mut_idx_val]
            variant = self.db.variants()[variant_idx]
            
            variant_str = f"{variant.gene.name}:{variant.pos}→{variant.variant}"
            exp_count = expected_unnorm[i].item()*self.total_mut_counts
            obs_count = obs_functional[i].item()
            div_contrib = abs_contributions[i].item()
            
            results.append((variant_str, exp_count, obs_count, div_contrib))
        
        results.sort(key=lambda x: x[3], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    def apply_normalization(self, theta):
       
        epsilon = 1e-10
        generatable_counts = self.generatable_counts.to(theta.device)
        # Divide each theta by its generatable count
        theta_per_position = theta / (self.generatable_counts + epsilon)
        
        # Renormalize to sum to 1
        theta_normalized = theta_per_position / (theta_per_position.sum() + epsilon)
        
        return theta_normalized
    
    # def compute_functional_kl_divergence(self, theta, beta, debug = False):
    #     eps = 1e-30
    #     # eps = 1e-10

    #     beta_functional = beta[:, self.functional_indices]  
        
    #     obs_functional = self.functional_observation_counts[self.functional_indices]
        
    #     expected_unnorm = torch.matmul(theta.unsqueeze(0), beta_functional).squeeze(0)
        
    #     P_observed = obs_functional / (obs_functional.sum() + eps)
    #     P_expected = expected_unnorm / (expected_unnorm.sum() + eps)
    #     #kld
    #     kl_loss = torch.sum(P_observed * torch.log(P_observed / (P_expected + eps) + eps))
        
    #     return kl_loss
    # def compute_functional_kl_divergence(self, theta, beta, debug=False):
    #     eps = 1e-10  # Better than 1e-30
        
    #     beta_functional = beta[:, self.functional_indices]  
    #     if debug:
    #         print(f"\n=== DIAGNOSTIC: Beta Functional ===")
    #         print(f"beta_functional shape: {beta_functional.shape}")
    #         print(f"beta_functional non-zero: {(beta_functional > 1e-6).sum().item()}")
    #         print(f"beta_functional sum per allele (first 5): {beta_functional.sum(dim=1)[:5]}")
    #         print(f"beta_functional min: {beta_functional.min().item():.6f}, max: {beta_functional.max().item():.6f}")
            
    #         # Check if any functional columns have non-zero values
    #         functional_mut_indices = torch.where(self.functional_indices)[0]
    #         print(f"\nChecking some functional column indices: {functional_mut_indices[:10].tolist()}")
    #         for i in range(min(5, len(functional_mut_indices))):
    #             col_idx = functional_mut_indices[i].item()
    #             col_sum = beta[:, col_idx].sum().item()
    #             col_nonzero = (beta[:, col_idx] > 1e-6).sum().item()
    #             print(f"  Column {col_idx}: sum={col_sum:.6f}, non-zero elements={col_nonzero}")
        
    #     obs_functional = self.functional_observation_counts[self.functional_indices]
    #     if debug:
    #         print(f"\n=== DIAGNOSTIC: Observations ===")
    #         print(f"obs_functional shape: {obs_functional.shape}")
    #         print(f"obs_functional sum: {obs_functional.sum().item():.2f}")
    #         print(f"obs_functional non-zero: {(obs_functional > 0).sum().item()}")
    #         print(f"obs_functional first 10: {obs_functional[:10]}")
        
    #     expected_unnorm = torch.matmul(theta.unsqueeze(0), beta_functional).squeeze(0)
        
    #     P_observed = obs_functional / (obs_functional.sum() + eps)
    #     P_expected = expected_unnorm / (expected_unnorm.sum() + eps)
    #     if debug:
    #         print(f"\n=== DIAGNOSTIC: Expected (Before Normalization) ===")
    #         print(f"expected_unnorm shape: {expected_unnorm.shape}")
    #         print(f"expected_unnorm sum: {expected_unnorm.sum().item():.6f}")
    #         print(f"expected_unnorm non-zero: {(expected_unnorm > 1e-6).sum().item()}")
    #         print(f"expected_unnorm first 10: {expected_unnorm[:10]}")
    #         print(f"expected_unnorm min: {expected_unnorm.min().item():.6f}, max: {expected_unnorm.max().item():.6f}")
        
    #     kl_loss = torch.sum(P_observed * torch.log(P_observed / (P_expected + eps) + eps))
        
    #     if debug:
    #         print("\n=== Functional KL Divergence Debug ===")
    #         print(f"Number of functional variants: {self.functional_indices.sum().item()}")
    #         print(f"Total observed count: {obs_functional.sum().item():.2f}")
    #         print(f"Total expected count: {expected_unnorm.sum().item():.2f}")
            
    #         # Print details for each functional variant
    #         print("\n--- Per-Variant Breakdown ---")
    #         print(f"{'Mut_Idx':<8} {'Variant':<30} {'Expected':<12} {'Observed':<12} {'P_exp':<12} {'P_obs':<12} {'KL_contrib':<12}")
    #         print("-" * 108)
            
    #         # Get actual indices from boolean mask
    #         functional_mut_indices = torch.where(self.functional_indices)[0]
            
    #         # Compute per-variant KL contributions
    #         kl_contributions = P_observed * torch.log(P_observed / (P_expected + eps) + eps)
            
    #         for i, func_mut_idx in enumerate(functional_mut_indices):
    #             func_mut_idx_val = func_mut_idx.item()
    #             variant_idx = self.sample.valid_indices[func_mut_idx_val]
    #             variant = self.db.variants()[variant_idx]
    #             variant_str = f"{variant.gene.name}:{variant.pos}→{variant.variant}"
                
    #             exp_count = expected_unnorm[i].item()*self.total_mut_counts
    #             obs_count = obs_functional[i].item()
    #             p_exp = P_expected[i].item()
    #             p_obs = P_observed[i].item()
    #             kl_contrib = kl_contributions[i].item()
                
    #             print(f"{func_mut_idx_val:<8} {variant_str:<30} {exp_count:<12.2f} {obs_count:<12.2f} {p_exp:<12.6f} {p_obs:<12.6f} {kl_contrib:<12.6f}")
            
    #         print(f"\nTotal KL Divergence: {kl_contributions.sum().item():.6f}")
    #         print("=" * 108)

    #         print(f"\nKL Divergence: {torch.sum(P_observed * torch.log(P_observed / (P_expected + eps) + eps)).item():.6f}")
    #         print("=" * 96)
        
    #     return kl_loss
    def compute_functional_kl_divergence(self, theta, beta, debug=False, use_js=True, 
                                         gene_specific_weight=1.0, 
                                         gene_specific_list=['KIR3DL3', 'KIR3DL2']):
 
        eps = 1e-10
        
        # ===== GLOBAL POOLED JS (keeps all functional positions in one pool) =====
        beta_functional = beta[:, self.functional_indices]
        obs_functional = self.functional_observation_counts[self.functional_indices]
        
        expected_unnorm = torch.matmul(theta.unsqueeze(0), beta_functional).squeeze(0)
        
        P_observed = obs_functional / (obs_functional.sum() + eps)
        P_expected = expected_unnorm / (expected_unnorm.sum() + eps)
        
        if use_js:
            M = 0.5 * (P_observed + P_expected)
            global_js = 0.5 * (
                torch.sum(P_observed * torch.log((P_observed + eps) / (M + eps))) +
                torch.sum(P_expected * torch.log((P_expected + eps) / (M + eps)))
            )
        else:
            # Standard KL divergence
            global_js = torch.sum(P_observed * torch.log((P_observed + eps) / (P_expected + eps)))
        
        if debug:
            print(f"\n=== Hybrid JS Divergence ===")
            print(f"Global Pooled JS: {global_js.item():.6f}")
        
        gene_specific_js = torch.tensor(0.0, device=beta.device)
        
        for i in range(self.num_genes):
            gene_name = self.gene_names[i]
            
            # Only compute gene-specific JS for genes in the list
            if gene_name not in gene_specific_list:
                continue
            
            gene_mask = self.gene_masks[i]
            
            beta_gene = beta[:, gene_mask]
            obs_gene = self.functional_observation_counts[gene_mask]
            
            if obs_gene.sum() < 1:
                if debug:
                    print(f"  {gene_name}: Skipped (no observations)")
                continue
            
            expected_gene = torch.matmul(theta.unsqueeze(0), beta_gene).squeeze(0)
            
            P_obs = obs_gene / (obs_gene.sum() + eps)
            P_exp = expected_gene / (expected_gene.sum() + eps)
            
            if use_js:
                M_gene = 0.5 * (P_obs + P_exp)
                js_gene = 0.5 * (
                    torch.sum(P_obs * torch.log((P_obs + eps) / (M_gene + eps))) +
                    torch.sum(P_exp * torch.log((P_exp + eps) / (M_gene + eps)))
                )
            else:
                js_gene = torch.sum(P_obs * torch.log((P_obs + eps) / (P_exp + eps)))
            
            gene_specific_js += js_gene
            
            if debug:
                print(f"  {gene_name} JS: {js_gene.item():.6f}")
        
        # ===== TOTAL LOSS =====
        total_loss = global_js + gene_specific_weight * gene_specific_js
        
        if debug:
            print(f"  Gene-specific JS sum: {gene_specific_js.item():.6f}")
            print(f"  Gene-specific weighted: {(gene_specific_weight * gene_specific_js).item():.6f}")
            print(f"  TOTAL: {total_loss.item():.6f}")
        
        return total_loss
    # def compute_functional_kl_divergence(self, theta, beta, debug=False, use_js=True):
    #     eps = 1e-10

    #     gene_masks_expanded = self.gene_masks.unsqueeze(1).float()  # [num_genes, 1, num_mutations]
    #     beta_expanded = beta.unsqueeze(0)  # [1, num_alleles, num_mutations]
        
    #     beta_per_gene = beta_expanded * gene_masks_expanded
        
    #     obs_per_gene = self.functional_observation_counts.unsqueeze(0) * self.gene_masks.float()
        

    #     theta_expanded = theta.view(1, -1, 1)
    #     expected_per_gene = (beta_per_gene * theta_expanded).sum(dim=1)
        
    #     obs_sums = obs_per_gene.sum(dim=1, keepdim=True).clamp(min=eps)
    #     exp_sums = expected_per_gene.sum(dim=1, keepdim=True).clamp(min=eps)
        
    #     valid_genes = (obs_sums.squeeze() > 1.0).float()
        
    #     P_obs = obs_per_gene / obs_sums
    #     P_exp = expected_per_gene / exp_sums
        
    #     M = 0.5 * (P_obs + P_exp)
        
    #     js_per_gene = 0.5 * (
    #         (P_obs * torch.log((P_obs + eps) / (M + eps))).sum(dim=1) +
    #         (P_exp * torch.log((P_exp + eps) / (M + eps))).sum(dim=1)
    #     )
        
    #     total_divergence = (js_per_gene * valid_genes).sum()
        
    #     if debug:
    #         for i in range(self.num_genes):
    #             if valid_genes[i] > 0:
    #                 print(f"{self.gene_names[i]}: JS={js_per_gene[i]:.4f}")
        
    #     return total_divergence


    #     beta_functional = beta[:, self.functional_indices]  
        
    #     # DIAGNOSTIC: Check beta_functional

    #     obs_functional = self.functional_observation_counts[self.functional_indices]
        

    #     expected_unnorm = torch.matmul(theta.unsqueeze(0), beta_functional).squeeze(0)

    #     P_observed = obs_functional / (obs_functional.sum() + eps)
    #     P_expected = expected_unnorm / (expected_unnorm.sum() + eps)
        
    #     if use_js:
    #         # Jensen-Shannon Divergence (symmetric, always finite)
    #         M = 0.5 * (P_observed + P_expected)
    #         kl_obs_m = torch.sum(P_observed * torch.log((P_observed + eps) / (M + eps)))
    #         kl_exp_m = torch.sum(P_expected * torch.log((P_expected + eps) / (M + eps)))
    #         divergence_loss = 0.5 * (kl_obs_m + kl_exp_m)
            
      
    #     else:
    #         # Standard KL Divergence (asymmetric)
    #         divergence_loss = torch.sum(P_observed * torch.log((P_observed + eps) / (P_expected + eps)))
            
   
 
    #     return divergence_loss

    

    
    def construct_beta_sparse(self, base_probs):
        batch_size = base_probs.size(0)
        
        sparse_flat = base_probs.view(batch_size, -1)
        
        beta_batch = []
        for b in range(batch_size):
            beta = torch.zeros(self.num_alleles, self.num_mutations, device=base_probs.device)
            beta[self.allele_indices, self.mut_indices] = sparse_flat[b, self.sparse_flat_indices]
            
            row_sums = beta.sum(dim=1, keepdim=True)
            beta = beta / (row_sums + 1e-10)
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
        allele_mu, allele_logvar, base_mu, base_logvar = self.encoder(mut_counts)
        allele_z = self.reparameterize(allele_mu, allele_logvar)
        base_z = self.reparameterize(base_mu, base_logvar)
        
        allele_props = F.softmax(allele_z, dim=-1)
        # normalized_allele_props = self.apply_normalization(allele_props)
        
        base_probs = F.softmax(base_z, dim=-1)
        beta = self.construct_beta_sparse(base_probs)
        
        # mutation_probs = torch.bmm(normalized_allele_props.unsqueeze(1), beta).squeeze(1)
        mutation_probs = torch.bmm(allele_props.unsqueeze(1), beta).squeeze(1)
        
        # Return normalized_allele_props instead of allele_props for loss calculation
        return (mutation_probs, beta, allele_mu, allele_logvar,
                base_mu, base_logvar, allele_props, base_probs)
    
    def loss_function(self, mutation_probs, mut_counts, 
                    allele_mu, allele_logvar,
                    base_mu, base_logvar, beta, 
                    allele_props, base_probs,
                    kld_weight=1.0, base_kld_weight=0.001, functional_kld_weight = 0.6, debug_functional = False):
        
        strength_mask = self.get_strength_mask()
        beta_adjusted = beta * strength_mask.unsqueeze(0)
        row_sums = beta_adjusted.sum(dim=2, keepdim=True)
        beta_adjusted = beta_adjusted / (row_sums + 1e-10)
        
        mutation_probs_adjusted = torch.bmm(
            allele_props.unsqueeze(1), 
            beta_adjusted
        ).squeeze(1)
        
        log_likelihood = torch.distributions.Multinomial(
            total_count=self.total_mut_counts,
            probs=mutation_probs_adjusted + 1e-10
        ).log_prob(mut_counts)
        
        allele_kld = -0.5 * torch.sum(
            1 + allele_logvar - allele_mu.pow(2) - allele_logvar.exp()
        )
        
        base_mu_flat = base_mu.view(-1)
        base_logvar_flat = base_logvar.view(-1)
        base_kld = -0.5 * torch.sum(
            1 + base_logvar_flat - self.sparse_prior_logvar
            - (base_logvar_flat.exp() + (base_mu_flat - self.sparse_prior_mu).pow(2)) / self.sparse_prior_logvar.exp()
        )
        
        functional_kl = self.compute_functional_kl_divergence(
            allele_props.squeeze(0),
            beta_adjusted.squeeze(0),
            debug = debug_functional,
            use_js=True
        )
        
        base_probs_reshaped = base_probs.view(base_probs.size(0), -1, 6)  
        base_entropy = -(base_probs_reshaped * torch.log(base_probs_reshaped + 1e-10)).sum(dim=-1).mean()
        
        # loss = -log_likelihood + kld_weight * allele_kld + base_kld_weight * base_kld + functional_kld_weight*functional_kl + 100*base_entropy
        loss = -log_likelihood + kld_weight * allele_kld + base_kld_weight * base_kld + 100*base_entropy
        
        return loss, -log_likelihood, allele_kld, base_kld, functional_kl
        
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
                        sparse_prior_mu, sparse_prior_logvar, mapping_indices, sample,db,
                        seed=0, num_iterations=3000, lr=0.005, print_every=500, 
                         initial_strength=None, strength_lb=None, strength_ub=None, 
                          functional_observation_counts = None,
                         functional_indices = None,
                           gene_masks = None,
                            gene_names = None,
                            bam_id = None
                         ):
   
    torch.manual_seed(seed)

    model = VAE(
        mut_counts.shape[0], len(valid_alleles), num_sparse_entries,
        total_mut_counts, sparse_prior_mu, sparse_prior_logvar,
        mapping_indices, sample,db,
        hidden=512,
        dropout=0,
        initial_strength=initial_strength,
        strength_lb=strength_lb,
        strength_ub=strength_ub,

        functional_observation_counts = functional_observation_counts,
        functional_indices = functional_indices,

        gene_masks = gene_masks,
        gene_names = gene_names
    ).to(device)
    
    model.encoder.initialize_base_params(sparse_prior_mu, sparse_prior_logvar)
    
    allele_params = [p for name, p in model.named_parameters() if 'allele' in name]
    base_params = [p for name, p in model.named_parameters() if 'base' in name]
    
    optimizer = optim.Adam([
        {'params': allele_params, 'lr': lr},
        {'params': base_params, 'lr': lr * 0.002}  
    ])
    
    # Train the model
    model.train()
    loss_history = []
    
    for step in range(num_iterations):
        optimizer.zero_grad()
        
        (mutation_probs, beta,
        allele_mu, allele_logvar,
        base_mu, base_logvar,
        allele_props,
        base_probs) = model(mut_counts.unsqueeze(0))
        debug_functional = (step % print_every == 0)
        # debug_functional = False
        loss, recon_loss, allele_kl, base_kl, functional_kl = model.loss_function(
            mutation_probs.squeeze(0),
            mut_counts,
            allele_mu.squeeze(0),
            allele_logvar.squeeze(0),
            base_mu.squeeze(0),
            base_logvar.squeeze(0),
            beta,
            allele_props, base_probs,
            kld_weight=1.0,
            base_kld_weight=0.001,
            functional_kld_weight = 600000,

            debug_functional = debug_functional
        )
        
        loss_history.append((loss.item(), recon_loss.item(), allele_kl.item(), base_kl.item(), functional_kl.item()))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if step % print_every == 0:
            print(f"Seed {seed}, Step {step:4d}: Loss = {loss.item():.2f} "
                f"(Recon: {recon_loss.item():.2f}, Allele KL: {allele_kl.item():.2f}, "
                f"Base KL: {base_kl.item():.2f}, "
                f"Functional KL: {functional_kl.item():.2f}")


    # Inference
    model.eval()
    with torch.no_grad():
        # Get final outputs
        (mutation_probs, beta, 
         allele_mu, allele_logvar,
         base_mu, base_logvar, 
        allele_props, 
           base_probs) = model(mut_counts.unsqueeze(0))

        theta = F.softmax(allele_mu, dim=-1).squeeze(0)
        mismatches = model.get_functional_coverage_mismatches(
            theta, 
            beta.squeeze(0),
            use_js = True, 
            top_k=600         
        )
        print("\nTop Variants with Coverage Mismatches:")
        print(f"{'Rank':<6} {'Variant':<30} {'Expected':<12} {'Observed':<12} {'Diff':<10} {'|Div|':<12}")
        print("-" * 90)
        
        for rank, (variant_str, exp_count, obs_count, div_contrib) in enumerate(mismatches, 1):
            diff = exp_count - obs_count
            marker = " ⚠️HALLUC" if obs_count < 1 and exp_count > 5 else ""
            print(f"{rank:<6} {variant_str:<30} {exp_count:<12.2f} {obs_count:<12.2f} {diff:<10.2f} {div_contrib:<12.6f}{marker}")

        theta_norm = model.apply_normalization(theta).cpu()
        # beta_adjusted = beta*model.get_strength_mask().squeeze(0)
        strength_mask = model.get_strength_mask()  # Shape: (num_alleles, num_mutations)
        beta_squeezed = beta.squeeze(0)  # Remove batch dimension first!
        beta_adjusted = beta_squeezed * strength_mask  # Now shapes match
        row_sums = beta_adjusted.sum(dim=1, keepdim=True)
        beta_adjusted = beta_adjusted / (row_sums + 1e-10)
        # torch.save(model.state_dict(), f'models/{bam_id}_vae_weights.pt')
        return theta_norm, theta, loss_history, beta_adjusted.cpu()
        
def run_vae_multi_seed(total_mut_counts, valid_alleles, mut_counts,  
                       num_sparse_entries, sparse_prior_mu, sparse_prior_logvar, 
                       mapping_indices, sample,db,
                       seeds=[42, 123, 456], num_iterations=3000, lr=0.005, 
                       print_every=500, plot_loss=False, save_plot=False,  
                       initial_strength=None, strength_lb=None, strength_ub=None, 
                     functional_observation_counts = None,
                       functional_indices = None,
                        gene_masks = None,
                        gene_names = None,
                        bam_id = None,
                       ):

    results = {}
    best_seed = None
    best_final_loss = float('inf')
    
    for seed in seeds:
        print(f"\n=== Running VAE with seed {seed} ===")
        theta, theta_un, loss_history, learned_beta = run_vae_single_seed(
            total_mut_counts, valid_alleles, mut_counts, 
            num_sparse_entries, sparse_prior_mu, sparse_prior_logvar, 
            mapping_indices, sample,db,
            seed=seed, num_iterations=num_iterations, lr=lr, print_every=print_every,
            initial_strength=initial_strength, strength_lb=strength_lb, strength_ub=strength_ub,
            functional_observation_counts = functional_observation_counts,
            functional_indices = functional_indices,
            gene_masks = gene_masks,
            gene_names = gene_names,
            bam_id = bam_id
        )
        
        # Extract final losses
        final_loss = loss_history[-1][0]
        final_recon = loss_history[-1][1]
        final_allele_kl = loss_history[-1][2]
        final_base_kl = loss_history[-1][3]
        
        # Store results
        results[seed] = {
            'theta': theta,
            'theta_un': theta_un,
            'loss_history': loss_history,
            'final_loss': final_loss,
            'final_recon': final_recon,
            'final_allele_kl': final_allele_kl,
            'final_base_kl': final_base_kl,
            'learned_beta': learned_beta
        }
        
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_seed = seed
            print(f"New best model with final loss: {best_final_loss:.2f}")

    # Print summary of results
    print("\n=== Multi-Seed VAE Results ===")
    print(f"Best model had seed {best_seed} with final loss: {best_final_loss:.2f}")
    print("All seeds final losses:")
    for seed in seeds:
        r = results[seed]
        print(f"  Seed {seed}: Total={r['final_loss']:.2f} "
              f"(Recon={r['final_recon']:.2f}, A_KL={r['final_allele_kl']:.2f}, "
              f"B_KL={r['final_base_kl']:.2f})" + 
              (" <- Best" if seed == best_seed else ""))
    
    best_result = results[best_seed]
    return best_result['theta'],best_result['theta_un'],results, best_result['learned_beta']


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
                if allele.extended_allele_vector[variant.index] != 1:
                    continue
                
                lb, ub = allele.get_position_strength(
                    pos_obj, 
                    variant.variant
                )
       
                lb_normalized = lb / sample.expected_coverage
                ub_normalized = ub / sample.expected_coverage
                
 
                # Assert valid bounds
                if variant.index in sample.valid_indices:
                    mut_idx = sample.valid_indices.index(variant.index)
                    
                    strength_lb[allele_idx, mut_idx] = lb_normalized
                    strength_ub[allele_idx, mut_idx] = ub_normalized
                    
                    initial_strength[allele_idx, mut_idx] = (lb_normalized + ub_normalized) / 2.0
                    
                    if not has_bounds:
                        print(f"\n{allele_name}:")
                        has_bounds = True
                    

    return initial_strength, strength_lb, strength_ub

@timeit
def run_vae(total_mut_counts, valid_alleles, mut_counts,
            sample=None, db=None, num_iterations=3000, bam_id = None):
    print('running 1017 version')
    # from V9_helper_1017 import create_sparse_priors_V2
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
    theta, theta_un, _, learned_beta = run_vae_multi_seed(
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
        gene_masks = gene_masks,
        gene_names = gene_names,
        bam_id = bam_id
    )
    
    return theta, theta_un, learned_beta

