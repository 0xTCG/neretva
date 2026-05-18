
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from vae_helper import BASES, classify_mutation_type, get_confidence_for_type, logit
from common_0620B import timeit

@timeit
def run_ilp(total_mut_counts, valid_alleles, mut_counts,
            sample=None, db=None, max_cn=4, base_cost_weight=0.01, time_limit=600):

    num_alleles = len(sample.valid_alleles)
    num_mutations = len(sample.valid_indices)
    obs = mut_counts.numpy() if isinstance(mut_counts, torch.Tensor) else np.array(mut_counts)


    sparse_to_mut = {}

    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            for base_idx, base in enumerate(BASES):
                if base in pos_obj.variants:
                    variant = pos_obj.variants[base]
                    if variant.index in sample.valid_indices:
                        mut_idx = sample.valid_indices.index(variant.index)
                        sparse_to_mut[(allele_idx, sparse_idx, base_idx)] = mut_idx
            sparse_idx += 1
    sparse_pos_count = sparse_idx

    base_costs = {}

    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            allele_variant = None
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variant = variant
                    break

            if allele_variant:
                mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
                confidence = get_confidence_for_type(allele.gene.name, allele.name, pos_obj.position, mut_type)
                n_other = len(pos_obj.variants) - 1
                non_var_conf = (1.0 - confidence) / n_other if n_other > 0 else 0.0

                for base_idx, base in enumerate(BASES):
                    if allele_variant.variant == base:
                        base_costs[(allele_idx, sparse_idx, base_idx)] = -float(logit(confidence))
                    else:
                        base_costs[(allele_idx, sparse_idx, base_idx)] = -float(logit(non_var_conf)) if non_var_conf > 1e-7 else 0.0

            sparse_idx += 1


    m = gp.Model("KIR_ILP")
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1

    c = m.addVars(num_alleles, vtype=GRB.INTEGER, lb=0, ub=max_cn, name="c")

    b = {}
    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            for base_idx in range(len(BASES)):
                b[(allele_idx, sparse_idx, base_idx)] = m.addVar(
                    vtype=GRB.BINARY, name=f"b_{allele_idx}_{sparse_idx}_{base_idx}"
                )
            sparse_idx += 1

    z = {}
    for key in sparse_to_mut:
        a, s, k = key
        z[key] = m.addVar(lb=0, ub=max_cn, vtype=GRB.CONTINUOUS, name=f"z_{a}_{s}_{k}")

    e_pos = m.addVars(num_mutations, lb=0, name="ep")
    e_neg = m.addVars(num_mutations, lb=0, name="en")

    m.update()


    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            m.addConstr(
                gp.quicksum(b[(allele_idx, sparse_idx, k)] for k in range(len(BASES))) == 1,
                name=f"onebase_{allele_idx}_{sparse_idx}"
            )
            sparse_idx += 1

    for (a, s, k), var in z.items():
        m.addConstr(var <= max_cn * b[(a, s, k)], name=f"lin1_{a}_{s}_{k}")
        m.addConstr(var <= c[a], name=f"lin2_{a}_{s}_{k}")
        m.addConstr(var >= c[a] - max_cn * (1 - b[(a, s, k)]), name=f"lin3_{a}_{s}_{k}")

    cov = sample.expected_coverage
    mut_to_z = {}
    for key, mut_idx in sparse_to_mut.items():
        mut_to_z.setdefault(mut_idx, []).append(z[key])

    for mut_idx in range(num_mutations):
        contributors = mut_to_z.get(mut_idx, [])
        if contributors:
            m.addConstr(
                cov * gp.quicksum(contributors) + e_neg[mut_idx] - e_pos[mut_idx] == obs[mut_idx],
                name=f"recon_{mut_idx}"
            )
        else:
            m.addConstr(e_pos[mut_idx] == obs[mut_idx], name=f"recon_nc_{mut_idx}")
            m.addConstr(e_neg[mut_idx] == 0, name=f"recon_ncn_{mut_idx}")

    recon_obj = gp.quicksum(e_pos[i] + e_neg[i] for i in range(num_mutations))

    base_cost_obj = gp.quicksum(
        base_costs[key] * b[key]
        for key in b
    )

    m.setObjective(recon_obj + base_cost_weight * base_cost_obj, GRB.MINIMIZE)

    print(f"ILP: {num_alleles} alleles, {num_mutations} mutations, "
          f"{sparse_pos_count} sparse positions, {len(z)} linearized products")

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible")
        m.computeIIS()
        m.write("infeasible.ilp")
        return None, None, None, None

    allele_cns = {}
    for a in range(num_alleles):
        cn = int(round(c[a].X))
        if cn > 0:
            allele = sample.valid_alleles[a]
            name = f"{allele.gene.name}*{allele.name}"
            allele_cns[name] = cn

    recon_err = sum(e_pos[i].X + e_neg[i].X for i in range(num_mutations))
    base_cost_val = sum(
        base_costs[key] * b[key].X
        for key in b
    )

    print(f"\n=== ILP Results ===")
    print(f"Reconstruction error: {recon_err:.2f}")
    print(f"Base selection cost:  {base_cost_val:.4f}")
    print(f"Objective:            {m.ObjVal:.4f}")
    print(f"\nAllele calls (CN > 0):")
    for name, cn in sorted(allele_cns.items()):
        print(f"  {name}: CN={cn}")

    # Count base deviations
    num_deviations = 0
    total_selections = 0
    sparse_idx = 0
    for allele_idx, allele in enumerate(sample.valid_alleles):
        for pos_obj in allele.generatable_positions:
            cn = int(round(c[allele_idx].X))
            if cn == 0:
                sparse_idx += 1
                continue
            db_base = None
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    db_base = variant.variant
                    break
            selected = None
            for base_idx in range(len(BASES)):
                if b[(allele_idx, sparse_idx, base_idx)].X > 0.5:
                    selected = BASES[base_idx]
                    break
            if selected and db_base and selected != db_base:
                num_deviations += 1
            total_selections += 1
            sparse_idx += 1

    print(f"Base deviations from database: {num_deviations}/{total_selections}")

    return allele_cns, m.ObjVal, recon_err, base_cost_val