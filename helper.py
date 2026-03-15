def print_variant_counts(sample, db):
    positions = {}
    position_has_func = {}
    for variant_idx in sample.valid_indices:
        variant = db.variants()[variant_idx]
        key = (variant.gene.name, variant.pos)
        if key not in positions:
            positions[key] = []
            position_has_func[key] = False

        # Mark wildtype variants with *
        variant_label = variant.variant
        if variant.is_wildtype:
            variant_label += "*"
        
        if variant.is_major:
            variant_label += '^'
            position_has_func[key] = True

        positions[key].append((variant_label, variant.coverage.count))
    for (gene, pos) in sorted(positions.keys()):
        variants = sorted(positions[(gene, pos)])
        variant_strs = [f"{var}:{count}" for var, count in variants]
        mark = ''
        if position_has_func[gene, pos]: mark = '^'
        print(f"{gene}:{pos}{mark} → {', '.join(variant_strs)}")
        