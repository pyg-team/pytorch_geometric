# Implement different sampling techniques
# Calculate p_{u,v} and p_v and return \alpha_{u,v} = p_{u,v} / p_v
# Calculate loss normalization \lambda_v = |V| * p_v
# p_v is the target node!

# For random node or random edge samplers, compute p_{u,v} and p_v analytically
# For other samplers, pre-generate and cache N sampled results:
# => Count number of occurrences C_v and C_{u,v} for each node and each edge.
# => Set \alpha_{u,v} to C_{u,v}/C_{v} and \lambda_v to C_v/N.

# Random node sampler:
# Sample nodes with node probability according to the in-degree of nodes.

# Random edge sampler:
# Sample edges with edge probability according to 1/deg(u) + 1/deg(v)

# Random walk sampler (multi-dimensional):
# r root nodes selected uniformly and each walker goes h hops
# Multi-dimensional random walk sampler as in:
# Ribeiro and Towsley - Estimating and sampling graphs with multidimensional
# random walks

# Add Flickr/Yelp and Amazon datasets.
