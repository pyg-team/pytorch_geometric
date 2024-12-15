import numpy as np
import torch
import torch.nn as nn


class RelationAwareDiscriminator(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_relations):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, embedding_dim, embedding_dim))

    def forward(self, u, v, r):
        u_embed = self.node_embeddings(u)
        v_embed = self.node_embeddings(v)
        relation_matrix = self.relation_matrices[r]
        score = torch.matmul(u_embed, relation_matrix)
        score = torch.matmul(score, v_embed.T)
        return torch.sigmoid(score)

    def loss(self, pos_samples, neg_samples, incorrect_relations, lambda_reg):
        # Positive samples loss
        u_pos, v_pos, r_pos = pos_samples
        pos_scores = self.forward(u_pos, v_pos, r_pos)
        loss_pos = -torch.log(pos_scores + 1e-8).mean()

        # Negative samples from incorrect relations
        u_neg, v_neg, r_neg = incorrect_relations
        neg_scores = self.forward(u_neg, v_neg, r_neg)
        loss_neg_rel = -torch.log(1 - neg_scores + 1e-8).mean()

        # Negative samples from generator
        u_fake, v_fake, r_fake = neg_samples
        fake_scores = self.forward(u_fake, v_fake, r_fake)
        loss_fake = -torch.log(1 - fake_scores + 1e-8).mean()

        # Regularization
        reg_loss = sum(torch.norm(param, 2) for param in self.parameters())

        return loss_pos + loss_neg_rel + loss_fake + lambda_reg * reg_loss


class GeneralizedGenerator(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_relations, sigma=1.0):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, embedding_dim, embedding_dim))
        self.sigma = sigma
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                 nn.ReLU(),
                                 nn.Linear(embedding_dim, embedding_dim))

    def forward(self, u, r):
        u_embed = self.node_embeddings(u)
        relation_matrix = self.relation_matrices[r]
        mean = torch.matmul(u_embed, relation_matrix)
        gaussian_noise = torch.randn_like(mean) * self.sigma
        latent_sample = mean + gaussian_noise
        return self.mlp(latent_sample)

    def loss(self, u, r, fake_v, discriminator, lambda_reg):
        fake_scores = discriminator(u, fake_v, r)
        reg_loss = sum(torch.norm(param, 2) for param in self.parameters())
        return -torch.log(fake_scores + 1e-8).mean() + lambda_reg * reg_loss


class HeGAN:
    def __init__(self, num_nodes, embedding_dim, num_relations, sigma=1.0,
                 lambda_reg=1e-4):
        self.generator = GeneralizedGenerator(num_nodes, embedding_dim,
                                              num_relations, sigma)
        self.discriminator = RelationAwareDiscriminator(
            num_nodes, embedding_dim, num_relations)
        self.lambda_reg = lambda_reg

    def train(self, hetero_data, num_epochs, batch_size, lr=0.001):
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                          lr=lr)

        for epoch in range(num_epochs):
            # Train discriminator
            for _ in range(5):
                pos_samples, neg_samples, incorrect_relations = self.sample(
                    hetero_data, batch_size)
                disc_optimizer.zero_grad()
                disc_loss = self.discriminator.loss(pos_samples, neg_samples,
                                                    incorrect_relations,
                                                    self.lambda_reg)
                disc_loss.backward()
                disc_optimizer.step()

            # Train generator
            for _ in range(1):
                u, r = self.sample_generator_inputs(hetero_data, batch_size)
                gen_optimizer.zero_grad()
                fake_v = self.generator(u, r)
                gen_loss = self.generator.loss(u, r, fake_v,
                                               self.discriminator,
                                               self.lambda_reg)
                gen_loss.backward()
                gen_optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}" +
                  f", Discriminator Loss: {disc_loss.item()}" +
                  f", Generator Loss: {gen_loss.item()}")

    def sample(self, hetero_data, batch_size):
        edge_types = list(hetero_data.edge_types)
        sampled_type = np.random.choice(edge_types)
        edge_index = hetero_data[sampled_type].edge_index

        indices = torch.randint(0, edge_index.size(1), (batch_size, ))
        u, v = edge_index[:, indices]
        r = torch.tensor([edge_types.index(sampled_type)] * batch_size)

        neg_v = torch.randint(0, hetero_data[sampled_type].num_nodes,
                              (batch_size, ))
        incorrect_r = torch.randint(0, len(edge_types), (batch_size, ))

        return (u, v, r), (u, neg_v, r), (u, v, incorrect_r)

    def sample_generator_inputs(self, hetero_data, batch_size):
        edge_types = list(hetero_data.edge_types)
        sampled_type = np.random.choice(edge_types)
        edge_index = hetero_data[sampled_type].edge_index

        indices = torch.randint(0, edge_index.size(1), (batch_size, ))
        u, _ = edge_index[:, indices]
        r = torch.tensor([edge_types.index(sampled_type)] * batch_size)
        return u, r
