import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DEMORdinary(nn.Module):
    def __init__(self, cnn, semantic_dim=300, hidden_dim=1024):
        super().__init__()
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False
        visual_dim = self.cnn.out_size
        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim),
            nn.ReLU()
        ])

    def forward(self, image, label, word_embeddings):
        self.cnn.eval()
        visual_emb, _ = self.cnn(image)
        word_embed = self.word_emb_transformer(word_embeddings[label])
        return word_embed, visual_emb

    def get_loss(self, image, label, word_embeddings):
        h_loss = 0
        #for p in self.word_emb_transformer.parameters():
            #h_loss += 0.0005*torch.sum(torch.abs(p))
        word_embed, visual_emb = self.forward(image, label, word_embeddings)
        loss = torch.sum(1-F.cosine_similarity(word_embed, visual_emb))
        #loss = nn.MSELoss()(word_embed, visual_emb)
        return loss

    def predict(self, image, word_embeddings):
        with torch.no_grad():
            n_class = word_embeddings.size(0)
            batch_size = image.size(0)

            self.cnn.eval()
            visual_emb, _ = self.cnn(image)

            visual_emb_ex = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
            semantic_emb = self.word_emb_transformer(word_embeddings).repeat(batch_size, 1)

            #score = (visual_emb_ex - semantic_emb).norm(dim=1).view(batch_size, n_class)
            score = 1-F.cosine_similarity(visual_emb_ex, semantic_emb).view(batch_size, n_class)
            pred = score.min(1)
            pred = pred[1]

        return pred.detach().cpu().numpy(), visual_emb.cpu().numpy()


