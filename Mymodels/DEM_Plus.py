import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM_Plus(nn.Module):
    def __init__(self, cnn, semantic_dim=300, hidden_dim=1024):
        super().__init__()
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False
        visual_dim = 128
        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, visual_dim),
            nn.LeakyReLU()
        ])
        self.MLP = nn.Sequential(*[
            nn.Linear(2*visual_dim, visual_dim),
            nn.LeakyReLU(),
            nn.Linear(visual_dim, visual_dim//2),
            nn.LeakyReLU(),
            nn.Linear(visual_dim, visual_dim//2),
            nn.LeakyReLU(),
            nn.Linear(visual_dim//2, visual_dim//2//2),
            nn.LeakyReLU()
        ])
        self.out = nn.Sequential(*[
            nn.Linear(visual_dim//2//2, 1),
            nn.LeakyReLU()
        ])

    def forward(self, image, label, word_embeddings):
        n_class = word_embeddings.size(0)
        batch_size = image.size(0)

        self.cnn.eval()
        visual_emb = self.cnn(image)

        visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
        semantic_emb = self.word_emb_transformer(word_embeddings).repeat(batch_size, 1)
        concat = torch.cat([visual_emb, semantic_emb], -1)
        x = self.MLP(concat)
        x = x.view(batch_size, -1)
        return x

    def get_loss(self, image, label, word_embeddings):
        x = self.forward(image, label, word_embeddings)
        loss = nn.CrossEntropyLoss()(x, label)
        return loss

    def predict(self, image, word_embeddings):
        with torch.no_grad():
            n_class = word_embeddings.size(0)
            batch_size = image.size(0)

            self.cnn.eval()
            visual_emb = self.cnn(image)

            visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
            semantic_emb = self.word_emb_transformer(word_embeddings).repeat(batch_size, 1)

            score = (visual_emb - semantic_emb).norm(dim=1).view(batch_size, n_class)
            pred = score.min(1)[1]

        return pred.detach().cpu().numpy()


