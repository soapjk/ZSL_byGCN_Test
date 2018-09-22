import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM(nn.Module):
    def __init__(self, cnn, semantic_dim=300, hidden_dim=1024):
        super().__init__()
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False
        visual_dim = 32
        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 32),
            nn.LeakyReLU()
        ])
        self.attr_emb_transformer = nn.Sequential(*[
            nn.Linear(30, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU()
        ])

        self.w_lat = nn.Sequential(*[
            nn.Linear(64, 32),
            nn.LeakyReLU()
        ])
        self.w_att = nn.Sequential(*[
            nn.Linear(64, 32),
            nn.LeakyReLU()
        ])

    def forward(self, image, label, word_embeddings):
        self.cnn.eval()
        visual_emb = self.cnn(image)
        word_embed = self.word_emb_transformer(word_embeddings[label])
        return word_embed, visual_emb

    def get_loss(self, image, label, word_embeddings, attribute):
        word_embed, visual_emb = self.forward(image, label, word_embeddings)
        visual_emb_lat = self.w_lat(visual_emb)
        visual_emb_attr = self.w_att(visual_emb)
        all_attribute_emb = self.attr_emb_transformer(attribute)
        attribute_emb = all_attribute_emb[label]

        loss = F.mse_loss(word_embed, visual_emb_lat)+F.mse_loss(attribute_emb, visual_emb_attr)

        return loss

    def predict(self, image, word_embeddings, attribute):
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


