import math
import torch
import torch.nn as nn

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class MM_MLP(nn.Module):
    def __init__(self, input_dim_g: int, input_dim_h: int, hidden_size_g: int, hidden_size_h: int, hidden_size_mm: int, dropout=0.25, n_classes=4):
        super(MM_MLP, self).__init__()  # Inherited from the parent class nn.Module
        self.fc_genomic = nn.Sequential(nn.Linear(input_dim_g, hidden_size_g), nn.ReLU(), nn.Dropout(p=dropout))
        self.fc_histology = nn.Sequential(nn.Linear(input_dim_h, hidden_size_h), nn.ReLU(), nn.Dropout(p=dropout))
        self.fc_mm = nn.Sequential(nn.Linear(hidden_size_g+hidden_size_h, hidden_size_mm), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier_g = nn.Linear(hidden_size_g, n_classes)  # 2nd Full-Connected Layer: hidden node -> output
        self.classifier_h = nn.Linear(hidden_size_h, n_classes)  # 2nd Full-Connected Layer: hidden node -> output
        self.classifier = nn.Linear(hidden_size_mm, n_classes)  # 2nd Full-Connected Layer: hidden node -> output

    def forward(self, g_x, h_x):
        g_features = self.fc_genomic(g_x)
        h_features = self.fc_histology(h_x)
        mm_features = self.fc_mm(torch.cat([g_features, h_features], dim=-1))
        g_logits = self.classifier_g(g_features)
        g_probs = torch.softmax(g_logits, dim=-1)
        h_logits = self.classifier_h(h_features)
        h_probs = torch.softmax(h_logits, dim=-1)
        logits = self.classifier(mm_features)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs, g_logits, g_probs, h_logits, h_probs, mm_features, h_features, g_features

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_genomic = nn.DataParallel(self.fc_genomic, device_ids=device_ids).to('cuda:0')
            self.fc_histology = nn.DataParallel(self.fc_histology, device_ids=device_ids).to('cuda:0')
            self.fc_mm = nn.DataParallel(self.fc_mm, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_genomic = self.fc_genomic.to(device)
            self.fc_histology = self.fc_histology.to(device)
            self.fc_mm = self.fc_mm.to(device)

        self.classifier_g = self.classifier_g.to(device)
        self.classifier_h = self.classifier_h.to(device)
        self.classifier = self.classifier.to(device)