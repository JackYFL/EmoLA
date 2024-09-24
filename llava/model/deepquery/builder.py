import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQuery(nn.Module):
    def __init__(self, seq_len=20, n_embd=128, n_layer=32, n_head=32):
        super(DeepQuery, self).__init__()
        self.input_query = nn.Embedding(seq_len, n_embd)
        self.encoder = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.Tanh(),
            nn.Linear(n_embd, n_layer*2*n_embd*n_head) # dim of past keys and values
        )
        # self._init_weights()
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.dropout = nn.Dropout()
        
    def _init_weights(self):
        # Initialize weights of nn.Linear layers using Xavier initialization
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, bsz):
        input_tokens = self.input_query.weight.unsqueeze(0).repeat(bsz, 1, 1)
        past_key_values = self.encoder(input_tokens) #bsz, seqlen, emb(n_layer*2*n_head*n_embd)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.n_layer*2, self.n_head, self.n_embd) #bsz, seqlen, n_layer*2, n_head, n_embd
        past_key_values = self.dropout(past_key_values) #bsz, seqlen, n_layer*2, n_head, n_embd
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #n_layer * (2, bsz, n_head, seqlen, n_embd)
        return past_key_values

class IConDeepQuery(nn.Module):
    def __init__(self, seq_len=20, n_embd=128, n_layer=32, n_head=32, n_imgtokens=576):
        super(IConDeepQuery, self).__init__()
        self.input_query = nn.Embedding(seq_len, n_embd)
        self.input_query_new = nn.Embedding(seq_len, n_embd*n_head)
        self.pseudo_img_tokens = nn.Embedding(n_imgtokens, n_embd*n_head)
        self.cls_token_metanet = nn.Sequential(
            nn.Linear(n_embd*n_head, n_embd*n_head//32),
            nn.GELU(),
            nn.Linear(n_embd*n_head//32, n_embd*n_head) # dim of past keys and values
        )

        self.cls_token_encoder = nn.Sequential(
            nn.Linear(n_embd*n_head, n_embd*n_head//32),
            nn.GELU(),
            nn.Linear(n_embd*n_head//32, n_layer*2*n_embd*n_head) # dim of past keys and values
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.Tanh(),
            nn.Linear(n_embd, n_layer*2*n_embd*n_head) # dim of past keys and values
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(n_embd*n_head, n_embd*n_head//32),
            nn.Tanh(),
            nn.Linear(n_embd*n_head//32, n_layer*2*n_embd*n_head) # dim of past keys and values
        )
        
        self.img_encoder = nn.Sequential(
            nn.Linear(n_embd*n_head, n_embd*n_head//32),
            nn.Tanh(),
            nn.Linear(n_embd*n_head//32, n_layer*2*n_embd*n_head) # dim of past keys and values
        )
        
        self.token_mixer = nn.Sequential(
            nn.Linear(n_imgtokens, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len)
        )
        
        self.channel_mixer = nn.Sequential(
            nn.Linear(n_embd*n_head, n_embd*n_head//32),
            nn.ReLU(),
            nn.Linear(n_embd*n_head//32, n_layer*2*n_embd*n_head)
        )
        
        self.LN1 = nn.LayerNorm(n_embd*n_head)
        self.LN2 = nn.LayerNorm(n_embd*n_head)
        
        self._init_weights()
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.dropout = nn.Dropout()
        
    def _init_weights(self):
        # Initialize weights of nn.Linear layers using Xavier initialization
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image_tokens): # image_tokens (bsz, img_seq_length, D)
        bsz, _, _ = image_tokens.shape
        cls_token = image_tokens[:, 0, :]
        
        #########Cross Attention##########
        cls_token = cls_token.unsqueeze(1)
        # mean_img_tokens = image_tokens[:, 1:, :].mean(dim=1)
        input_tokens = self.input_query_new.weight.unsqueeze(0).repeat(bsz, 1, 1)
        fusion_query = self.cross_att(input_tokens, cls_token.to(input_tokens.device), cls_token.to(input_tokens.device))
        # fusion_query = self.cross_att(input_tokens, mean_img_tokens.to(input_tokens.device), mean_img_tokens.to(input_tokens.device))
        past_key_values = self.query_encoder(fusion_query) #bsz, seqlen, emb(n_layer*2*n_head*n_embd)
        past_key_values = past_key_values.view(bsz, self.seq_len, self.n_layer*2, self.n_head, self.n_embd) #bsz, seqlen, n_layer*2, n_head, n_embd
        past_key_values = self.dropout(past_key_values) #bsz, seqlen, n_layer*2, n_head, n_embd
        self.past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #n_layer * (2, bsz, n_head, seqlen, n_embd)        
        ##################################
