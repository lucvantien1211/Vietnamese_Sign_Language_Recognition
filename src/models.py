'''
Models definition

Credit to AI VIET NAM: https://aivietnam.edu.vn for model architectures
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CRNN(nn.Module):
    '''
    Convolutional Recurrent Neural Network (CRNN) for video classification

    This model combines a CNN backbone for spatial feature extraction
    and an LSTM layer for temporal sequence modeling.

    Args:
        num_classes (int)                  : number of output classes,
                                             default: 100
        hidden_size (int)                  : hidden dimension of the
                                             LSTM layer, default: 256
        resnet_pretrained_weights (object) : pretrained weights for
                                             ResNet18 backbone,
                                             default: None

    Attributes:
        cnn (nn.Sequential) : ResNet18 feature extractor without
                              classification layers
        feature_dim (int)   : feature dimension extracted from CNN
        pool (nn.Module)    : adaptive average pooling layer
        rnn (nn.LSTM)       : temporal sequence modeling layer
        dropout (nn.Dropout): dropout regularization layer
        fc (nn.Linear)      : final classification layer
    '''
    
    def __init__(self, num_classes=100, hidden_size=256, resnet_pretrained_weights=None):
        super(CRNN, self).__init__()
        resnet = models.resnet18(weights=resnet_pretrained_weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(self.feature_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''
        Forward pass of the CRNN model

        Args:
            x (torch.Tensor): input video tensor with shape
                              (batch_size, num_frames,
                              channels, height, width)

        Returns:
            torch.Tensor: classification logits with shape
                          (batch_size, num_classes)

        Notes:
            - Each frame is processed independently by the CNN backbone
            - Extracted frame features are passed through the LSTM
            for temporal modeling
            - Only the final LSTM hidden state is used for classification
        '''
        
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        seq = pooled.view(B, T, self.feature_dim)
        rnn_out, _ = self.rnn(seq)
        final = self.dropout(rnn_out[:, -1, :])
        return self.fc(final)
    
    
class PositionalEncoding(nn.Module):
    '''
    Sinusoidal positional encoding for temporal sequences

    This module adds positional information to sequence embeddings
    using fixed sinusoidal encoding as introduced in the Transformer
    architecture.

    Args:
        d_model (int)  : embedding dimension
        max_len (int)  : maximum supported sequence length,
                         default: 64
        dropout (float): dropout probability applied after adding
                         positional encoding, default: 0.1

    Attributes:
        dropout (nn.Dropout): dropout layer
        pe (torch.Tensor)   : precomputed positional encoding tensor
                              with shape (1, max_len, d_model)

    Notes:
        - Even indices use sine functions
        - Odd indices use cosine functions
        - Positional encodings are stored as a non-trainable buffer
    '''
    
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        '''
        Add positional encoding to input sequence embeddings

        Args:
            x (torch.Tensor): input tensor with shape
                              (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: input embeddings with positional encoding added
        '''
        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    '''
    Attention-based temporal pooling layer

    This module computes attention weights over temporal features
    and produces a weighted feature representation.

    Args:
        dim (int): feature dimension of input embeddings

    Attributes:
        attention (nn.Sequential): attention scoring network
                                   consisting of two linear layers
                                   and a Tanh activation
    '''
    
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        '''
        Apply attention pooling over temporal features

        Args:
            x (torch.Tensor): input tensor with shape
                              (batch_size, sequence_length, dim)

        Returns:
            torch.Tensor: pooled feature representation with shape
                          (batch_size, dim)

        Notes:
            - Attention weights are normalized across the temporal dimension
            - Frames with higher attention weights contribute more to the
            final pooled representation
        '''
        
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, dim)
        return pooled
    

class ConvNeXtTransformer(nn.Module):
    '''
    ConvNeXt-Tiny + Transformer model for video classification

    This architecture combines:
        - ConvNeXt-Tiny backbone for spatial feature extraction
        - Positional encoding for temporal order information
        - Transformer encoder for temporal modeling
        - Attention pooling for sequence aggregation

    Input:
        (B, T, C, H, W)
        Example: (B, 16, 3, 224, 224)

    Output:
        (B, num_classes)
        Example: (B, 100)

    Args:
        num_classes (int)                    : number of output classes,
                                               default: 100
        convnext_pretrained_weights (object): pretrained weights for
                                              ConvNeXt-Tiny backbone,
                                              default: None

    Attributes:
        cnn (nn.Module)                 : ConvNeXt-Tiny feature extractor
        pool (nn.Module)                : adaptive average pooling layer
        feature_dim (int)               : feature dimension extracted
                                          from ConvNeXt-Tiny
        pos_encoder (PositionalEncoding): positional encoding module
        transformer (nn.TransformerEncoder):
                                          temporal transformer encoder
        attention_pool (AttentionPooling):
                                          attention-based temporal pooling
        fc (nn.Sequential)              : classification head

    Notes:
        - Each frame is processed independently by the CNN backbone
        - Temporal relationships are modeled using a Transformer encoder
        - Attention pooling aggregates frame-level features into
          a single video representation
    '''
    
    def __init__(self, num_classes=100, convnext_pretrained_weights=None):
        super().__init__()

        # 1. ConvNeXt-Tiny Backbone
        convnext = models.convnext_tiny(weights=convnext_pretrained_weights)
        self.cnn = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ConvNeXt-Tiny output = 768
        self.feature_dim = 768

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.feature_dim,
            max_len=64,
            dropout=0.1
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Attention Pooling
        self.attention_pool = AttentionPooling(self.feature_dim)

        # 5. Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        '''
        Initialize weights for Transformer and attention pooling layers

        Notes:
            - Linear layers are initialized using truncated normal
            distribution with std=0.02
            - Bias terms are initialized to zero
        '''
        
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_convnext_features(self, freeze_until=3):
        '''
        Freeze ConvNeXt backbone layers up to a specified stage

        Args:
            freeze_until (int): index of the last ConvNeXt stage to freeze,
                                default: 3

        Returns:
            None

        Notes:
            - Parameters in frozen layers will have requires_grad=False
            - Useful for transfer learning and fine-tuning
        '''
        
        for i in range(freeze_until + 1):
            for p in self.cnn[i].parameters():
                p.requires_grad = False

    def forward(self, x):
        '''
        Forward pass of the ConvNeXt-Transformer model

        Args:
            x (torch.Tensor): input video tensor with shape
                              (batch_size, num_frames,
                              channels, height, width)

        Returns:
            torch.Tensor: classification logits with shape
                          (batch_size, num_classes)

        Notes:
            - CNN extracts spatial features for each frame
            - Positional encoding adds temporal order information
            - Transformer encoder models temporal dependencies
            - Attention pooling aggregates temporal features
        '''
    
        B, T, C, H, W = x.shape

        # CNN: (B, T, C, H, W) → (B, T, 768)
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(B, T, self.feature_dim)

        # Transformer: (B, T, 768) → (B, T, 768)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Pooling: (B, T, 768) → (B, 768)
        x = self.attention_pool(x)

        # Classifier: (B, 768) → (B, num_classes)
        x = self.fc(x)

        return x