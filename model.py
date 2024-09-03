import torch
import torch.nn as nn
import open_clip

class PyramidForwarding(nn.Module):
    def __init__(self, base_model, patch_size=224, pyramid_levels=3, allow_overlap=True, transform=None):
        super(PyramidForwarding, self).__init__()
        self.base_model = base_model  # Pre-trained CLIP visual tower (ViT-L/14)
        self.patch_size = patch_size
        self.pyramid_levels = pyramid_levels
        self.allow_overlap = allow_overlap
        self.transform = transform

    def forward(self, images):
        B, C, H, W = images.shape
        #print(f"[PyramidForwarding] Initial images shape: {images.shape}")

        log_d_h = int(torch.log2(torch.tensor(H // self.patch_size)).item())
        log_d_w = int(torch.log2(torch.tensor(W // self.patch_size)).item())
        pyramid_levels = min(self.pyramid_levels, log_d_h + 1, log_d_w + 1)
        #print(f"[PyramidForwarding] Calculated pyramid levels: {pyramid_levels}")

        patches_list = []

        for i in range(pyramid_levels):
            scale_factor_h = 2 ** i
            scale_factor_w = 2 ** i
            resized_size = (self.patch_size * scale_factor_h, self.patch_size * scale_factor_w)
            #print(f"[PyramidForwarding] Level {i}: Resizing images to {resized_size}")

            resized_images = nn.functional.interpolate(images, size=resized_size)
            #print(f"[PyramidForwarding] Resized images shape at level {i}: {resized_images.shape}")

            stride_h = stride_w = self.patch_size

            if self.allow_overlap:
                stride_h_div = resized_images.size(2) // self.patch_size - 1
                stride_w_div = resized_images.size(3) // self.patch_size - 1

                if stride_h_div > 0:
                    stride_h = (resized_images.size(2) - self.patch_size) // stride_h_div
                if stride_w_div > 0:
                    stride_w = (resized_images.size(3) - self.patch_size) // stride_w_div

                #print(f"[PyramidForwarding] Level {i}: Stride calculated as (h: {stride_h}, w: {stride_w})")

            patches = resized_images.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
            patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4)
            #print(f"[PyramidForwarding] Level {i}: Patches shape before final view: {patches.shape}")

            patches = patches.reshape(-1, C, self.patch_size, self.patch_size)  # Use reshape instead of view
            #print(f"[PyramidForwarding] Level {i}: Final patches shape: {patches.shape}")

            if self.transform:
                patches = torch.stack([self.transform(patch) for patch in patches])
                #print(f"[PyramidForwarding] Level {i}: Transformed patches shape: {patches.shape}")

            patches_list.append(patches)

        total_patches = sum(patch.size(0) for patch in patches_list)
        #print(f"[PyramidForwarding] Total patches generated: {total_patches}")

        return patches_list

class DMDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(DMDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.multihead_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q_lbl, k_img, v_img, mask=None):
        #print(f"[DMDecoderLayer] Initial q_lbl shape: {q_lbl.shape}")
        #print(f"[DMDecoderLayer] Initial k_img shape: {k_img.shape}")
        #print(f"[DMDecoderLayer] Initial v_img shape: {v_img.shape}")

        q1_mid = self.norm1(q_lbl + self.dropout(q_lbl))
        #print(f"[DMDecoderLayer] q1_mid shape after norm1: {q1_mid.shape}")

        q2_mid, _ = self.multihead_attn1(q1_mid, k_img, v_img, key_padding_mask=mask)
        #print(f"[DMDecoderLayer] q2_mid shape after multihead_attn1: {q2_mid.shape}")

        q3_mid = self.norm2(q2_mid + q1_mid)
        #print(f"[DMDecoderLayer] q3_mid shape after norm2: {q3_mid.shape}")

        q4_mid = self.ffn1(q3_mid)
        q4_mid = self.dropout(q4_mid)
        #print(f"[DMDecoderLayer] q4_mid shape after ffn1 and dropout: {q4_mid.shape}")

        q5_mid = self.norm3(q4_mid + q3_mid)
        #print(f"[DMDecoderLayer] q5_mid shape after norm3: {q5_mid.shape}")

        v1_img, _ = self.multihead_attn2(v_img, q5_mid, q5_mid, key_padding_mask=mask)
        #print(f"[DMDecoderLayer] v1_img shape after multihead_attn2: {v1_img.shape}")

        v_img = self.norm4(v1_img + v_img)
        #print(f"[DMDecoderLayer] v_img shape after norm4: {v_img.shape}")

        return q5_mid, v_img

class DMDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=32, ff_dim=3072, num_layers=6):
        super(DMDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DMDecoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, embed_dim)  # Final output transformation layer

    def forward(self, visual_embeddings, textual_embeddings, mask=None):
        #print(f"[DMDecoder] Initial visual_embeddings shape: {visual_embeddings.shape}")
        #print(f"[DMDecoder] Initial textual_embeddings shape: {textual_embeddings.shape}")

        q_lbl = textual_embeddings
        k_img = visual_embeddings
        v_img = visual_embeddings

        for i, layer in enumerate(self.layers):
            q_lbl, v_img = layer(q_lbl, k_img, v_img, mask=mask)
            k_img = v_img  # Update key for the next layer
            #print(f"[DMDecoder] After layer {i}, q_lbl shape: {q_lbl.shape}")
            #print(f"[DMDecoder] After layer {i}, v_img shape: {v_img.shape}")

        # Final processing of the output after the last decoder layer
        q_lbl = self.fc(q_lbl)  # Pass through the final fully connected layer
        #print(f"[DMDecoder] Final q_lbl shape after fc: {q_lbl.shape}")
        return q_lbl

class ADDSModel(nn.Module):
    def __init__(
        self,
        clip_model,
        patch_size=224,
        pyramid_levels=1,
        allow_overlap=True,
        embed_dim=768,
        num_heads=32,
        num_layers=6,
        num_labels=414
    ):
        super(ADDSModel, self).__init__()
        self.base_model = clip_model  # Use the passed-in CLIP model (ViT-L/14)
        
        # Freeze the CLIP model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.pyramid_forwarding = PyramidForwarding(
            clip_model.visual, patch_size, pyramid_levels, allow_overlap
        )
        self.decoder = DMDecoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.final_fc = nn.Linear(embed_dim, 1)  # Map to a single probability per label
        self.dropout = nn.Dropout(p=0.1)
        self.num_labels = num_labels

    def forward(self, images, text_embeddings, patch_indices=None):
        #print(f"[ADDSModel] Initial images shape: {images.shape}")  # [3, 3, 224, 224]

        # Apply Pyramid Forwarding to the images
        patches_list = self.pyramid_forwarding(images)
        #print(f"[ADDSModel] Number of patches generated: {len(patches_list)}")
        #for i, patches in enumerate(patches_list):
            #print(f"[ADDSModel] Patches shape at level {i}: {patches.shape}")

        # Encode visual embeddings
        visual_embeddings = torch.cat(
            [self.base_model.encode_image(patch) for patch in patches_list], dim=0
        )  # [3, 768]
        #print(f"[ADDSModel] Concatenated visual_embeddings shape: {visual_embeddings.shape}")  # [3, 768]

        batch_size = images.size(0)  # 3
        num_labels = self.num_labels  # 414

        # Repeat visual_embeddings for each label
        visual_embeddings = visual_embeddings.unsqueeze(1).repeat(1, num_labels, 1)  # [3, 414, 768]
        visual_embeddings = visual_embeddings.view(-1, visual_embeddings.size(-1))    # [1242, 768]
        #print(f"[ADDSModel] Reshaped visual_embeddings shape: {visual_embeddings.shape}")  # [1242, 768]

        # Repeat text_embeddings for each image
        text_embeddings = text_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)      # [3, 414, 768]
        text_embeddings = text_embeddings.view(-1, text_embeddings.size(-1))          # [1242, 768]
        #print(f"[ADDSModel] Reshaped text_embeddings shape: {text_embeddings.shape}")    # [1242, 768]

        # Pass through the DM-Decoder
        decoder_output = self.decoder(visual_embeddings, text_embeddings)             # [1242, 768]
        #print(f"[ADDSModel] Decoder output shape: {decoder_output.shape}")             # [1242, 768]

        # Final layer to get the probability for each label
        output = self.final_fc(decoder_output)                                       # [1242, 1]
        output = self.dropout(output)                                                # [1242, 1]
        #print(f"[ADDSModel] Output shape after final_fc and dropout: {output.shape}")  # [1242, 1]

        # Reshape to [batch_size, num_labels]
        output = output.view(batch_size, num_labels)                                 # [3, 414]
        #print(f"[ADDSModel] Final output shape after reshaping: {output.shape}")      # [3, 414]

        return output

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.01, eps=1e-8, reduction='sum'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        targets = targets.float()
        
        if self.clip is not None and self.clip > 0:
            clip_value = torch.tensor(self.clip).to(logits.device)
            probas = torch.where(probas > clip_value, probas, torch.tensor(self.eps).to(logits.device))

        pos_loss = -targets * ((1 - probas) ** self.gamma_pos) * torch.log(probas + self.eps)
        neg_loss = -(1 - targets) * (probas ** self.gamma_neg) * torch.log(1 - probas + self.eps)

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

