import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import os

from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from .monkey_patch_module import Qwen2_5_VLVisionBlock_Pano, Qwen2_5_VisionTransformerPretrainedModel_Pano
except ImportError:
    from monkey_patch_module import Qwen2_5_VLVisionBlock_Pano, Qwen2_5_VisionTransformerPretrainedModel_Pano

from qwen_vl_utils import process_vision_info


class InteractionIndexerWithLearnedRPE(nn.Module):
    """
    Implements an interactive gated (GLU-style) indexer.

    This version uses *Learnable Relative Position Encoding (RPE)*.
    
    1. Data stream: ReLU( (q_data * k_data) + RPE_Bias )
    2. Gate stream: MLP( (q_gate * k_gate) + RPE_Bias )
    """
    def __init__(self, dim: int, index_heads: int = 4, index_dim: int = 32, 
                 gate_mlp_expansion: int = 2,
                 max_relative_position: int = 128):
        """
        Initialize the indexer.
        
        Args:
            dim (int): Backbone network dimension.
            index_heads (int): Number of indexer heads (H^I).
            index_dim (int): Indexer Q/K dimension (d_I).
            gate_mlp_expansion (int): Hidden layer expansion ratio for gate MLP.
            max_relative_position (int): 
                Maximum learnable relative distance.
                This creates an embedding table of size (2*max+1).
                For large images, you may need a larger value (like 64 or 128) 
                to capture longer dependencies.
        """
        super().__init__()
        self.num_heads = index_heads
        self.head_dim = index_dim
        self.max_relative_position = max_relative_position
        
        # 1. QK Projections
        self.q_proj = nn.Linear(dim, index_heads * index_dim * 2, bias=False)
        self.k_proj = nn.Linear(dim, index_heads * index_dim * 2, bias=False)
        
        # 2. Dynamic gating MLP
        hidden_dim = index_heads * gate_mlp_expansion
        self.gate_mlp = nn.Sequential(
            nn.Linear(index_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, index_heads),
            nn.Sigmoid()
        )
        
        # 3. Learnable RPE Table
        # We need (2 * max_relative_position + 1) buckets:
        # - 0 to max-1:           Represents -max to -1
        # - max:                  Represents 0
        # - max+1 to 2*max:       Represents +1 to +max
        self.num_buckets = 2 * self.max_relative_position + 1
        
        # Learnable parameters: [num_buckets, num_heads]
        # Each bucket provides a learnable bias value for H heads
        self.relative_position_bias_table = nn.Embedding(
            self.num_buckets, self.num_heads
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize RPE table to 0
        nn.init.zeros_(self.relative_position_bias_table.weight)

    def _compute_relative_buckets(self, query_pos: torch.Tensor, 
                                  key_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positions and map them to learnable bucket IDs.
        """
        # 1. Compute relative distance [N_q, N_k]
        # (e.g., t=5, s=3 -> 2; t=3, s=5 -> -2)
        relative_distance = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)
        
        # 2. Clip to [-max, +max] range
        clipped_distance = torch.clamp(
            relative_distance, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 3. Convert to embedding table indices (0 to 2*max)
        # (e.g., -max -> 0; 0 -> max; +max -> 2*max)
        bucket_ids = clipped_distance + self.max_relative_position
        
        # Return [N_q, N_k] integer bucket IDs
        return bucket_ids.long()

    def forward(self, 
                query_states: torch.Tensor, 
                key_states: torch.Tensor,
                query_pos: torch.Tensor,
                key_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_states (torch.Tensor): [N_q, dim]
            key_states (torch.Tensor): [N_k, dim]
            query_pos (torch.Tensor): 1D position indices for Q, [N_q]
            key_pos (torch.Tensor): 1D position indices for K, [N_k]
        """
        N_q, _ = query_states.shape
        N_k, _ = key_states.shape
        H, D_i = self.num_heads, self.head_dim

        # 1. Project Q, K
        q_I = self.q_proj(query_states).reshape(N_q, H * 2, D_i)
        k_I = self.k_proj(key_states).reshape(N_k, H * 2, D_i)

        # 2. Compute (q_t * k_s)
        q_I = q_I.permute(1, 0, 2)
        k_I = k_I.permute(1, 2, 0)
        dot_prod = torch.bmm(q_I, k_I).permute(1, 2, 0) # [N_q, N_k, H*2]

        # 3. Separate data flow and gate flow
        dot_prod_data, dot_prod_gate = dot_prod.chunk(2, dim=-1) # [N_q, N_k, H]
        
        # 4. Compute learnable RPE biases
        # 4.1. Compute bucket ID: [N_q, N_k]
        bucket_ids = self._compute_relative_buckets(query_pos, key_pos)
        
        # 4.2. Lookup learnable biases from table: [N_q, N_k, H]
        rpe_bias = self.relative_position_bias_table(bucket_ids)

        # 5. Apply data stream and gate stream 
        
        # 5.1. Data Stream
        # Input = (q_data * k_data) + RPE_Bias
        data_stream_input = dot_prod_data + rpe_bias
        relu_scores = F.relu(data_stream_input)
        
        # 5.2. Gate Stream
        # Input = (q_gate * k_gate) + RPE_Bias
        gate_stream_input = dot_prod_gate + rpe_bias
        
        # Feed fused inputs (interactive + positional) into MLP
        gate_scores = self.gate_mlp(gate_stream_input) # [N_q, N_k, H]

        # 6. Final weighting
        weighted_scores = relu_scores * gate_scores 

        # 7. Summation
        index_scores = weighted_scores.sum(dim=-1)
        
        return index_scores


class PanoramaSparseAttention(nn.Module):
    """
    Lightweight sparse attention, utilizing a panorama-aware Interaction Indexer.
    """
    def __init__(self, dim: int, bottle_dim: int = 128, num_heads: int = 196, sparse_k: int = 523,
                 index_heads: int = 4, index_dim: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparse_k = sparse_k
        
        # Bottleneck QKV
        self.qkv_down_proj = nn.Linear(dim, bottle_dim)
        self.qkv_up_proj = nn.Linear(bottle_dim, dim * 3) 

        # Bottleneck Proj
        self.proj_down_proj = nn.Linear(dim, bottle_dim)
        self.proj_up_proj = nn.Linear(bottle_dim, dim)

        self.indexer = InteractionIndexerWithLearnedRPE(
            dim,
            index_heads=index_heads,
            index_dim=index_dim,
            max_relative_position=64
        )
        
        self.scaler = nn.Parameter(torch.ones(1), requires_grad=True)
        
        # Visualization flag
        self.enable_vis = False
        self.vis_data = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_coords: torch.Tensor,         # Required: Patch coordinates for positions
        cu_seqlens: torch.Tensor,
        cu_seqlens_whole: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        if cu_seqlens_whole is None:
            cu_seqlens_whole = cu_seqlens
        seq_length = hidden_states.shape[0]
        
        if self.enable_vis:
            self.vis_data = []

        # Step 1: Generate Q, K, V through bottleneck
        bottleneck_qkv = self.qkv_down_proj(hidden_states)
        qkv_states = self.qkv_up_proj(bottleneck_qkv)
        q, k, v = qkv_states.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        # Application of RoPE
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings 
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Step 2: Build dynamically composed attention mask
        final_attention_mask = torch.full(
            [seq_length, seq_length], torch.finfo(q.dtype).min, 
            device=q.device, dtype=q.dtype
        )
        
        for i in range(1, len(cu_seqlens_whole)): # Iterate through each image
            s_start, s_end = cu_seqlens_whole[i - 1], cu_seqlens_whole[i]
            num_tokens_img = s_end - s_start
            
            if num_tokens_img <= 0: continue
            
            # Extract hidden_states and patch_coords for the current image
            img_states = hidden_states[s_start:s_end]
            img_coords = patch_coords[s_start:s_end] # [N_i, 2] formatted as (y, x)
            img_pos_1d = img_coords[:, 1].float()
            
            # Compute dynamic index scores
            local_index_scores = self.indexer(
                img_states, 
                img_states, 
                query_pos=img_pos_1d,
                key_pos=img_pos_1d,
            )

            # Find Top-K Keys for each Query
            current_k = min(self.sparse_k, num_tokens_img)
            
            top_k_scores, top_k_indices = torch.topk(
                local_index_scores, k=current_k, dim=-1
            )
            
            if self.enable_vis:
                self.vis_data.append({
                    'img_idx': i,
                    'top_k_indices': top_k_indices.detach().cpu(),
                    'local_index_scores': local_index_scores.detach().cpu(),
                    'img_coords': img_coords.detach().cpu()
                })
            
            # Construct local sparse mask 
            local_mask = torch.full_like(local_index_scores, torch.finfo(q.dtype).min)
            local_mask.scatter_(dim=-1, index=top_k_indices, value=0.0)
            
            # Apply local mask back to the global final mask
            final_attention_mask[s_start:s_end, s_start:s_end] = local_mask

        # Broadcast mask configuration to (Batch, Heads, S, S)
        final_attention_mask = final_attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Step 3: Compute localized Flash Attention
        q = q.transpose(0, 1).unsqueeze(0) 
        k = k.transpose(0, 1).unsqueeze(0) 
        v = v.transpose(0, 1).unsqueeze(0) 

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=final_attention_mask, 
            is_causal=False
        ).squeeze(0) 
        
        # Step 4: Linearly project outputs through bottleneck
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        bottleneck_proj = self.proj_down_proj(attn_output)
        final_output = self.proj_up_proj(bottleneck_proj)

        return final_output * self.scaler


class PanoramaSparseAttentionFast(nn.Module):
    """
    Optimized version of PanoramaSparseAttention.
    Uses chunked processing for index score calculation to save memory and
    uses Flash Attention 2 natively (via PyTorch's SDPA with memory-efficient backend).
    """
    def __init__(self, dim: int, bottle_dim: int = 128, num_heads: int = 196, sparse_k: int = 523,
                 index_heads: int = 4, index_dim: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparse_k = sparse_k
        
        # Bottleneck QKV
        self.qkv_down_proj = nn.Linear(dim, bottle_dim)
        self.qkv_up_proj = nn.Linear(bottle_dim, dim * 3) 

        # Bottleneck Proj
        self.proj_down_proj = nn.Linear(dim, bottle_dim)
        self.proj_up_proj = nn.Linear(bottle_dim, dim)

        self.indexer = InteractionIndexerWithLearnedRPE(
            dim,
            index_heads=index_heads,
            index_dim=index_dim,
            max_relative_position=64
        )
        
        self.scaler = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_coords: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_whole: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        if cu_seqlens_whole is None:
            cu_seqlens_whole = cu_seqlens
        seq_length = hidden_states.shape[0]

        # Step 1: Generate Q, K, V
        bottleneck_qkv = self.qkv_down_proj(hidden_states)
        qkv_states = self.qkv_up_proj(bottleneck_qkv)
        q, k, v = qkv_states.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        # Application of RoPE
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings 
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Reshape for SDPA (Flash Attention): [1, batch*heads, S, head_dim]
        # Instead of 1 global mask with size [seq_length, seq_length] which causes OOM,
        # we will chunk the attention computation per image.
        
        final_output_chunks = []

        # Step 2 & 3: Iterate through each image to compute fast attention
        for i in range(1, len(cu_seqlens_whole)):
            s_start, s_end = cu_seqlens_whole[i - 1], cu_seqlens_whole[i]
            num_tokens_img = s_end - s_start
            
            if num_tokens_img <= 0: continue
            
            # --- Faster Indexing ---
            with torch.no_grad(): # Use no_grad for top-K calculations if gradients aren't needed for masking
                img_states_detached = hidden_states[s_start:s_end].detach()
                img_pos_1d = patch_coords[s_start:s_end, 1].float()
                
                # Chunk index scores calculation to prevent N^2 memory blowup on very large images
                chunk_size = 2048 
                top_k_indices_list = []
                
                for chunk_start in range(0, num_tokens_img, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_tokens_img)
                    chunk_query = img_states_detached[chunk_start:chunk_end]
                    chunk_pos = img_pos_1d[chunk_start:chunk_end]
                    
                    local_index_scores_chunk = self.indexer(
                        chunk_query, 
                        img_states_detached, 
                        query_pos=chunk_pos,
                        key_pos=img_pos_1d,
                    )
                    
                    current_k = min(self.sparse_k, num_tokens_img)
                    # Use fast topk on GPU
                    _, chunk_top_k_idx = torch.topk(local_index_scores_chunk, k=current_k, dim=-1)
                    top_k_indices_list.append(chunk_top_k_idx)
                
                top_k_indices = torch.cat(top_k_indices_list, dim=0)
            
            # Construct localized boolean mask directly ([N_i, N_i])
            local_mask = torch.zeros((num_tokens_img, num_tokens_img), dtype=torch.bool, device=q.device)
            local_mask.scatter_(dim=-1, index=top_k_indices, value=True)
            
            # Add head dimension: [1, 1, N_i, N_i]
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)

            # --- Flash Attention Computation for this image ---
            # q_img: [N_i, heads, dim] -> [1, heads, N_i, dim]
            q_img = q[s_start:s_end].transpose(0, 1).unsqueeze(0).contiguous()
            k_img = k[s_start:s_end].transpose(0, 1).unsqueeze(0).contiguous()
            v_img = v[s_start:s_end].transpose(0, 1).unsqueeze(0).contiguous()

            # Utilize PyTorch's SDPA with memory-efficient config
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_out_img = torch.nn.functional.scaled_dot_product_attention(
                    q_img, k_img, v_img,
                    attn_mask=local_mask,  # Using boolean mask is much faster and memory efficient in SDPA
                    is_causal=False
                )
            
            # Revert shape: [1, heads, N_i, dim] -> [N_i, heads * dim]
            attn_out_img = attn_out_img.squeeze(0).transpose(0, 1).reshape(num_tokens_img, -1)
            final_output_chunks.append(attn_out_img)

        # Concatenate outputs back
        attn_output = torch.cat(final_output_chunks, dim=0)
        
        # Step 4: Linearly project outputs through bottleneck
        bottleneck_proj = self.proj_down_proj(attn_output)
        final_output = self.proj_up_proj(bottleneck_proj)

        return final_output * self.scaler
    

def modify_qwen2_5_vl_vision_attention(type_: str, config, adaptor_config: Optional[dict] = None):
    """
    Modifies the Qwen2.5 VL Vision Block to inject a parallel Attention module.
    
    Args:
        type_ (str): The type of Attention module to inject.
        config: The model's configuration object.
        adaptor_config: Configuration dictionary for the adapter (the new Attention module).
    """
    
    module_factory = None

    if type_ == "panorama_sparse_attention":
        print("Injecting Panorama Sparse Attention.")
        print("adaptor_config:", adaptor_config)

        def create_module(init_config):
            try:
                if not adaptor_config:
                    raise ValueError("adaptor_config is None, using defaults.")
                
                # Switch to using the Fast version of the sparse attention
                return PanoramaSparseAttentionFast(
                    dim=init_config.hidden_size,
                    bottle_dim=adaptor_config.get("bottle_dim", 128), 
                    num_heads=init_config.num_heads,
                    sparse_k=adaptor_config.get("sparse_k", 512),
                    index_heads=adaptor_config.get("index_heads", 4),
                    index_dim=adaptor_config.get("index_dim", 16),
                )
            except Exception as e:
                print(f"Using default parameters for PanoramaSparseAttentionFast (error: {e})")
                return PanoramaSparseAttentionFast(
                    dim=init_config.hidden_size,
                    bottle_dim=512, 
                    num_heads=init_config.num_heads,
                    sparse_k=128,
                    index_heads=4,
                    index_dim=32,
                )
        
        module_factory = create_module

    else:
        print("No supported modification identified for Qwen2.5-VL vision block's attention.")
        return 

    if module_factory is None:
        print(f"No attention module factory created for type '{type_}'. No patch applied.")
        return

    # Dynamic creation to apply patches securely
    def new_init(self, config, attn_implementation: str = "flash_attention_2"):
        # Instantiate a detached module exclusively per block initialized
        attention_module_to_inject = module_factory(config)
        
        # Enforce consistent precision dtypes
        model_dtype = config.torch_dtype if hasattr(config, 'torch_dtype') else torch.float32

        # Initialize the base class with customized arguments
        Qwen2_5_VLVisionBlock_Pano.__init__(
            self, 
            config, 
            atten_adapter=attention_module_to_inject.to(model_dtype), 
            FFN_adapter=None, 
            attn_implementation=attn_implementation
        )

    # Instantiate our modified overarching class
    PatchedBlock = type(
        "PatchedQwen2_5_VLVisionBlockWithAttention", 
        (Qwen2_5_VLVisionBlock_Pano,), 
        {"__init__": new_init}
    )
    
    # Execute class monkey-patching overrides 
    print(f"Monkey-patching Qwen2_5_VLVisionBlock with {type_}.")
    modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock = PatchedBlock
    
    # Enable Custom Vision Transformer patching logic globally 
    print("Monkey-patching Qwen2_5_VisionTransformerPretrainedModel to use Pano forward.")
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel = Qwen2_5_VisionTransformerPretrainedModel_Pano


def run_tests():
    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": "./datasets/nuscenes-360/train/1532402927647951.jpg", "resized_height": 4800, "resized_width": 300}, 
                {"type": "text", "text": "Describe this image."}
            ]
        },
        {
            "role": "user", "content": [
                {"type": "image", "image": "./datasets/nuscenes-360/train/1532402927647951.jpg", "resized_height": 4800, "resized_width": 300}, 
                {"type": "text", "text": "tell me what difference between two images."}
            ]
        }
    ]
    
    qwen_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Process visual content logic 
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    config = AutoConfig.from_pretrained(qwen_model_path)

    adaptor_config = {
        "name": "panorama_sparse_attention",
        "bottle_dim": 512, 
        "sparse_k": 128
    }
    
    # Implement configurations and attention module variations 
    modify_qwen2_5_vl_vision_attention('panorama_sparse_attention', config.vision_config, adaptor_config)
    
    # Setup full conditionally generated model  
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        attn_implementation="flash_attention_2",
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )

    visual = model.visual
    processor = AutoProcessor.from_pretrained(qwen_model_path)
    
    # Process text representations into embeddings appropriately
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(model.device)
    
    pixel_values = inputs['pixel_values'].type(model.visual.dtype)

    print("Inputs keys:", inputs.keys())
    print("Pixel Values Shape:", inputs['pixel_values'].shape)
    
    # Process visual representations explicitly
    image_embeds = visual(pixel_values, grid_thw=inputs['image_grid_thw'])
    print("Image Embeddings Shape:", image_embeds.shape)

    # Generate test responses 
    output = model.generate(**inputs, max_new_tokens=2048)
    print("Output Dimensions:", output.shape)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Inferred Text Iterations Result:\n", output_text)


if __name__ == "__main__":
    # Remove the comment and correctly test environment initialization setups:
    # run_tests()
    pass
