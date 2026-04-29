from transformers import PretrainedConfig

# Huggingface的一个配置类，用于存储模型的超参数和配置选项。它继承自PretrainedConfig，并定义了MokioMind模型的特定配置选项。
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.nn import functional as F
from .activation_functions import ACT2FN

# 继承nn.Model类
class RMSNorm(nn.Module):
    # __init__初始化
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.dim = dim #纬度
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim)) # 初始化张量

    #__norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        
    # forward方法
    def forward(self, x):
        return self.weight * self._norm(x.float()).typed_as(x) * x
    
def precompute_freqs_cis(dim:int, end:int(32*1024), rope_base, rope_scaling:Optional[dict]=None):
    # 初始化RoPE频率
    freqs, attn_factor = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0)
    

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"], # 模型原本能承受的极限（“舒适区”）
            rope_scaling["factor"],  # 我们要把上下文扩展到原来的多少倍（128K / 32K = 4）
            rope_scaling["beta_fast"],  # 那些“波长”小于 beta_fast 的维度我们认为它是“高速转盘”，坚决不能动
            rope_scaling["beta_slow"],  # 多慢的齿轮我们认为它是“低速转盘”，可以减速
        )

        # 推断的长度大于训练长度，用缩放
        if end > orig_max:
            # 波长b到i的映射
            # “波长” = “转一整圈需要的 Token 数量”
            inv_dim = lambda b: (dim * math.log(orig_max / (b*2*math.pi)))/(2*math.log(rope_base))

            #划分高低维度
            #low： 不需要缩放的高频部分
            # high： 需要缩放的低频部分
            low, high = (max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2))

            # 计算缩放因子
            # low之前，ramp为0，high之后，ramp为1，中间线性过度
            ramp = torch.clamp(
                (torch.arange(dim//2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0,
                1,
            )
            # 当 ramp = 0 时（高频），系数为1，保持频率不变
            # 当 ramp = 1 时（低频），系数为 1/factor，频率进行线性插值缩放
            # ramp在0和1之间时，平滑过渡
            freqs = freqs * (1 - ramp + ramp * factor)

        # 根据end， 生成位置索引t
        t = torch.arange(end, device=freqs.device).float()

        
        # 计算外积， t和频率部分相乘， 得到每个位置的旋转角度
        freqs = torch.outer(t, freqs).float()
        freqs_cos = (
            torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        )
        freqs_sin = (
            torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        )
        return freqs_cos, freqs_sin
    
# 编写RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    # [a,b] -> [-b,a]
    def rotate_half(x):
        # x.shape[-1]取最后一个维度的重点
        # x[..., x.shape[-1] // 2 :]：取最后一半的维度
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    # x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

# 把 key/value 头复制多份
# num_key_value_heads  →  num_key_value_heads * n_rep
def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    """
    重复key-value张量以匹配query头数 (用于分组查询注意力GQA)
    等价于torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效
    
    在GQA中，key和value的头数少于query，需要重复来匹配
    例如：8个query头，2个kv头，则需要每个kv头重复4次
    
    Args:
        x: kv张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的张量 [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x # 无需重复直接返回
    
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头自注意力机制，支持分组查询注意力(GQA)和Flash Attention优化
    
    GQA介绍：
    - 传统MHA：query、key、value头数相同
    - GQA：key、value头数少于query头数，通过重复匹配
    - 优点：减少KV cache内存占用，保持性能
    """
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        # 处理GQA： 如没有指定kv头数，则使用与query相同的头数
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        # GQA 的核心思想是：将 num_attention_heads 个 Query 头分组，每一组共享一个 KV 头。
        # 这就意味着 num_attention_heads / num_key_value_heads 必须是一个整数，这个整数=n_rep（每个 KV 头需要复制的次数）
        assert args.num_attention_heads % self.num_key_value_heads == 0,
        "num_attention_heads must be divisible by num_key_value_heads"

        # 设置注意力头配置
        self.n_local_heads = args.num_attention_heads                 # query头
        self.n_local_kv_heads = self.num_key_value_heads              # kv头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads      # 每个kv头需要复制的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # 定义线性投影层 (无偏置，节省参数)
        # nn.Linear(in_features, out_features, bias=False)
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)     # Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Value投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)     # 输出投影
        
        # Dropout层用于正则化
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接dropout
        self.dropout = args.dropout                      # 保存dropout率
        
        # 检查是否支持Flash Attention
        # hasattr(obj, 'attr'): 检查对象是否有指定属性
        # Flash Attention需要PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 如果不支持可以打印警告: print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[Tuple[torch.Tensor, torch.Tensor]], #修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache = False,
                attention_mask: Optional[torch.Tensor] = None):
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            [batch, seq_len, hidden_size]
        position_embeddings : Tuple[Tuple[torch.Tensor, torch.Tensor]]
            预计算的RoPE位置编码的cos和sin
        use_cache : bool, optional
            是否缓存当前K/V
        attention_mask : Optional[torch.Tensor], optional
            用于屏蔽padding位置

        Returns
        -------
        _type_
            output, past_kv
        """
        # 输入x
        # x:[batch_size, seq_len, hidden]
        bsz, seq_len, _ = x.shape

        # ------------------ 线性投影 + 多头reshape------------------
        # 线性投影 Q，K，V
        # q_proj: hidden -> num_heads * head_dim
        # k_proj/v_proj: hidden -> num_kv_heads * head_dim(GQA情形)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 将投影结果reshape成多头格式
        # q：[bsz, seq_len, num_heads, head_dim]
        # k/v: [bsz, seq_len, n_local_kv_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # -------------------- RoPE 处理 --------------------
        # position_embeddings是预计算的（cos，sin），按序列位置切片并应用RoPE
        # 比如 [max_seq_len=2048, 64]
        cos, sin = position_embeddings
        # 只取当前序列长度的前缀
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # -------------------- KV cache 处理 --------------------
        # past_key_value: (past_k, past_v) 或 None
        # 当存在past时，将past拼接到当前k,v的时间维度上，便于自回归推理
        if past_key_value is not None:
            # past_key_value[0] 的shape为 [bsz, past_seq_len, n_local_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 如果需要缓存，返回拼接后的(k,v)，否则past_kv置为None
        # 有缓存：只算新token的K/V → O(n)
        # 无缓存：算所有token的K/V → O(n^2)
        past_kv = (xk, xv) if use_cache else None

        # -------------------- GQA: 对KV重复以匹配Q头 --------------------
        # 转置: [batch, seq, heads, dim] -> [batch, heads, seq, dim] 以便矩阵乘法
        xq = xq.transpose(1, 2)

        # repeat_kv会把k/v的头数从 n_local_kv_heads -> n_local_kv_heads * n_rep (即等于n_local_heads)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2) # [2, 2, 4, 64] -> [2, 8, 4, 64]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2) 
        
        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现 内存复杂度 O(N)
            # is_causal=True: 自动创建因果mask（上三角为-inf）
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # 自回归（因果）注意力
            )
        else:
            # 标准实现：scores = Q @ K^T / sqrt(d)
            # Step 1: 计算注意力分数
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)

            # Step 2: 因果mask（不让当前位置看到未来）
            # causal mask: 上三角（对角线以上）置为 -inf
            causal_mask = torch.tril(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0) # 扩展batch和head维度

            # Step 3: Padding mask
            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # 在最后一维做softmax，即每一行
            scores = self.attn_dropout(scores)
            # 加权求和得到输出
            output = scores @ xv

        # 恢复形状并做输出投影 + 残差dropout
        # # [batch, heads, seq_len, head_dim] ->[batch, seq_len, heads, head_dim] -> [batch, seq_len, hidden]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # -1 表示自动计算: heads * head_dim = 8 * 64 = 512
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    

class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8/3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64) # 64的整数倍

        # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        # up_proj: hidden -> intermediate (用于被gate的部分)
        # down_proj: intermediate -> hidden (用于投影回hidden维度)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # ACT2FN是transformers里激活函数的映射表，支持'silu','gelu'等
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, x):
        """
        forward实现使用SwiGLU风格的门控激活：
        output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        并在输出前应用dropout
        """
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))