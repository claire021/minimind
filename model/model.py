from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import init

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
from typing import Optional, Tuple, Union, List
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, GenerationMixin


class RMSNorm(torch.nn.Module):
    """
    RMS归一化 (Root Mean Square Normalization)
    相比LayerNorm，RMSNorm去掉了均值中心化，只保留方差缩放
    计算更简单，效果相当，在大模型中广泛使用
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: 归一化的维度大小
            eps: 防止除零的小常数
        """
        super().__init__()                              # 调用父类nn.Module的构造函数
        self.eps = eps                                  # 存储epsilon值
        # nn.Parameter: 将tensor注册为可学习参数，会自动加入optimizer
        # torch.ones(dim): 创建全1的tensor作为缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        RMSNorm的核心计算：x / sqrt(mean(x^2) + eps)
        """
        # x.pow(2): 对x每个元素平方
        # .mean(-1, keepdim=True): 在最后一维求均值，保持维度
        # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，shape为[batch, seq_len, dim]
        Returns:
            归一化后的tensor
        """
        # .float(): 转换为float32进行计算，提高数值稳定性
        # .type_as(x): 将结果转换回x的原始数据类型
        # self.weight *: 可学习的缩放参数
        return self.weight * self._norm(x.float()).type_as(x)
    
def precompute_freqs(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
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
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is not None else args.num_key_value_heads

        # GQA 的核心思想是：将 num_attention_heads 个 Query 头分组，每一组共享一个 KV 头。
        # 这就意味着 num_attention_heads / num_key_value_heads 必须是一个整数，这个整数=n_rep（每个 KV 头需要复制的次数）
        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_attention_heads must be divisible by num_key_value_heads"

        # 设置注意力头配置
        self.n_local_heads = args.num_attention_heads          # query头数
        self.n_local_kv_heads = self.num_key_value_heads       # key-value头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv头需要重复的次数
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
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
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
        升维→非线性→降维
        forward实现使用SwiGLU风格的门控激活：
        output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        并在输出前应用dropout
        """
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
    
class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    
class MoEFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache
    

# 按照架构图将前面的内容拼接
class MokioMindBlock(nn.Module):
    """
    输入 hidden_states
    |
    ├─────────────────┐ (残差路径1)
    |                 |
    |   input_layernorm (RMSNorm)
    |         |
    |    self_attn (多头注意力)
    |         |
    └────────→ (+) 相加
              |
              ├─────────────────┐ (残差路径2)
              |                 |
              | post_attention_layernorm
              |         |
              |      mlp (FFN)
              |         |
              └────────→ (+) 相加
                        |
                    输出
    """
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config) # 创建注意力层

        self.layer_id = layer_id # 记录这是第几层（用于调试和位置编码）
        # 两个RMSNorm层（Pre-Norm模式
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        # 前馈网络（支持MoE或普通FFN）
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  
        )


    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """_summary_

        Parameters
        ----------
        hidden_states : _type_
            输入: [batch, seq_len, hidden_size]
        position_embeddings : _type_
            RoPE位置编码 (cos, sin)
        past_key_value : _type_, optional
            KV缓存 (推理时用)
        use_cache : bool, optional
            是否缓存
        attention_mask : _type_, optional
            注意力掩码 (padding等)

        Returns
        -------
        _type_
            hidden_states, present_key_value
        """
        # 残差连接模式：先做LayerNorm -> Attention -> 残差相加 -> LayerNorm -> FFN -> 残差相加
        # Step 1: 保存原始输入（残差路径）
        residual = hidden_states

        # 注意力子层：输入先归一化（RMSNorm），返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            # Step 2: 处理输入（变换路径）
            # 在注意力之前归一化pre-norm
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        # Step 3: 残差相加（原始 + 变换）
        # 注意力输出与残差相加
        hidden_states = residual + hidden_states

        # Step 4: 重复上述过程
        # 前馈子层（post-attention layernorm）并相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # 在FFN之前归一化
        return hidden_states, present_key_value
    

class MiniMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MokioMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(dim = config.hidden_size // config.num_attention_heads,
                                                    end = config.max_position_embeddings, rope_base = config.rope_theta,
                                                    rope_scaling = config.rope_scaling) 
        self.register_buffer("freqs_cos", freqs_cos, persistent = False)
        self.register_buffer("freqs_sin", freqs_sin, persistent = False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        # input_ids: [bsz, seq_len]
        batch_size,seq_length = input_ids.shape

        # 兼容性检查：某些框架会传入包含.layers属性的对象，视为不携带past信息
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        
        # past_key_values为每层的(past_k, past_v)列表，如果为None则创建与层数相同的None列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        # past_key_values[0] 形如 (k, v)，k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding + dropout
        # 上一层的输出（或embedding）
        hidden_states = self.dropout(self.embed_tokens(input_ids)) # [bsz, seq_len, hidden]

        # 从注册的buffer中取出对应位置范围的cos/sin作为position_embeddings
        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向，通过zip把layer和对应的past_key_value配对
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer( # MiniMindBlock 实例（Transformer层）
                hidden_states,
                position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            presents.append(present)

        # 最后归一化
        hidden_states = self.norm(hidden_states)

        # 如果使用MoE，收集每层的aux_loss并求和返回以便训练使用
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MoEFeedForward)
        )
        return hidden_states, presents, aux_loss


class MokioMindForCausalLM(PreTrainedModel,
                           GenerationMixin):
    '''
    PreTrainedModel: HuggingFace 基类，提供保存/加载、配置管理等功能
    GenerationMixin: HuggingFace 生成混入类，提供 generate() 方法（文本生成）
    '''

    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig): 
        super().__init__(config)
        self.model = MiniMindModel(config)
        # 输出形状：[batch_size, seq_len, hidden_size]）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        # 权重共享（Weight Tying）——让 embedding 层和 lm_head 使用相同的权重矩阵。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None, # [batch, seq_len]
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None, # [batch, seq_len]: 用于计算损失的真实 token ID
            past_key_value: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # (K, V)
            use_cache: bool = None,
            #只保留最后几个位置的 logits（节省内存）
            logits_to_keep: Union[int, torch.Tensor] = 0, # int/Tensor
            **args # 接收额外参数，传给 self.model
            ):
        
            # 调用主干模型向前传播
            hidden_states, past_key_value, aux_loss = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                past_key_values = past_key_value,
                use_cache = use_cache,
                **args,
            )

            # 切片logits（内存优化）
            # 决定保留哪些位置的 logits
            # 计算损失时，只需要最后一个位置预测下一个 token
            slice_indices = (
                # slice(start, stop) 创建切片对象
                # 如果 logits_to_keep=1 → slice(-1, None) → 只保留最后一个位置
                slice(-logits_to_keep, None) 
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            # 对隐藏状态切片后，通过 lm_head 计算 logits
            # [batch, kept_len, 512] -> [batch, kept_len, vocab_size]
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            # 计算损失
            loss = None
            # 如果提供了标签，就计算交叉熵损失
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous() # 移位操作——去掉 logits 的最后一个位置
                shift_labels = labels[..., 1:].contiguous() # 去掉 labels 的第一个位置（因为没有对应的 logits 预测它）
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), # [batch, seq-1, 6400] → [batch*(seq-1), 6400]
                    shift_labels.view(-1), # [batch, seq-1] → [batch*(seq-1)]
                    ignore_index = -100, # 跳过标签为 -100 的位置
                )

            # 构造输出对象, 创建HuggingFace标准输出格式
            output = CausalLMOutputWithPast(
                loss = loss,
                logits = logits,
                past_key_value = past_key_value,
                hidden_states = hidden_states,
            )
            output.aux_loss = aux_loss
            return output
        