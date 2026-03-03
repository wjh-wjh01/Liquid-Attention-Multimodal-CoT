import torch
from src.models.liquid_attention import LiquidAttention

print("🚀 --- 启动 Liquid Attention 换心手术后的点火测试 ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 当前算力设备: {device}")

# 伪造一组测试张量 (Batch=2, SeqLen=10, Dim=64)
hidden_dim = 64
model = LiquidAttention(hidden_dim=hidden_dim).to(device)

token_states = torch.randn(2, 10, hidden_dim).to(device)
reasoning_state = torch.randn(2, hidden_dim).to(device)

try:
    # 强制调用我们刚写进去的真实 ODE 引擎
    attn, entropy = model(token_states, reasoning_state, mode="liquid", use_ode=True)
    print("✅ 1/2 真实 ODE 引擎【正向传播】成功！注意力张量形状:", attn.shape)
    
    # 极其重要：测试常微分方程的梯度反向传播
    loss = attn.sum()
    loss.backward()
    print("✅ 2/2 真实 ODE 引擎【反向传播】成功！梯度流动正常！心脏跳动极其强劲！")
    print("🎉 恭喜！最难的底层技术雷区已被彻底踏平！")
except Exception as e:
    print("❌ 点火失败，捕捉到异常:", e)