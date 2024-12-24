import torch
import torch.nn.functional as F

# 定义 i-exp 函数
def i_exp(x_tilde):
    # 计算 z 和 p
    ln2 = torch.log(torch.tensor(2.0))
    z = torch.floor(-x_tilde / ln2)
    p = x_tilde + z * ln2
    
    # 计算 exp(p) 的多项式近似
    p_plus_1_353 = p + 1.353
    exp_p = 0.3585 * (p_plus_1_353 ** 2) + 0.344
    
    # 返回 exp(p) 按照 z 右移
    return exp_p, z

# 定义 Softmax 函数
class SecureIntSoftmax:
    def __init__(self, output_bit, max_bit=32):
        self.output_bit = output_bit
        self.max_bit = max_bit

    def forward(self, x):
        # 步骤1：计算 x_tilde = x_j - x_max
        x_max = torch.max(x, dim=-1, keepdim=True).values
        x_tilde = x - x_max
        
        # 步骤2：计算每个 x_tilde 的 exp(x_tilde) 和 z
        exp_values, zs = i_exp(x_tilde)  # Apply i_exp to the entire x_tilde tensor
        
        
        # 步骤4：Softmax 输出
        #factor = torch.floor(2 ** self.max_bit / exp_sum)
        #exp_values = exp_values * factor / 2 ** (self.max_bit - self.output_bit)
        # 步骤5：归一化结果
        #scaling_factor = 1 / 2 ** self.output_bit
        #softmax_output = exp_values * scaling_factor


        exp_values_right_shifted = exp_values * (2 ** (-zs)) 
        
        exp_sum = exp_values_right_shifted.sum(dim=-1, keepdim=True)
        
        softmax_output = exp_values_right_shifted / exp_sum
        return(softmax_output )

# 测试示例
if __name__ == "__main__":
    # 模拟输入张量（多分类任务）
    x = torch.tensor([[0.2, 1.0, 0.5, -0.1], [0.3, -0.2, 0.1, 0.8]], dtype=torch.float32)
    
    # 创建 SecureIntSoftmax 层实例
    secure_softmax = SecureIntSoftmax(output_bit=8)
    
    # 计算 SecureIntSoftmax 输出
    softmax_output_secure = secure_softmax.forward(x)
    
    # 使用原始 torch.softmax 计算标准 Softmax 输出
    softmax_output_standard = F.softmax(x, dim=-1)
    
    # 打印结果
    print("Secure Integer-only Softmax Output:")
    print(softmax_output_secure)
    
    print("\nStandard Softmax Output (torch.softmax):")
    print(softmax_output_standard)
