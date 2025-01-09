import torch
import torch.nn.functional as F

def i_exp(x_tilde):

    ln2 = torch.log(torch.tensor(2.0))
    z = torch.floor(-x_tilde / ln2)
    p = x_tilde + z * ln2
    

    p_plus_1_353 = p + 1.353
    exp_p = 0.3585 * (p_plus_1_353 ** 2) + 0.344
    

    return exp_p, z


class SecureIntSoftmax:
    def __init__(self, output_bit, max_bit=32):
        self.output_bit = output_bit
        self.max_bit = max_bit

    def forward(self, x):

        x_max = torch.max(x, dim=-1, keepdim=True).values
        x_tilde = x - x_max
        

        exp_values, zs = i_exp(x_tilde)  # Apply i_exp to the entire x_tilde tensor
        
        

        #factor = torch.floor(2 ** self.max_bit / exp_sum)
        #exp_values = exp_values * factor / 2 ** (self.max_bit - self.output_bit)

        #scaling_factor = 1 / 2 ** self.output_bit
        #softmax_output = exp_values * scaling_factor


        exp_values_right_shifted = exp_values * (2 ** (-zs)) 
        
        exp_sum = exp_values_right_shifted.sum(dim=-1, keepdim=True)
        
        softmax_output = exp_values_right_shifted / exp_sum
        return(softmax_output )


if __name__ == "__main__":

    x = torch.tensor([[0.2, 1.0, 0.5, -0.1], [0.3, -0.2, 0.1, 0.8]], dtype=torch.float32)
    

    secure_softmax = SecureIntSoftmax(output_bit=8)
    

    softmax_output_secure = secure_softmax.forward(x)
    

    softmax_output_standard = F.softmax(x, dim=-1)

    print("Secure Integer-only Softmax Output:")
    print(softmax_output_secure)
    
    print("\nStandard Softmax Output (torch.softmax):")
    print(softmax_output_standard)
