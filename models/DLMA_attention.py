import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Attention

class DLMAAttention(nn.Module):
    r"""
    Applies a dot attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.DotAttention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, output_dim):
        super(DLMAAttention, self).__init__()
        self.linear_in = nn.Linear(output_dim, 2*output_dim)
        self.linear_out = nn.Linear(3*output_dim, output_dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        # output: (batch, de_len, output_size)
        # context: (batch, en_len, 2 * output_size)
        batch_size = output.size(0)
        hidden_size = output.size(2)

        en_len = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        output_linear = self.linear_in(output)
        output_linear = torch.tanh(output_linear)
        attn = torch.bmm(output_linear, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, en_len), dim=1).view(batch_size, -1, en_len)
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 3 * hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn

class DLMAAddAttention(nn.Module):
    def __init__(self, output_dim, class_num):
        super(DLMAAddAttention, self).__init__()

        self.att = Attention(output_dim, class_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')

    def forward(self, output, context):
        # output : (batch, de_len, output_size)
        # context : (batch, time_steps, 2* hidden_unit)
        # Return:
        # output :  (batch, class_num * 2)
        midpoint = context.shape[2] // 2
        emb1, emb2 = context[:,:,:midpoint], context[:,:,midpoint:]
        # transform to (batch, output_size,time_step,1)
        emb1 = emb1.transpose(1,2).unsqueeze(3)
        emb2 = emb2.transpose(1,2).unsqueeze(3)
        
        batch_size, de_len, output_size = output.shape
        # TODO: do not support de_len > 1       
        output1 = self.att(emb1 + output.view((batch_size, output_size,1,1)))
        output2 = self.att(emb2 + output.view((batch_size, output_size,1,1)))
        out = torch.cat((output1,output2),dim = 1).unsqueeze(1)
        return out, None
