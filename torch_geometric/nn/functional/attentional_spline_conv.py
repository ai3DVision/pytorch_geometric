import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch_spline_conv.functions.spline_weighting import spline_weighting


def attentional_spline_conv(x, edge_index, pseudo, weight, kernel_size,
                            is_open_spline, degree, att_weight, negative_slope,
                            dropout, root_weight, bias):

    row, col = edge_index
    n, e, m_out = x.size(0), row.size(0), weight.size(2)

    x = x.unsqueeze(-1) if x.dim() == 1 else x
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    output = torch.mm(x, root_weight)
    x_row = output[row]

    # Convolve over each node.
    x_col = spline_weighting(x[col], pseudo, weight, kernel_size,
                             is_open_spline, degree)

    # Compute attention across edges.
    alpha = torch.cat([x_row, x_col], dim=-1)
    alpha = torch.matmul(alpha, att_weight.unsqueeze(-1)).squeeze()
    alpha = F.leaky_relu(alpha, negative_slope)

    # Scatter softmax.
    alpha = alpha.exp_()
    alpha /= alpha.new(n).fill_(0).scatter_add_(0, Variable(row), alpha)[row]

    # Apply attention.
    alpha = F.dropout(alpha, p=dropout, training=True)
    x_col = alpha.unsqueeze(-1) * x_col

    # Sum up neighborhoods.
    var_row = Variable(row.view(e, 1).expand(e, m_out))
    output.scatter_add_(0, var_row, x_col)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    # Sample from Bernoulli distribution.
    # z = torch.bernoulli(alpha.data * (1 - dropout))

    return output
