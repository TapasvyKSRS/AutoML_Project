import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from src.utils import ReLUConvBN, Conv, Identity, FactorizedReduce  # noqa: E402

# this object will be needed to represent the discrete architecture extracted
# from the architectural parameters. See the method genotype() below.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# operations set
OPS = {
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride,
                                                   padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'conv_3x3': lambda C, stride: Conv(C, C, 3, stride, 1),
}

PRIMITIVES = list(OPS.keys())  # operations set as list of strings


class MixedOp(nn.Module):
    """Base class for the mixed operation."""

    def __init__(self, C, stride, one_shot_optimizer="darts"):
        """
        :C: int; number of filters in each convolutional operation
        :stride: int; stride of the convolutional/pooling kernel
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        # iterate thtough the operation set and append them to self._ops
        self.one_shot_optimizer = one_shot_optimizer
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, alphas):
        """
        Compute the softmax of alphas and multiply that element-wise with the
        corresponding output of the operations.

        :x: torch.Tensor; input tensor
        :alphas: torch.Tensor; architectural parameters, either alphas_normal
        or alphas_reduce
        """
        # TODO: for the darts optimizer compute the softmax of the alphas parameter
        weights = F.softmax(alphas, dim=-1)
        list_of_tensors = []
        # TODO: for each operation in self._ops compute the output when it gets
        #      x as input, i.e. op(x) and then multiply it element-wise with
        #      the corresponding alpha vector and add it to the list_of_tensors
        for w, op in zip(weights, self._ops):
            list_of_tensors.append(w * op(x))

        # TODO: add element-wise all the tensors in list of tensors and return
        #      it as output of this method
        return sum(list_of_tensors)


class Cell(nn.Module):
    """Base class for the cells in the search model."""

    def __init__(self, nodes, C_prev, C, reduction, one_shot_optimizer="darts"):
        """
        :nodes: int; number of intermediate nodes in the cell
        :C_prev: int; number of feature maps incoming to the cell
        :C: int; number of filters in each convolutional operation
        :reduction: bool; if it is a reduction or normal cell
        """
        super(Cell, self).__init__()
        self.reduction = reduction
        self.one_shot_optimizer = one_shot_optimizer
        # this preprocessing operation is added to keep the dimensions of the
        # tensors going to the intermediate nodes the same.
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        self._nodes = nodes

        self._ops = nn.ModuleList()
        # iterate throughout each edge of the cell and create a MixedOp
        for i in range(self._nodes):
            for j in range(1 + i):
                stride = 2 if reduction and j < 1 else 1
                op = MixedOp(C, stride, one_shot_optimizer=self.one_shot_optimizer)
                self._ops.append(op)

    def forward(self, input, alphas):
        preprocessed_input = self.preprocess(input)
        alphas = alphas[1] if self.reduction else alphas[0]

        states = [preprocessed_input]
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](h, alphas[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # concatenate the outputs of only the intermediate nodes to form the
        # output node.
        out = torch.cat(states[-self._nodes:], dim=1)
        return out


class Network(nn.Module):
    """Base class for the search model (one-shot model)."""

    def __init__(self, device, nodes=2, num_channels=4, one_shot_optimizer="darts"):
        """
        :device: str; 'cuda' or 'cpu'
        :nodes: int; number of intermediate nodes in each cell
        """
        super(Network, self).__init__()
        self._nodes = nodes
        self.one_shot_optimizer = one_shot_optimizer
        # the one-shot model we are going to use is composed of one reduction
        # cell followed by one normal cell and another reduction cell. The
        # architecture of the 2 reduction cells is the same (they share the
        # alpha_reduction parameter). However the weights of the corresponding
        # operations (convolutional filters) is different.
        reduction_cell_1 = Cell(nodes, 1, num_channels,
                                reduction=True, one_shot_optimizer=self.one_shot_optimizer)
        normal_cell = Cell(nodes, 2 * num_channels, num_channels,
                           reduction=False, one_shot_optimizer=self.one_shot_optimizer)
        reduction_cell_2 = Cell(nodes, 2 * num_channels, 8,
                                reduction=True, one_shot_optimizer=self.one_shot_optimizer)
        self.cells = nn.ModuleList([reduction_cell_1,
                                    normal_cell,
                                    reduction_cell_2])

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2 * 8, 10)

        # initialize the architectural parameters to be equal. We also add a
        # tiny randomly sampled noise for numerical stability.
        self._initialize_alphas(device)

    def forward(self, input):
        x = input
        arch_parameters = [self.alphas_normal, self.alphas_reduce]
        for i, cell in enumerate(self.cells):
            x = cell(x, arch_parameters)
        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self, device):
        """
        Initialize the architectural parameters for the normal and reduction
        cells. The dimensions of each of these variables will be k x num_ops,
        where k is the number of edges in the cell and num_ops is the
        operation set size.
        """
        k = sum(1 for i in range(self._nodes) for n in range(1 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops, device=device),
                                      requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops, device=device),
                                      requires_grad=True)
        self.arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def genotype(self):
        """
        Method for getting the discrete architecture, represented as a Genotype
        object from the DARTS search model.
        """

        def _parse(alphas):
            gene = []
            n = 1
            start = 0
            for i in range(self._nodes):
                end = start + n
                W = alphas[start:end].copy()
                for j in range(i+1):
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(1, self._nodes + 1)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype
