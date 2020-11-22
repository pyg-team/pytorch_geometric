from torch_geometric.nn.conv.digcn_conv import DIGCNConv,get_appr_directed_adj,get_second_directed_adj
from torch.nn import Module,Linear,ModuleList
from torch import cat
from torch.nn.functional import dropout
class DiGCN(Module):
    def __init__(self, in_channels,out_channels,dropout_prob,nos_block,fusion_type="sum"):
        super(DiGCN, self).__init__()
        self.Lin = Linear(in_channels,out_channels) #0th order
        self.conv1 = DIGCNConv(in_channels, out_channels)
        self.conv2 = DIGCNConv(in_channels, out_channels)
        self.dropout_prob = dropout_prob
        self.nos_block = nos_block
        self.fusion_type = fusion_type
        if fusion_type == "concat": 
            self.concat_lin = ModuleList([Linear(out_channels*3,in_channels) for i in range(nos_block)])     
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def encode(self,x,edge_index,alpha,**kwargs):
        """encoder block for receptive fields based inception convolutional layers"""
        edge_index,edge_weight = get_appr_directed_adj(alpha,edge_index,x.size(0))
        edge_index_2,edge_weight_2 = get_second_directed_adj(edge_index,x.size(0))
        for i in range(self.nos_block):
            Out0 = dropout(self.Lin(x), p=self.dropout_prob, training=self.training) 
            Out1 = dropout(self.conv1(x,edge_index,edge_weight), p=self.dropout_prob, training=self.training)
            Out2 = dropout(self.conv2(x,edge_index_2,edge_weight_2), p=self.dropout_prob, training=self.training)
            if self.fusion_type == "sum":
                x = dropout(Out0+Out1+Out2, p=self.dropout_prob, training=self.training) 
            elif self.fusion_type == "concat":
                x = dropout(self.concat_lin[i](cat((Out0,Out1,Out2),1)), p=self.dropout_prob, training=self.training) 

        return x
if __name__ == "__main__":
    import torch
    Model = DiGCN(100,100,0.3,3,"concat")
    X = torch.randn(7,100)
    Edge_Index = torch.stack([torch.LongTensor(i) for i in [[0,1],[2,3],[5,6],[3,4],[3,6]]]).t()
    print("Starting...")
    Output = Model.encode(X,Edge_Index,0.1)
    print(Output)