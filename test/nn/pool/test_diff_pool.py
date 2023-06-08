from torch_geometric.nn.dense import dense_diff_pool

def test_dense_diff_pooling():
    B, N, F, C = 2, 32, 8, 4
    x = torch.randn((B,N,F,), dtype=torch.float, device='cuda')
    adj = torch.randint(high=2, size=(B,N,N), dtype=torch.float).cuda()
    s = torch.randint(high=2, size=(B,N,C), dtype=torch.float).cuda()
    out, _, _, _dense_diff_pool(x, adj, s)
    assert out.size() == (B, C, F)

if __name__ == '__main__':
    from torch_geometric import seed_everything
    from torch_geometric.profile import benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(12345)
    for B in [2**i for i in range(1,8)]:
      for N in [2**i for i in range(1,10)]:
        for F in [2**i for i in range(1,8)]:
          for C in [2**i for i in range(1,4)]:
            print(f'(B, N, F, C) = {(B, N, F, C)}')
            x = torch.randn((B,N,F,), dtype=torch.float, device='cuda')
            adj = torch.randint(high=2, size=(B,N,N), dtype=torch.float).cuda()
            s = torch.randint(high=2, size=(B,N,C), dtype=torch.float).cuda()
            funcs = [dense_diff_pool]
            func_names = ['dense_diff_pool']
            args_list = [(x, adj, s)]
            benchmark(
                funcs=funcs,
                func_names=func_names,
                args=args_list,
                num_steps=50 if device == 'cpu' else 500,
                num_warmups=10 if device == 'cpu' else 100,
            )