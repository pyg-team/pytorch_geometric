import torch

# from ...sparse import mm, sum


def spline_gcn(adj, features, weight, bias, kernel_size, spline_degree):
    B = torch.FloatTensor()
    C = torch.LongTensor()
    dim = len(kernel_size)
    # Adj has size: torch.Size([n, n, dim])
    # -------------------------------------------------------------------------
    # COMPUTE B AND C with size [|E|, dim, spline_degree + 1]
    # 1. Scale ang1 and ang2 to be in interval [0, 1]

    # 2. Calculate close b-spline for ang1 and ang2

    # 3. Calculate open b-spline for dist

    # 4. Concatenate to one big tensor (maybe unneccassary, will see)

    # THIS SHOULD BE REALLY REALLY FAST!!
    # -------------------------------------------------------------------------
    # DO THE CONVOLUTION:
    C = B.view(-1, dim * (spline_degree + 1))
    C = C.view(-1, dim * (spline_degree + 1))

    # 1. We need to compute [|E|, M_out]:
    #   1. Slice the features F => [|E|, M_in]
    #
    #   2. for one i in |E| compute the following:
    #       1. Get b := B_{i} => [dim, spline_degree + 1] (slice, no problem)
    #       2. Get c := C_{i} => [dim, spline_degree + 1] (slice, no problem)
    #       3. Get w := W_{c} => [dim, spline_degree + 1, M_in, M_out]
    #               (slice, no problem)
    #       4. Get the current row r := R_{i} (no problem)
    #       5. Get f := F_{i} => [M_in]
    #
    #       6. Compute d := B * W => [dim, spline_degree + 1, M_in, M_out]
    #       7. Compute e := [dim, spline_degree + 1, M_out]
    #       8. Sum to [M_out]
    #
    # TODO: it's maybe better to flatten [dim, spline_degree + 1 to one
    # dimension.

    # 2. Then we can do a `sum` over the rows to obtain [N, M_out]
    # 3. Finally we can weighten the root nodes (extra weights?)

    # -------------------------------------------------------------------------
    # TODO: Implement spline_degree.values > 1
    pass
