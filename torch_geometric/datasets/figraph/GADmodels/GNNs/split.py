
for year in range(2010,2023,1):
    # 读取原始文本文件
    file_name = 'data/graph_edges_listed/graph_edges_listed'+str(year)+'.txt'
    with open(file_name, 'r') as input_file:
        lines = input_file.readlines()

    # 处理每一行数据
    modified_lines = []
    for line in lines:
        # 分割每一行数据
        start_node, end_node, edge_feature = line.strip().split()

        # 对起始节点和终止节点进行修改
        start_node = str(int(start_node[1:])) + str(year)
        end_node = str(int(end_node[1:])) + str(year)

        # 拼接成新的行
        modified_line = f"{start_node} {end_node} {edge_feature}\n"
        modified_lines.append(modified_line)
    file_name1 = 'data/newEdge/graph_edges_listed' + str(year) + '_new.txt'
    # 将修改后的数据写入新的文本文件
    with open(file_name1, 'w') as output_file:
        output_file.writelines(modified_lines)
