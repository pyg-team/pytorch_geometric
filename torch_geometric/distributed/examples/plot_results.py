import pandas as pd
import plotly.express as px

file = 'train'
for file in ['train', 'test']:
    df = pd.read_csv(f"/home/pyg/graphlearn-dev/res/{file}.csv")
    fig = px.line(df.loss, 
                  labels={"index": "batch",
                          "value": "loss"},
                  title=f"{file} loss")
    fig.write_image(f"/home/pyg/graphlearn-dev/res/{file}.png")
