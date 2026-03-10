import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

batch_idx = 0
a =target_joints[batch_idx,:,:].detach().cpu().numpy()
point_numbers = [str(i) for i in range(len(a))]
fig = go.Figure(data=[go.Scatter3d(
    x=a[:, 0],
    y=a[:, 1],
    z=a[:, 2],
    mode='markers+text',
    text=point_numbers,
    textposition="middle center",
    hoverinfo='text+x+y+z',
)])
fig.update_layout(scene=dict(
    xaxis_title='X',
    xaxis=dict(dtick=0.1),
    yaxis_title='Y',
    yaxis=dict(dtick=0.1),
    zaxis_title='Z',
    zaxis=dict(dtick=0.1),
    aspectmode='manual',
    aspectratio=dict(x=0.5, y=0.2, z=1),
))
fig.write_html("tools/tmp/joints.html")