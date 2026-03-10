import numpy as np
import plotly.graph_objects as go


data_name = "train/sequence_0/mesh/frame_1500.npz"
data_path='/media/dlz/Data/yanglj/dataset/mmBody/'+data_name
data = np.load( data_path)
joints = data['joints']
pose = data['pose']
pose_hand = data['pose_hand']
shape = data['shape']
vertices = data['vertices']

x_joints = joints[:,0]
y_joints = joints[:,1]
z_joints = joints[:,2]


fig = go.Figure(data=[go.Scatter3d(
    x=x_joints,
    y=y_joints,
    z=z_joints,
    mode='markers'
)])

fig.update_layout(scene = dict(
                    xaxis_title='X',xaxis=dict(dtick=0.1),
                    yaxis_title='Y',yaxis=dict(dtick=0.1),
                    zaxis_title='Z',zaxis=dict(dtick=0.1),
                    aspectmode='manual',
                    aspectratio=dict(x=0.5, y=0.2, z=1),
))
try:

except:
    fig.write_html("tmp/joints.html")

import shutil
image_name = data_path.replace("mesh", "image/master")
image_name = image_name.replace("npz", "png")
print(image_name)
shutil.copy2(image_name, "tmp")




