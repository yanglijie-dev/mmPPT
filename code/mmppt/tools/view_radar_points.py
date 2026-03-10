import numpy as np
import plotly.graph_objects as go
import os


def visualize_point_cloud(npz_file_path, output_folder, point_size=2):
    try:
        points = np.load(npz_file_path)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        fig_xz = go.Figure(data=[go.Scatter(
            x=x,
            y=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color='red',
                opacity=0.8
            )
        )])

        fig_xz.update_layout(
            xaxis_title='X',
            yaxis_title='Z',
            title='Point clouds in XZ plane',
            xaxis=dict(
                showline=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0)',
                zeroline=False,
                tickfont=dict(color='rgba(0,0,0,0)'),
                title_font=dict(color='rgba(0,0,0,0)')
            ),
            yaxis=dict(
                showline=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0)',
                zeroline=False,
                tickfont=dict(color='rgba(0,0,0,0)'),
                title_font=dict(color='rgba(0,0,0,0)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_png_path = os.path.join(output_folder, 'xz_point_cloud_visualization.png')
        fig_xz.write_image(output_png_path, format='png', scale=1, width=800, height=600)

        print(f"2D point clouds in XZ plane have been saved as: {output_png_path}")

    except FileNotFoundError:
        print(f"Error, file not found: {npz_file_path}。")
    except Exception as e:
        print(f"Unknown error: {e}")


if __name__ == "__main__":
    npz_file_path = '/media/dlz/Data/yanglj/mmppt/code/mmppt/occlusion_seq_1_frame_1610.npy'
    output_folder = './tmp/radarPoint'
    point_size = 5
    visualize_point_cloud(npz_file_path, output_folder, point_size)

