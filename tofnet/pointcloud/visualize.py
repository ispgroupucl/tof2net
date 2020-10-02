"""Visualizing pointcloud data."""
import numpy as np
import open3d

from tofnet.data.generators import DEFAULT_SEGMENT_COLORS

def visualize_pointcloud(points, color=None, geometries=None, color_map=None):
    """Visualize a set of points, including possible geometries.

    Arguments:
        points (np.array): array of shape (N,3)
        color (np.array): array of shape (N,)
        geometries (list of open3d.Geometry): lines, vertices, ...
        color_map (np.array): array of shape (N,3) to transform "gray" points
            into RGB

    """
    points = np.reshape(points, (-1, 3))
    geometries = geometries or []
    # if color is None:
    #     color = np.zeros(points.shape[0], dtype=np.uint8)

    # geometries.append(create_lineset([0, 0, 0], [1, 0, 0], color=[1, 0, 0]))
    # geometries.append(create_lineset([0, 0, 0], [0, 1, 0], color=[0, 1, 0]))
    # geometries.append(create_lineset([0, 0, 0], [0, 0, 1], color=[0, 0, 1]))
    opcd = open3d.geometry.PointCloud()
    good_points = points[~np.isnan(points[:, 0])]
    opcd.points = open3d.utility.Vector3dVector(good_points)
    if color is not None:
        if color_map is None:
            color = np.reshape(color, (-1,))
        if color_map is False:
            if color.shape[-1] != 3:
                color = np.reshape(color, (-1,))
                color = np.stack([color]*3, axis=-1)
        color_map = (np.array(DEFAULT_SEGMENT_COLORS)
                    if color_map is None else
                    color_map)
        colors = color_map[color] if color_map is not False else color
        colors = colors[~np.isnan(points[:, 0])]
        opcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([opcd, *geometries])


def create_lineset(point1, point2, color=None):
    """Creates a line from two points with an (optional) color."""
    line = open3d.geometry.LineSet()
    line.points = open3d.utility.Vector3dVector([point1, point2])
    line.lines = open3d.utility.Vector2iVector([[0, 1]])
    if color is not None:
        line.colors = open3d.utility.Vector3dVector([color])
    return line

def create_bbox(center, x_axis, y_axis, z_axis, color=None):
    """Creates a bounding box with color."""
    bbox = open3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.x_axis = np.array(x_axis)
    bbox.y_axis = np.array(y_axis)
    bbox.z_axis = np.array(z_axis)
    if color is not None:
        bbox.color = np.array(color)
    return bbox
