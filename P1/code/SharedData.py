import numpy as np


class SharedData:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.norm_x = None
        self.norm_y = None
        self.list_pos = None
        self.current_zone = -1
        self.zones = None
        self.angle_center = -1
        self.distance_center = -1
        self.qom = -1
        self.top = -1
        self.dispersion = -1
        self.lhpos = [0, 0, 0]
        self.rhpos = [0, 0, 0]
        self.headpos = [0, 0, 0]
        self.backpos = [0, 0, 0]
        self.data_buffer = np.zeros((100, 3))
        self.df_population = None
        self.df_dead = None
        self.df_tsne = None
        self.knn_distances = None
        self.knn_indices = None
        self.closest_point_index = None
        self.selected_params = None
        self.musical_params = None
        self.cluster_labels = None
        self.start_time = None

    # TODO
