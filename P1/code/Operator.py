import time
import math
import random
import config as cfg
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from umap import UMAP


class Operator:
    def __init__(self, sd, pop):
        self.sd = sd
        self.pop = pop
        self.threshold = None
        self.current_population = 0

    def init_populate(self):
        self.sd.logger.info("Populating for the first time...")
        self.current_population += 1

        for i in range(cfg.POPULATION_SIZE):
            # f_carrier = random.randint(100, 1000)
            # f_mod = random.randint(10, 250)
            # f_mod2 = random.randint(5, 25)
            # new_ind = Individual(f_carrier, f_mod, f_mod2)

            # random
            p1 = random.random()
            p2 = random.random()
            p3 = random.random()
            p4 = random.random()
            p5 = random.random()
            p6 = random.random()

            # ["p1", "p2", "p3", "p4", "p5", "p6", "score", "pop"]
            new_ind = [p1, p2, p3, p4, p5, p6, -1, None]

            self.pop.list_population.append(new_ind)
        self.pop.df_population = pd.DataFrame(
            self.pop.list_population, columns=cfg.DATA_FIELDS
        )
        self.pop.df_population["pop"] = self.current_population
        self.sd.df_population = self.pop.df_population
        self.sd.logger.info(f"Populated {len(self.sd.df_population.index)} individuals")

    def tsne_project(self):
        tsne = TSNE(
            n_components=2,
            # random_state=42,
            # perplexity=10,
            # early_exaggeration=8,
        )
        tsne_data = tsne.fit_transform(self.sd.df_population.iloc[:, :-2])
        return pd.DataFrame(tsne_data)

    def umap_project(self):
        self.sd.logger.info("UMAP projecting the population...")
        umap_model = UMAP(
            n_components=3,
            # random_state=42,  # Optional: for reproducible results
            # n_neighbors=15,  # Default is 15, controls the balance between local vs global structure
            min_dist=0.1
            * 2
            * self.current_population,  # Default is 0.1, controls how tightly UMAP is allowed to pack points together
        )
        umap_data = umap_model.fit_transform(self.sd.df_population.iloc[:, :-2])
        return pd.DataFrame(umap_data)

    def normalize_projection_data(self, df_proj):
        proj_min_x, proj_min_y, proj_min_z = df_proj.min(axis=0)
        proj_max_x, proj_max_y, proj_max_z = df_proj.max(axis=0)

        df_proj_norm = pd.DataFrame()
        df_proj_norm[0] = (
            (df_proj.iloc[:, 0] - proj_min_x) / (proj_max_x - proj_min_x) * 2
        )
        df_proj_norm[1] = (
            (df_proj.iloc[:, 1] - proj_min_y) / (proj_max_y - proj_min_y) * 2
        )
        df_proj_norm[2] = (
            (df_proj.iloc[:, 2] - proj_min_z) / (proj_max_z - proj_min_z) * 2
        )
        df_proj_norm.iloc[:, 0] -= 1
        df_proj_norm.iloc[:, 1] -= 1
        df_proj_norm.iloc[:, 2] -= 1

        # plt.figure()
        # plt.scatter(df_proj_norm.iloc[:, 0], df_proj_norm.iloc[:, 1], s=1.5)
        # plt.savefig("norm_proj.png")

        return df_proj_norm

    def rate(self, score):
        # if not self.sd.flag_populating:
        # self.sd.flag_rating = True
        selected_idx = self.sd.knn_indices[0][0 : cfg.NUMBER_OF_IDX_TO_APPLY_SCORE]
        self.sd.df_population.loc[selected_idx, "score"] = score
        # self.sd.flag_rating = False
        self.sd.logger.info(f"index {selected_idx} scored {score}")

    def apply_selection(self, threshold):
        # mask_select = self.sd.df_population["score"] < self.threshold
        # self.sd.df_dead = pd.concat(
        #     [self.sd.df_dead, self.sd.df_population.loc[~mask_select, :]]
        # )
        # self.sd.df_population = self.sd.df_population.loc[mask_select, :]

        df_top = self.sd.df_population.nlargest(threshold, "score", keep="first")
        self.sd.df_dead = pd.concat(
            [self.sd.df_dead, self.sd.df_population], ignore_index=True
        ).reset_index(drop=True)
        self.sd.df_population.drop(self.sd.df_population.index, inplace=True)
        return df_top.copy(deep=True).reset_index(drop=True)

    def _cross_parts(self, arr1, arr2):
        crossover_point = np.random.randint(1, len(arr1) - 1)
        offspring1 = np.concatenate((arr1[:crossover_point], arr2[crossover_point:]))
        offspring2 = np.concatenate((arr2[:crossover_point], arr1[crossover_point:]))
        return [offspring1, offspring2]

    def apply_crossover(self, df):
        nd_new_values = []
        for i in range(len(df.index)):
            for j in range(i + 1, len(df.index), 1):
                arr1 = df.iloc[i].values[:-2]
                arr2 = df.iloc[j].values[:-2]
                list_new_values = self._cross_parts(arr1, arr2)
                list_new_values_reshaped = np.append(
                    list_new_values, [[None, self.current_population]] * 2, axis=1
                )
                nd_new_values.extend(list_new_values_reshaped)
        df_new_values = pd.DataFrame(nd_new_values, columns=cfg.DATA_FIELDS)
        # df = pd.concat([df, df_new_values], ignore_index=True)
        df_new_values["score"] = -1
        return df_new_values

    def apply_mutation(self, df):
        # self.sd.df_population = pd.concat(
        #     [
        #         self.sd.df_population,
        #         self.sd.df_population,
        #         self.sd.df_population,
        #     ]
        # ).reset_index(drop=True)

        mutation_rate = 0.5
        self.sd.logger.info(f"Mutation rate: {mutation_rate}")

        list_new_datapoints = []
        for i in range(len(df)):
            if np.random.random() < mutation_rate:
                new_datapoint = df.iloc[i].copy(deep=True)
                for j in range(len(new_datapoint[:-2])):
                    if np.random.random() < mutation_rate:
                        new_datapoint[j] += np.random.uniform(
                            -cfg.MUTATION_SCALE, cfg.MUTATION_SCALE
                        )
                list_new_datapoints.append(new_datapoint)
        if len(list_new_datapoints) > 0:
            df_new_datapoints = pd.DataFrame(
                list_new_datapoints, columns=cfg.DATA_FIELDS
            )
            df = pd.concat([df, df_new_datapoints])
        return df

    def fit_knn(self):
        self.sd.knn = NearestNeighbors(n_neighbors=cfg.K)
        self.sd.knn.fit(self.sd.df_tsne.to_numpy())

    def operate(self, event_populating):
        while True:
            # df_tsne = self.tsne_project()  # temp
            df_umap = self.umap_project()

            # df_tsne_norm = normalize_projection_data(df_tsne)  # temp
            df_umap_norm = self.normalize_projection_data(df_umap)

            # sd.df_tsne = df_tsne_norm
            self.sd.df_tsne = df_umap_norm  # temp assinging to tsne for convenience
            # !!!

            self.fit_knn()

            # # normalize if using the pygame window
            # self.sd.norm_x = self.sd.x / WINDOW_WIDTH * 2 - 1
            # self.sd.norm_y = self.sd.y / WINDOW_HEIGHT * 2 - 1
            # self.sd.norm_y *= -1

            # # normalize if using mocap
            # # sd.norm_x = sd.x / MOCAP_WIDTH_PROJECTION
            # # sd.norm_y = sd.y / MOCAP_HEIGHT_PROJECTION * -1

            # user_point = np.array([[self.sd.norm_x, self.sd.norm_y]])
            # distances, indices = self.sd.knn.kneighbors(user_point)

            # self.sd.knn_distances = distances
            # self.sd.knn_indices = indices

            self.sd.logger.info("Done projecting.")
            self.sd.logger.info(
                f"Selection threshold divisor: {cfg.SELECTION_THRESHOLD_DIVISOR}"
            )
            event_populating.set()

            while True:
                threshold = round(
                    len(self.sd.df_population.index) / cfg.SELECTION_THRESHOLD_DIVISOR
                )  # cfg.N_LARGEST_TO_SELECT
                total_rated = (self.sd.df_population["score"] > 0).sum()
                if total_rated >= threshold:
                    self.sd.logger.info(f"Rated {total_rated} individuals.")

                    event_populating.clear()

                    self.current_population += 1

                    df = self.apply_selection(threshold)
                    self.sd.logger.info(
                        f"Selected, populating from {len(df.index)} individuals..."
                    )
                    df = self.apply_crossover(df)
                    self.sd.logger.info(
                        f"Crossover resulted {len(df.index)} individuals."
                    )
                    df = self.apply_mutation(df)
                    self.sd.logger.info(
                        f"Mutation resulted {len(df.index)} individuals."
                    )
                    self.sd.df_population = df.reset_index(drop=True)

                    self.sd.logger.info(
                        f"New population with {len(self.sd.df_population.index)}"
                    )

                    break

                time.sleep(0.5)
