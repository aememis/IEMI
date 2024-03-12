import random
import pygame
import librosa
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from datetime import datetime


class Graphics:
    def __init__(self):
        # self.background_image = pygame.image.load("plot-tsne.png")
        # self.background_image = pygame.transform.scale(
        #     self.background_image, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
        # )
        self.up = False
        self.down = False
        self.zup = False
        self.zdown = False
        self.left = False
        self.right = False

    def draw_circle(self, color, center, size):
        pygame.draw.circle(
            self.screen,
            color,
            (center[0], center[1]),
            size,
        )

    def draw_rects(self, rects, text=True, colors=None):
        for i, points in enumerate(rects):
            rect = pygame.Rect(
                points[0],
                points[1],
                points[2] - points[0],
                points[3] - points[1],
            )
            pygame.draw.rect(self.screen, cfg.BLACK, rect, 1)
            if colors:
                pygame.draw.rect(self.screen, colors[i % len(colors)], rect, 1)
            else:
                pygame.draw.rect(self.screen, cfg.GREY, rect, 1)
            if text:
                self.draw_text(
                    str(i),
                    points[0] + 10,
                    points[1] + 10,
                    font_size=12,
                )

    def draw_frame(self, x, y, width, length):
        rect = pygame.Rect(x, y, width, length)
        pygame.draw.rect(self.screen, cfg.BLACK, rect, 1)

    def draw_user(self, sd):
        # Draw a white dot at the specified position
        # rect = pygame.Rect(x, y, block_size, block_size)
        # pygame.draw.rect(self.screen, WHITE, rect)

        # sd.x, sd.y = self.rotate_point_for_draw([sd.x, sd.y])

        if cfg.USE_SIMULATION:
            x = sd.x
            y = sd.y
            z = sd.z
        else:
            x = sd.x / cfg.MOCAP_WIDTH_PROJECTION * cfg.WINDOW_X / 2 + cfg.WINDOW_X / 2
            y = sd.y / cfg.MOCAP_HEIGHT_PROJECTION * cfg.WINDOW_Y / 2 + cfg.WINDOW_Y / 2
            z = sd.z / cfg.MOCAP_HEIGHT_Z_PROJECTION * cfg.WINDOW_Z

        # draw floor
        self.draw_circle(
            pygame.Color(0, 0, 0, a=0.3),
            (x, y),
            cfg.USER_DOT_SIZE,
        )
        self.draw_circle(
            pygame.Color(255, 255, 255, a=0.5),
            (x, y),
            cfg.USER_DOT_SIZE / 4,
        )

        # draw side
        self.draw_circle(
            pygame.Color(0, 0, 0, a=0.3),
            (x, z + cfg.WINDOW_Y),
            cfg.USER_DOT_SIZE,
        )
        self.draw_circle(
            pygame.Color(255, 255, 255, a=0.5),
            (x, z + cfg.WINDOW_Y),
            cfg.USER_DOT_SIZE / 4,
        )

        # draw front
        self.draw_circle(
            pygame.Color(0, 0, 0, a=0.3),
            (y + cfg.WINDOW_X, z),
            cfg.USER_DOT_SIZE,
        )
        self.draw_circle(
            pygame.Color(255, 255, 255, a=0.5),
            (y + cfg.WINDOW_X, z),
            cfg.USER_DOT_SIZE / 4,
        )

    def draw_dots(self, sd, event_updating_knn_data):
        array_plot = sd.df_tsne.to_numpy()
        # array_plot = self.rotate_data_for_draw(array_plot)

        # scale the projection data for drawing on window
        if cfg.USE_SIMULATION:
            array_plot[:, 1] *= -1
            array_plot[:, 2] *= -1

            array_plot[:, 0] *= cfg.WINDOW_X / 2
            array_plot[:, 1] *= cfg.WINDOW_Y / 2
            array_plot[:, 2] *= cfg.WINDOW_Z / 2

            array_plot[:, 0] += cfg.WINDOW_X / 2
            array_plot[:, 1] += cfg.WINDOW_Y / 2
            array_plot[:, 2] += cfg.WINDOW_Z / 2
        else:
            array_plot[:, 1] *= -1
            array_plot[:, 2] *= -1

            array_plot[:, 0] *= cfg.TARGET_WIDTH_PROJECTION / 2
            array_plot[:, 1] *= cfg.TARGET_HEIGHT_PROJECTION / 2
            array_plot[:, 2] *= cfg.TARGET_HEIGHT_Z_PROJECTION / 2

            array_plot[:, 0] += cfg.TARGET_WIDTH_PROJECTION / 2
            array_plot[:, 1] += cfg.TARGET_HEIGHT_PROJECTION / 2
            array_plot[:, 2] += cfg.TARGET_HEIGHT_Z_PROJECTION / 2

        # draw all points floor
        for point in array_plot:
            self.draw_circle(
                pygame.Color(255, 0, 0, a=0.75),
                (point[0].item(), point[1].item()),
                cfg.DOT_SIZE,
            )

        # draw all points side
        for point in array_plot:
            self.draw_circle(
                pygame.Color(255, 255, 255, a=0.75),
                (point[0].item(), point[2].item() + cfg.WINDOW_Y),
                cfg.DOT_SIZE,
            )

        # draw all points front
        for point in array_plot:
            self.draw_circle(
                pygame.Color(255, 255, 0, a=0.75),
                (point[1].item() + cfg.WINDOW_X, point[2].item()),
                cfg.DOT_SIZE,
            )

        # get knn points
        if not sd.knn_indices is None:
            # print("drawdots", event_updating_knn_data.is_set())
            # print("draw: ", id(sd))
            while True:
                try:
                    # print("draw list: ", sd.knn_indices[0])
                    # print("draw id: ", id(sd.knn_indices[0]))
                    list_knn_indices = sd.knn_indices[0]
                    list_projected_points = array_plot[list_knn_indices, :]
                    break
                except Exception as e:
                    sd.logger.info(f"!!! Skipping once: {str(e)}")
                    time.sleep(0.5)  # left off here problem here!!!!
            list_knn_distances = sd.knn_distances[0]
            mask_distance = list_knn_distances < cfg.SELECT_DISTANCE_THRESHOLD
            list_projected_points_closer = list_projected_points[mask_distance]

            # print((sd.x, sd.y), list_projected_points_closer[0, :], "---")

            # list_projected_points_closer = list_projected_points  # temp!

            # draw knn points
            if len(list_projected_points_closer) > 0:

                # draw highlight knn points
                for point in list_projected_points_closer:

                    # draw floor
                    self.draw_circle(
                        pygame.Color(0, 0, 255, a=100),
                        (point[0].item(), point[1].item()),
                        cfg.SURR_DOT_SIZE,
                    )

                    # draw side
                    self.draw_circle(
                        pygame.Color(255, 255, 0, a=100),
                        (point[0].item(), point[2].item() + cfg.WINDOW_Y),
                        cfg.SURR_DOT_SIZE,
                    )

                    # draw front
                    self.draw_circle(
                        pygame.Color(0, 255, 255, a=100),
                        (point[1].item() + cfg.WINDOW_X, point[2].item()),
                        cfg.SURR_DOT_SIZE,
                    )

                # draw highlight closest knn point
                # draw floor
                self.draw_circle(
                    pygame.Color(0, 170, 0),
                    (
                        list_projected_points_closer[0, 0].item(),
                        list_projected_points_closer[0, 1].item(),
                    ),
                    cfg.SURR_DOT_SIZE_2,
                )

                # draw side
                self.draw_circle(
                    pygame.Color(0, 170, 0),
                    (
                        list_projected_points_closer[0, 0].item(),
                        list_projected_points_closer[0, 2].item() + cfg.WINDOW_Y,
                    ),
                    cfg.SURR_DOT_SIZE_2,
                )

                # draw front
                self.draw_circle(
                    pygame.Color(0, 170, 0),
                    (
                        list_projected_points_closer[0, 1].item() + cfg.WINDOW_X,
                        list_projected_points_closer[0, 2].item(),
                    ),
                    cfg.SURR_DOT_SIZE_2,
                )
                # print(
                #     "closest: ",
                #     (
                #         list_projected_points_closer[0, 0].item(),
                #         list_projected_points_closer[0, 1].item(),
                #         list_projected_points_closer[0, 2].item() + cfg.WINDOW_HEIGHT,
                #     ),
                #     "current: ",
                #     (sd.x, sd.y, sd.z),
                # )

    def draw_text(self, text, x, y, font_size=20):
        font = pygame.font.SysFont("ubuntumono", font_size)
        text = font.render(text, True, cfg.BLACK)
        self.screen.blit(text, (x, y))

    def init_player(self, sd, event_init_player):
        # pygame.mixer.init()
        # sd.logger.info("Initiating player...")
        # pygame.mixer.quit()
        # pygame.mixer.pre_init(
        #     frequency=22050,
        #     size=-16,
        #     channels=4,
        #     buffer=512,
        #     allowedchanges=0,
        # )
        # pygame.mixer.init(
        #     frequency=22050,
        #     size=-16,
        #     channels=4,
        #     buffer=512,
        #     allowedchanges=0,
        # )
        # sd.logger.info(f"Player mixer details: {pygame.mixer.get_init()}")

        sd.logger.info("Loading sounds...")
        sd.sounds = {}
        for i in sd.df_samples.index.values:
            audio = sd.df_samples.iloc[i]["signal"]
            audio = librosa.resample(audio, orig_sr=22050, target_sr=5512.5)
            audio_int16 = np.array(np.int16(audio * 32767), dtype=np.int16)
            mask = np.zeros(4)
            mask[random.randint(0, 3)] = 1
        #     sd.sounds[i] = pygame.mixer.Sound(
        #         array=(
        #             np.column_stack(
        #                 (audio_int16, audio_int16, audio_int16, audio_int16)
        #             )
        #             * mask
        #         ).astype(np.int16)
        #     )

        # sd.logger.info(f"Loaded {len(sd.sounds)} samples in memory.")
        print("donedonedonedonedone")  ####
        event_init_player.set()

    def start_graphics(
        self,
        sd,
        operator,
        event_populating,
        event_updating_knn_data,
        event_init_player,
        event_umap_projection,
    ):
        """Start the graphics loop"""

        event_populating.wait()

        pygame.init()
        self.clock = pygame.time.Clock()
        if cfg.FULLSCREEN:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(
                (cfg.WINDOW_WIDTH_APP, cfg.WINDOW_HEIGHT_APP)
            )
        self.speed = 1
        sd.x = cfg.WINDOW_X / 2
        sd.y = cfg.WINDOW_Y / 2
        sd.z = cfg.WINDOW_Z / 2

        flag_timer_started = False

        self.init_player(sd, event_init_player)
        prev_indexes = np.zeros(3)

        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if cfg.USE_SIMULATION:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.left = True
                        if event.key == pygame.K_RIGHT:
                            self.right = True
                        if event.key == pygame.K_UP:
                            self.up = True
                        if event.key == pygame.K_DOWN:
                            self.down = True
                        if event.key == pygame.K_LSHIFT:
                            self.speed = 6
                        if event.key == pygame.K_w:
                            self.zup = True
                        if event.key == pygame.K_s:
                            self.zdown = True
                        if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                            if (
                                event_populating.is_set()
                                and event_updating_knn_data.is_set()
                            ):
                                operator.rate(int(chr(event.key)))
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT:
                            self.left = False
                        if event.key == pygame.K_RIGHT:
                            self.right = False
                        if event.key == pygame.K_UP:
                            self.up = False
                        if event.key == pygame.K_DOWN:
                            self.down = False
                        if event.key == pygame.K_LSHIFT:
                            self.speed = 1
                        if event.key == pygame.K_w:
                            self.zup = False
                        if event.key == pygame.K_s:
                            self.zdown = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # if event.key == pygame.BUTTON_LEFT:
                    if event_populating.is_set() and event_updating_knn_data.is_set():
                        operator.rate(1)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t and not flag_timer_started:
                        sd.start_time = datetime.now()

            if cfg.USE_SIMULATION:
                if self.left:
                    sd.x -= cfg.MOVE * self.speed
                if self.right:
                    sd.x += cfg.MOVE * self.speed
                if self.up:
                    sd.y -= cfg.MOVE * self.speed
                if self.down:
                    sd.y += cfg.MOVE * self.speed
                if self.zup:
                    sd.z -= cfg.MOVE * self.speed
                if self.zdown:
                    sd.z += cfg.MOVE * self.speed

                # Keep the dot within the window boundaries
                sd.x = max(
                    0 + cfg.USER_DOT_SIZE, min(sd.x, cfg.WINDOW_X - cfg.USER_DOT_SIZE)
                )
                sd.y = max(
                    0 + cfg.USER_DOT_SIZE, min(sd.y, cfg.WINDOW_Y - cfg.USER_DOT_SIZE)
                )
                sd.z = max(
                    0 + cfg.USER_DOT_SIZE, min(sd.z, cfg.WINDOW_Z - cfg.USER_DOT_SIZE)
                )

            event_populating.wait()
            event_updating_knn_data.wait()

            self.screen.fill(cfg.WHITE)

            ####
            # self.clock.tick(24)  # Limit frames per second
            # continue  ###
            ####

            self.draw_dots(sd, event_updating_knn_data)
            self.draw_user(sd)

            # draw frames
            self.draw_frame(0, 0, cfg.WINDOW_X, cfg.WINDOW_Y)
            self.draw_frame(0, cfg.WINDOW_Y, cfg.WINDOW_X, cfg.WINDOW_Z)
            self.draw_frame(cfg.WINDOW_X, 0, cfg.WINDOW_Y, cfg.WINDOW_Z)

            # draw zones
            ## scale xs
            rects_plot = np.copy(sd.zones)
            rects_plot[:, 0] = (
                (rects_plot[:, 0] / cfg.MOCAP_WIDTH_PROJECTION + 1) * cfg.WINDOW_X / 2
            )
            rects_plot[:, 2] = (
                (rects_plot[:, 2] / cfg.MOCAP_WIDTH_PROJECTION + 1) * cfg.WINDOW_X / 2
            )

            ## scale ys
            rects_plot[:, 1] = (
                (rects_plot[:, 1] / cfg.MOCAP_HEIGHT_PROJECTION + 1) * cfg.WINDOW_Y / 2
            )

            rects_plot[:, 3] = (
                (rects_plot[:, 3] / cfg.MOCAP_HEIGHT_PROJECTION + 1) * cfg.WINDOW_Y / 2
            )

            self.draw_rects(rects_plot)  # , colors)

            # draw texts
            self.draw_text("TOP (X & Y)", 0, 0)
            self.draw_text("SIDE (X & Z)", 0, cfg.WINDOW_Y)
            self.draw_text("FRONT (Y & Z)", cfg.WINDOW_X, 0)

            ts = []
            ts.append(f"Position: {(round(sd.x, 2), round(sd.y, 2), round(sd.z, 2))}")
            ts.append(f"Zone: {sd.current_zone}")
            ts.append(f"Id: {sd.closest_point_index}")

            ts.append(
                f"Time: {str(datetime.now() - sd.start_time) if not sd.start_time is None else 'N/A'}"
            )  # .strftime('%H:%M:%S')

            for i, t in enumerate(ts):
                self.draw_text(t, cfg.WINDOW_X + 20, cfg.WINDOW_Z + 20 + i * 10, 18)

            pygame.display.update()

            self.clock.tick(24)  # Limit frames per second
