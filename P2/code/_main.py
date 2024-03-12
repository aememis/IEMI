import random
import logging
import threading
import time
import pygame
import sounddevice
import math
import librosa
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygetwindow as gw
from pythonosc.udp_client import SimpleUDPClient
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as cfg
from Graphics import Graphics
from Graphics3D import Graphics3D
from Operator import Operator
from Population import Population
from SharedData import SharedData
from Streamer import Streamer
from CorpusReader import CorpusReader
from concurrent.futures import ThreadPoolExecutor
from collections import deque

EXECUTOR = ThreadPoolExecutor(max_workers=5)
current_playbacks = deque(maxlen=5)


def play_sound(
    audio, samplerate, playback_queue, channel_indices=None, volume_mask=None
):
    # Determine the desired number of channels (e.g., 8 for an 8-channel setup)
    desired_channels = 8

    # Check if audio is mono; if so, expand it to the desired number of channels
    if audio.ndim == 1:
        # Create a multichannel array filled with zeros
        multichannel_audio = np.zeros((len(audio), desired_channels))

        # If specific channels are provided, copy the mono audio to those channels
        if channel_indices:
            for index in channel_indices:
                multichannel_audio[:, index] = audio * (
                    volume_mask[index] if volume_mask else 1
                )
        else:
            # Default to playing mono audio on the first channel or as specified
            multichannel_audio[:, random.randint(0, 8)] = audio

        stream = sounddevice.OutputStream(
            samplerate=samplerate, channels=desired_channels if audio.ndim > 1 else 1
        )
        stream.start()
        stream.write(audio)
        stream.stop()
        stream.close()


def play_sound_hold(audio, samplerate, playback_queue):
    print(audio.shape)
    stream = sounddevice.OutputStream(
        samplerate=samplerate, channels=audio.shape[1] if audio.ndim > 1 else 1
    )
    stream.start()
    stream.write(audio)
    stream.stop()
    stream.close()

    # Remove the finished stream from the queue
    with playback_queue.mutex:
        playback_queue.remove(stream)

    # If there's a new sound to play and the queue is full, stop the oldest one
    if len(playback_queue) >= playback_queue.maxlen:
        oldest_stream = playback_queue.popleft()
        oldest_stream.abort()  # Stop the oldest playback

    # Add the new stream to the queue
    playback_queue.append(stream)


def play_sound_OLD(audio, samplerate):
    pygame.mixer.Sound(array=np.column_stack((audio, audio, audio, audio))).play(
        fade_ms=200
    )
    # sounddevice.play(audio, samplerate)
    # sounddevice.wait()  # Wait until the audio is done playing


def netsend(sd, event_umap_projection, event_init_player):
    print(sd.x)
    client = SimpleUDPClient(cfg.TARGET_IP, cfg.TARGET_PORT)
    prev_indexes = np.zeros(3)
    prev_index = 0
    pygame.mixer.quit()
    pygame.mixer.init(
        frequency=22050,
        size=-16,
        channels=4,
        buffer=512,
        allowedchanges=0,
    )
    sd.logger.info(f"Player mixer details: {pygame.mixer.get_init()}")

    try:
        params = []
        print("letsgo")
        while True:
            event_umap_projection.wait()
            event_init_player.wait()
            if (
                not sd.index_closest_in_corpus is None
                and sd.index_closest_in_corpus != prev_index
            ):
                # index_closest_in_corpus = sd.index_lookup[sd.closest_point_index]
                audio = sd.df_samples.loc[sd.index_closest_in_corpus, "signal"]
                # audio = librosa.resample(audio, orig_sr=22050, target_sr=5512.5)
                # audio_int16 = np.array(np.int16(audio * 32767), dtype=np.int16)

                EXECUTOR.submit(play_sound, audio, 22050, current_playbacks)
                # EXECUTOR.submit(play_sound, audio_int16, 22050)

                # play_sound(sd.df_samples.loc[sd.index_closest_in_corpus, "signal"])
                # sounddevice.play(
                #     sd.df_samples.loc[sd.index_closest_in_corpus, "signal"],
                #     22050,
                #     blocking=False,
                # )
                # sd.sounds[index_closest_in_corpus].play(fade_ms=250)

                prev_index = sd.index_closest_in_corpus
                # prev_indexes = np.roll(prev_indexes, -1)
                # prev_indexes[-1] = sd.index_closest_in_corpus

                # print(prev_indexes, end="\r")

            # time.sleep(1)
            # continue
            # if not None in [
            #     sd.musical_params,
            #     sd.current_zone,
            #     sd.angle_center,
            #     sd.distance_center,
            #     sd.qom,
            #     sd.top,
            # ]:
            # print("SENDING----------")
            if not sd.musical_params is None:
                client.send_message("/params", np.around(sd.musical_params, 3))
            client.send_message(
                "/ctrl",
                [
                    sd.current_zone,
                    sd.angle_center,
                    sd.distance_center,
                    sd.qom,
                    sd.top,
                    sd.dispersion,
                ],
            )
            time.sleep(cfg.NETSEND_DELAY)
    except KeyboardInterrupt:
        sd.logger.info("Stopping...")


def find_similar_in_corpus(sd):
    distances = np.linalg.norm(
        sd.df_corpus.iloc[:, :-2] - sd.df_population.iloc[sd.closest_point_index, :-2],
        axis=1,
    )
    closest_index = np.argmin(distances)
    # closest_match = sd.df_corpus.iloc[closest_index]

    # os.system("cls")
    # print(sd.closest_point_index, closest_index, "------------", "\n")
    # print(sd.df_population.iloc[sd.closest_point_index, :-2], "\n")
    # print(closest_match)

    return closest_index


def back_projection(sd, event_populating, event_updating_knn_data):
    event_populating.wait()

    while True:
        if sd.x and sd.y and sd.z:  # and not sd.flag_populating and not sd.flag_rating:
            event_updating_knn_data.clear()  # moved this from line 73
            event_populating.wait()

            sd.current_zone = calculate_zone(sd)
            sd.angle_center, sd.distance_center = calculate_circular(sd)

            if cfg.USE_SIMULATION:
                # normalize if using the pygame window
                sd.norm_x = sd.x / cfg.WINDOW_X * 2 - 1
                sd.norm_y = sd.y / cfg.WINDOW_Y * 2 - 1
                sd.norm_z = sd.z / cfg.WINDOW_Z * 2 - 1

                sd.norm_y *= -1
                sd.norm_z *= -1
            else:
                # normalize if using mocap
                sd.norm_x = sd.x / cfg.MOCAP_WIDTH_PROJECTION
                sd.norm_y = sd.y / cfg.MOCAP_HEIGHT_PROJECTION * -1
                sd.norm_z = (sd.z / cfg.MOCAP_HEIGHT_Z_PROJECTION * 2 - 1) * -1

            user_point = np.array([[sd.norm_x, sd.norm_y, sd.norm_z]])
            distances, indices = sd.knn.kneighbors(user_point)

            sd.knn_distances = distances
            sd.knn_indices = indices
            # print("back: ", id(sd))
            # print("back: ", id(sd.knn_indices[0]))
            list_knn_indices = indices[0]
            closest_index = list_knn_indices[0]
            sd.closest_point_index = closest_index

            event_updating_knn_data.set()

            event_populating.wait()

            sd.index_closest_in_corpus = sd.index_lookup[sd.closest_point_index]
            selected_params = sd.df_population.loc[closest_index].values[:-2]
            sd.selected_params = selected_params  # moved this here from line 117 !!!!

            # # scale for fm
            # carrier = _scale(selected_params[0], 100, 2000)
            # mod1 = _scale(selected_params[1], 3, 30)
            # index1 = _scale(selected_params[2], 100, 1000)
            # mod2 = _scale(selected_params[3], 3, 15)
            # index2 = _scale(selected_params[4], 50, 600)
            # n1 = _scale(selected_params[5], -1, 1)
            # musical_params = [carrier, mod1, index1, mod2, index2, n1]

            # # scale for additive synth patch (2nd)
            # carrier = _scale(selected_params[0], 100, 1200)
            # ratio = _scale(selected_params[1], 1, 4)
            # metrodev = _scale(selected_params[2], 1, 1000)
            # att = _scale(selected_params[3], 10, 50)
            # sus = _scale(selected_params[4], 20, 150)
            # musical_params = [carrier, ratio, metrodev, att, sus]

            # scale for vocal mocap
            headrot = _scale(selected_params[0], -1, 1)
            neckfold = _scale(selected_params[1], 0, 1)
            handdistlh = _scale(selected_params[2], 0, 1)
            handdistrh = _scale(selected_params[3], 0, 0.6)
            spinefold = _scale(selected_params[4], 0, 1)
            deltime = _scale(selected_params[4], 0, 1000)
            musical_params = [
                headrot,
                neckfold,
                handdistlh,
                handdistrh,
                spinefold,
                deltime,
            ]

            sd.musical_params = musical_params
            sd.qom = calculate_rms(sd) * 100
            if not sd.list_pos is None:
                sd.top = max(sd.list_pos[:, 2]) / 2.2
                distances = np.sqrt(np.sum((sd.list_pos - sd.meanpos) ** 2, axis=1))
                sd.dispersion = np.std(distances)

            time.sleep(0.1)


def normalize_projection_data(df_proj):
    proj_min_x, proj_min_y = df_proj.min(axis=0)
    proj_max_x, proj_max_y = df_proj.max(axis=0)

    df_proj_norm = pd.DataFrame()
    df_proj_norm[0] = (df_proj.iloc[:, 0] - proj_min_x) / (proj_max_x - proj_min_x) * 2
    df_proj_norm[1] = (df_proj.iloc[:, 1] - proj_min_y) / (proj_max_y - proj_min_y) * 2
    df_proj_norm.iloc[:, 0] -= 1
    df_proj_norm.iloc[:, 1] -= 1

    # plt.figure()
    # plt.scatter(df_proj_norm.iloc[:, 0], df_proj_norm.iloc[:, 1], s=1.5)
    # plt.savefig("norm_proj.png")

    return df_proj_norm


def _scale(n, mini, maxi):
    return n * (maxi - mini) + mini


def calculate_zone(sd):
    for i, rect in enumerate(sd.zones):
        x, y, x2, y2 = rect
        if x <= sd.x < x2 and y <= sd.y < y2:
            return i
    return -1


def dist2D(x, y):
    return math.sqrt(x**2 + y**2)


def calculate_circular(sd):
    """Return the angle with x axis and distqance from the center, normalized to 0-1"""
    angle = math.atan2(sd.y, sd.x)
    angle -= math.pi * 1.5  # make 270 new 0
    angle %= 2 * math.pi  # make continuous
    angle /= 2 * math.pi  # normalize 0-1
    distance = dist2D(sd.x, sd.y) / dist2D(
        cfg.MOCAP_HEIGHT_PROJECTION, cfg.MOCAP_WIDTH_PROJECTION
    )
    return angle, distance


def calculate_rms(sd):
    abs_velocity_buffer = np.abs(np.diff(sd.data_buffer, axis=0))
    squared = np.square(abs_velocity_buffer)
    mean_squared = np.mean(squared, axis=0)
    rms3d = np.sqrt(mean_squared)
    return np.mean(rms3d)


def plot_neighbours(sd):
    plt.ion()  # Turn on interactive plotting mode

    df_plot = sd.df_tsne.copy(deep=True)
    df_plot.iloc[:, 1] *= -1

    df_plot.iloc[:, 0] *= cfg.TARGET_WIDTH_PROJECTION
    df_plot.iloc[:, 1] *= cfg.TARGET_HEIGHT_PROJECTION

    # df_plot.iloc[:, 0] -= df_plot.iloc[:, 0].max() / 2
    # df_plot.iloc[:, 1] -= df_plot.iloc[:, 1].max() / 2

    # Initialize the plot outside the loop
    fig, ax = plt.subplots(
        figsize=(cfg.TARGET_WIDTH_PROJECTION, cfg.TARGET_HEIGHT_PROJECTION)
    )

    ax.scatter(df_plot.iloc[:, 0], df_plot.iloc[:, 1], c="r", s=3)
    (scatter_knn,) = ax.plot([], [], "bo", markersize=5, alpha=0.5)
    (scatter_closest,) = ax.plot([], [], "go", markersize=8, alpha=0.5)
    (scatter_user,) = ax.plot([], [], "yo", markersize=5, alpha=1)

    while True:
        if sd.knn_indices is not None:
            list_knn_indices = sd.knn_indices[0]
            list_projected_points = df_plot.loc[list_knn_indices].to_numpy()

            list_knn_distances = sd.knn_distances[0]
            mask_distance = list_knn_distances < cfg.SELECT_DISTANCE_THRESHOLD
            list_projected_points_closer = list_projected_points[mask_distance]

            # Update the scatter plot data for KNN points
            if len(list_projected_points_closer) > 0:
                scatter_knn.set_data(
                    list_projected_points_closer[:, 0],
                    list_projected_points_closer[:, 1],
                )
                scatter_closest.set_data(
                    [list_projected_points_closer[0, 0]],
                    [list_projected_points_closer[0, 1]],
                )
            scatter_user.set_data(
                [sd.norm_x],
                [sd.norm_y] * -1,
            )

            # Necessary to re-draw the plot with updated dataplt.xlim([0, x_max])
            # plt.xlim([0, TARGET_WIDTH_PROJECTION])
            # plt.ylim([0, TARGET_HEIGHT_PROJECTION])
            # plt.axis("off")
            plt.draw()
            plt.pause(0.1)  # Pause to ensure the plot updates visually

            try:
                main_app_window_title = "pygame window"
                main_app_win = gw.getWindowsWithTitle(main_app_window_title)[0]
                main_app_win.activate()
            except:
                sd.logger.error("Couldn't active the window!")

            time.sleep(1)  # Sleep to simulate time between updates

        plt.pause(0.1)  # Add a small pause to keep the plot responsive


def plot_neighbours_static(sd):
    while True:
        if not sd.knn_indices is None:
            list_knn_indices = sd.knn_indices[0]
            list_projected_points = sd.df_tsne.loc[list_knn_indices].to_numpy()
            plt.figure(figsize=(6, 8))
            plt.scatter(sd.df_tsne.iloc[:, 0], sd.df_tsne.iloc[:, 1], c="r", s=3)
            plt.scatter(
                list_projected_points[:, 0],
                list_projected_points[:, 1],
                c="b",
                s=5,
                alpha=0.5,
            )
            plt.show()
            time.sleep(3)


def save_background_image_kmeans_clusters(sd):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(sd.df_tsne.to_numpy())
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    sd.cluster_labels = cluster_labels

    plt.figure(figsize=(cfg.TARGET_WIDTH_PROJECTION, cfg.TARGET_HEIGHT_PROJECTION))
    for i in range(n_clusters):
        plt.scatter(
            sd.df_tsne.to_numpy()[cluster_labels == i, 0],
            sd.df_tsne.to_numpy()[cluster_labels == i, 1],
            color=colors[i],
            label=f"Cluster {i}",
            s=15,
        )
    plt.xlim([0, cfg.TARGET_WIDTH_PROJECTION])
    plt.ylim([0, cfg.TARGET_HEIGHT_PROJECTION])
    plt.axis("off")
    plt.savefig("plot-tsne.png", bbox_inches="tight", dpi=200)


def save_background_image(sd):
    # plot and save the tsne projection
    plt.figure(figsize=(cfg.TARGET_WIDTH_PROJECTION, cfg.TARGET_HEIGHT_PROJECTION))
    plt.scatter(
        sd.df_tsne.iloc[:, 0],
        sd.df_tsne.iloc[:, 1],
        c="r",
        s=6,
    )
    plt.xlim([0, cfg.TARGET_WIDTH_PROJECTION])
    plt.ylim([0, cfg.TARGET_HEIGHT_PROJECTION])
    plt.axis("off")
    plt.savefig("plot-tsne.png", bbox_inches="tight", dpi=200)


def silhuette_score(sd):
    score = silhouette_score(sd.df_tsne, sd.cluster_labels)
    sd.logger.info(f"Silhouette Score: {score}")


def get_zones():
    # draw zones 2D
    width = cfg.MOCAP_WIDTH_PROJECTION * 2 / 3
    height = cfg.MOCAP_HEIGHT_PROJECTION * 2 / 2
    xs = np.arange(-cfg.MOCAP_WIDTH_PROJECTION, cfg.MOCAP_WIDTH_PROJECTION, width)
    ys = np.arange(-cfg.MOCAP_HEIGHT_PROJECTION, cfg.MOCAP_HEIGHT_PROJECTION, height)
    rects = []

    ## get colors
    num_rects = len(xs) * len(ys)
    # cmap = plt.get_cmap("rainbow")
    # colors = [cmap(i / num_rects) for i in range(num_rects)]
    # colors = [
    #     (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors
    # ]

    ## calculate points and draw
    for x in xs:
        for y in ys:
            rects.append(((x, y, x + width, y + height)))

    return rects


def create_logger(timestamp):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s  %(message)s")

    # Create a file handler - no need for now
    # log_filename = timestamp + ".log"
    # file_handler = logging.FileHandler(
    #     os.path.join("log", log_filename), encoding="utf-8"
    # )
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # Create a console (stream) handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def start_player(sd, event_umap_projection, event_init_player):
    prev_indexes = np.zeros(3)
    while True:
        # continue
        # event_umap_projection.wait()
        # event_init_player.wait()
        if (
            not sd.index_closest_in_corpus is None
            and sd.index_closest_in_corpus not in prev_indexes
        ):
            print("as")
            # index_closest_in_corpus = sd.index_lookup[sd.closest_point_index]
            # sounddevice.play(
            #     sd.df_samples.loc[sd.index_closest_in_corpus, "signal"],
            #     22050,
            #     blocking=False,
            # )
            # sd.sounds[index_closest_in_corpus].play(fade_ms=250)
            # prev_indexes = np.roll(prev_indexes, -1)
            # prev_indexes[-1] = sd.index_closest_in_corpus
            # print(prev_indexes, end="\r")
            time.sleep(0.1)


def main():
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = create_logger(start_timestamp)
    logger.info("Starting...")

    sd = SharedData()

    sd.logger = logger

    # pop = Population()
    corpus = CorpusReader(sd)
    corpus.prepare(source="checkpoint", save=False)
    df_corpus, df_population = corpus.get_as_population()
    sd.df_samples = corpus.df_samples

    operator = Operator(sd)
    # operator.init_populate()
    operator.init_from_corpus(df_corpus, df_population)

    # save_background_image_kmeans_clusters(sd)
    # save_background_image(sd)
    # silhuette_score(sd)

    zones = get_zones()
    sd.zones = zones

    graphics = Graphics()
    # graphics = Graphics3D()
    streamer = Streamer()

    event_populating = threading.Event()
    event_updating_knn_data = threading.Event()
    event_updating_knn_data.set()
    event_umap_projection = threading.Event()
    event_init_player = threading.Event()

    t_operator = threading.Thread(
        target=operator.operate,
        args=(event_populating, event_umap_projection),
    )
    t_graphics = threading.Thread(
        target=graphics.start_graphics,
        args=(
            sd,
            operator,
            event_populating,
            event_updating_knn_data,
            event_init_player,
            event_umap_projection,
        ),
    )
    t_backprojection = threading.Thread(
        target=back_projection,
        args=(sd, event_populating, event_updating_knn_data),
    )
    t_streamer = threading.Thread(
        target=streamer.start_data_acquisition,
        args=(sd,),
    )
    t_player = threading.Thread(
        target=start_player,
        args=(sd, event_umap_projection, event_init_player),
    )
    t_netsend = threading.Thread(
        target=netsend,
        args=(sd, event_umap_projection, event_init_player),
    )

    # start threads
    t_operator.start()
    t_graphics.start()
    t_backprojection.start()
    if not cfg.USE_SIMULATION:
        t_streamer.start()
    # t_player.start()
    t_netsend.start()

    # join threads
    t_operator.join()
    t_graphics.join()
    t_backprojection.join()
    if not cfg.USE_SIMULATION:
        t_streamer.join()
    # t_player.join()
    t_netsend.join()

    sd.logger.info("Initialization done!")

    # plot_neighbours(sd)
    # plot_kmeans_clusters(sd)


if __name__ == "__main__":
    main()
