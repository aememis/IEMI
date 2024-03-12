import random
import pandas as pd
import numpy as np
import os
import glob
import librosa
import pickle

import config as cfg
from typing import Literal
from pydub import AudioSegment


class CorpusReader:
    def __init__(self, sd):
        self.sd = sd
        self.df_corpus = None
        self.df_features = None
        self.df_features_norm = None
        self.window_length = 1024
        self.hop_length = 512

    def read_corpus_from_files(self):
        self.sd.logger.info("Loading the dataset...")
        list_paths = glob.glob(cfg.DATASET_PATH_FSD50K)
        list_paths = random.sample(list_paths, cfg.CORPUS_SIZE)  ####
        len_paths = len(list_paths)
        list_audio = []
        list_notes = []
        for i, file in enumerate(list_paths[:-1]):
            print(f"Loading {str(i).zfill(5)}/{len_paths}", end="\r")

            y, sr = librosa.load(file)
            len_y = len(y)
            if len_y < cfg.SAMPLES_THRESHOLD_LOW or len_y > cfg.SAMPLES_THRESHOLD_HIGH:
                continue
            y /= np.max(np.abs(y))

            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, win_length=1024, hop_length=512
            )[0]
            flux = librosa.onset.onset_strength(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, n_mfcc=1, win_length=1024, hop_length=512)[
                0
            ]
            sc = librosa.feature.spectral_centroid(
                y=y, win_length=1024, hop_length=512
            )[0]
            sf = librosa.feature.spectral_flatness(
                y=y, win_length=1024, hop_length=512
            )[0]

            # notes_ = [
            #     os.path.basename(file),
            #     sr,
            #     len_y,
            #     rms,
            #     spectral_bandwidth,
            #     flux,
            #     mfcc,
            #     sc,
            #     sf,
            # ]
            notes = [
                os.path.basename(file),
                sr,
                len_y,
                np.mean(rms),
                np.mean(spectral_bandwidth),
                np.mean(flux),
                np.mean(mfcc),
                np.mean(sc),
                np.mean(sf),
            ]
            audio = [
                y,
            ]

            list_notes.append(notes)
            list_audio.append(audio)
        self.sd.logger.info(f"Loaded {len(list_audio)} files.")

        self.df_features = pd.DataFrame(
            list_notes,
            columns=[
                "name",
                "sr",
                "length",
                "rms",
                "spectral_bandwidth",
                "flux",
                "mfcc",
                "sc",
                "sf",
            ],
        ).reset_index(drop=True)
        self.df_samples = pd.DataFrame(
            list_audio,
            columns=[
                "signal",
            ],
        ).reset_index(drop=True)
        print(self.df_features.describe())

        """
        # # Separate the harmonic and percussive components of the audio
        # y_harmonic, y_percussive = librosa.effects.hpss(y)
        # harmonic_energy = librosa.feature.rms(y=y_harmonic)[0]
        # percussive_energy = librosa.feature.rms(y=y_percussive)[0]
        # harmonic_percentile_energy = np.percentile(harmonic_energy, 50)
        # percussive_percentile_energy = np.percentile(percussive_energy, 50)
        # if percussive_percentile_energy > 0:
        #     harmonic_perc = harmonic_percentile_energy / percussive_percentile_energy
        # else:
        #     harmonic_perc = -1
        
        ftr["rms"] = librosa.feature.rms(y=y, frame_length=win_l, hop_length=hop_l)[0]
        ftr["sc"] = librosa.feature.spectral_centroid(y=y, win_length=512, hop_length=hop_l)[0]
        ftr["sb"] = librosa.feature.spectral_bandwidth(y=y, win_length=512, hop_length=hop_l)[0]
        ftr["sr"] = librosa.feature.spectral_flatness(y=y, win_length=512, hop_length=hop_l)[0]
        ftr["sf"] = librosa.feature.spectral_rolloff(y=y, win_length=512, hop_length=hop_l)[0]
        ftr["mfcc"] = librosa.feature.mfcc(y=y, n_mfcc=1, win_length=win_l, hop_length=hop_l)[0]
        """

    def read_corpus_from_saved(self):
        with open("features.pkl", "rb") as f:
            self.df_features_norm = pickle.load(f)
        with open("samples.pkl", "rb") as f:
            self.df_samples = pickle.load(f)

    def save_corpus(self):
        self.sd.logger.info("Saving corpus...")
        with open("features_raw.pkl", "wb") as file:
            pickle.dump(self.df_features, file)
        with open("features.pkl", "wb") as file:
            pickle.dump(self.df_features_norm, file)
        with open("samples.pkl", "wb") as file:
            pickle.dump(self.df_samples, file)
        self.sd.logger.info("Corpus saved.")

    def normalize_corpus(self):
        self.df_features_norm = self.df_features.copy(deep=True)
        for col in [
            "length",
            "rms",
            "spectral_bandwidth",
            "flux",
            "mfcc",
            "sc",
            "sf",
        ]:
            min_val = self.df_features[col].min()
            max_val = self.df_features[col].max()
            self.df_features_norm[col] = (self.df_features[col] - min_val) / (
                max_val - min_val
            )

    def prepare(self, source: Literal["files", "checkpoint"], save=False):
        if source == "checkpoint":
            self.read_corpus_from_saved()
        elif source == "files":
            self.read_corpus_from_files()
            self.normalize_corpus()
        else:
            raise Exception(f"{source} is not excepted as a source")
        if save:
            self.save_corpus()

    def get_as_population(self, population_size=cfg.POPULATION_SIZE):
        # keep: ["length","rms","spectral_bandwidth","flux","mfcc","sc","sf",]
        df_corpus = (
            self.df_features_norm.drop(["name", "sr"], axis="columns")
            .assign(score=-1)
            .assign(pop=None)
            .loc[:, cfg.DATA_FIELDS_CORPUS_LABEL]
            .set_axis(cfg.DATA_FIELDS_CORPUS, axis=1)
        )
        print(df_corpus)
        print(cfg.DATA_FIELDS_CORPUS)
        df_population = df_corpus.sample(population_size, random_state=42).reset_index(
            drop=True
        )
        print(df_population)
        return df_corpus, df_population
