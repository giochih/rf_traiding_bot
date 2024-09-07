import numpy as np
import pandas as pd
import os
from itertools import combinations
import random
import gc
import torch
from warnings import simplefilter


class Dataset:
    def __init__(
        self, device="cpu", data_folder=None, transformed_data_folder=None
    ):
        self.data_folder = data_folder
        self.transformed_data_folder = transformed_data_folder
        self.train_files = None
        self.val_files = None
        self.device = device

    @staticmethod
    def transform(input_dataframe):
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        df = input_dataframe.copy()
        columns_to_remove = df.columns.drop(["OPEN"])
        for i in range(1, 11):
            df["DATE_SHIFT_" + str(i)] = df["DATE"].shift(i)
            df["SAME_DATE_SHIFT_" + str(i)] = (
                (pd.to_datetime(df["DATE_SHIFT_" + str(i)])).dt.day
                == (pd.to_datetime(df["DATE"])).dt.day
            ).astype(int)
            df = df.drop(columns="DATE_SHIFT_" + str(i))

            df["OPEN_SHIFT_" + str(i)] = df["OPEN"].shift(i)
            df["CLOSE_SHIFT_" + str(i)] = df["CLOSE"].shift(i)
            df["HIGH_SHIFT_" + str(i)] = df["HIGH"].shift(i)
            df["LOW_SHIFT_" + str(i)] = df["LOW"].shift(i)

            df["TOP_TAIL_SHIFT_" + str(i)] = (
                df["HIGH_SHIFT_" + str(i)] - df[["OPEN", "CLOSE"]].max(axis=1)
            ) / (df["HIGH_SHIFT_" + str(i)] - df["LOW_SHIFT_" + str(i)])
            df["BOT_TAIL_SHIFT_" + str(i)] = (
                df[["OPEN", "CLOSE"]].min(axis=1) - df["LOW_SHIFT_" + str(i)]
            ) / (df["HIGH_SHIFT_" + str(i)] - df["LOW_SHIFT_" + str(i)])
            df["BODY_SHIFT_" + str(i)] = (
                df["OPEN_SHIFT_" + str(i)] - df["CLOSE_SHIFT_" + str(i)]
            ) / (df["HIGH_SHIFT_" + str(i)] - df["LOW_SHIFT_" + str(i)])

            df["HIGH_DIV_LOW_SHIFT" + str(i)] = (df["HIGH_SHIFT_" + str(i)]) / (
                df["LOW_SHIFT_" + str(i)]
            )
            df["OPEN_DIV_LOW_SHIFT" + str(i)] = (df["OPEN_SHIFT_" + str(i)]) / (
                df["LOW_SHIFT_" + str(i)]
            )
            df["CLOSE_DIV_LOW_SHIFT" + str(i)] = (
                df["CLOSE_SHIFT_" + str(i)]
            ) / (df["LOW_SHIFT_" + str(i)])
            df = df.drop(columns="OPEN_SHIFT_" + str(i))
            df = df.drop(columns="CLOSE_SHIFT_" + str(i))
            df = df.drop(columns="HIGH_SHIFT_" + str(i))
            df = df.drop(columns="LOW_SHIFT_" + str(i))

        open_avg = df["OPEN"].rolling(window=5).mean()
        close_avg = df["CLOSE"].rolling(window=5).mean()
        high_avg = df["HIGH"].rolling(window=5).mean()
        low_avg = df["LOW"].rolling(window=5).mean()

        for i in range(2, 42):
            df["OPEN_DIV_SHIFT_" + str(i)] = df["OPEN"].shift(1) / df[
                "OPEN"
            ].shift(i)
            df["CLOSE_DIV_SHIFT_" + str(i)] = df["CLOSE"].shift(1) / df[
                "CLOSE"
            ].shift(i)
            df["HIGH_DIV_SHIFT_" + str(i)] = df["HIGH"].shift(1) / df[
                "HIGH"
            ].shift(i)
            df["LOW_DIV_SHIFT_" + str(i)] = df["LOW"].shift(1) / df[
                "LOW"
            ].shift(i)

        for i in range(2, 84, 4):
            df["OPEN_DIV_WINDOW_SHIFT_" + str(i)] = df["OPEN"].shift(
                1
            ) / open_avg.shift(i)
            df["CLOSE_DIV_WINDOW_SHIFT_" + str(i)] = df["CLOSE"].shift(
                1
            ) / close_avg.shift(i)
            df["HIGH_DIV_WINDOW_SHIFT_" + str(i)] = df["HIGH"].shift(
                1
            ) / high_avg.shift(i)
            df["LOW_DIV_WINDOW_SHIFT_" + str(i)] = df["LOW"].shift(
                1
            ) / low_avg.shift(i)

        df = df.drop(columns=columns_to_remove)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        return df[84:].copy()

    def transform_all_data(self):
        if not os.path.isdir(self.data_folder):
            raise ValueError("data_folder folder does not exist")
        if not os.path.isdir(self.transformed_data_folder):
            raise ValueError("transformed_data_folder folder does not exist")
        for root, dirs, files in os.walk(self.data_folder):
            n = len(files)
            i = 1
            for file in files:
                if ".csv" not in file:
                    continue
                df = pd.read_csv(self.data_folder + "\\" + file)
                print(
                    f"\rTransforming file {i}\tout of {n}", end="", flush=True
                )
                df = self.transform(df)
                df.to_csv(
                    self.transformed_data_folder + "\\" + file, index=False
                )
                i += 1

    def split_files_train_val(self, train_part=0.8):
        if not os.path.isdir(self.transformed_data_folder):
            raise ValueError("transformed_data_folder folder does not exist")
        files_and_sizes = []
        total_size = 0
        for root, dirs, files in os.walk(self.transformed_data_folder):
            for file in files:
                if ".csv" not in file:
                    continue
                size = os.path.getsize(
                    self.transformed_data_folder + "\\" + file
                )
                total_size += size
                files_and_sizes.append((file, size))

        if len(files_and_sizes) <= 1:
            raise ValueError("not enough files in transformed_data_folder")

        target_train_size = total_size * train_part
        closest_sum = float("inf")
        closest_subset = []
        for i in range(len(files_and_sizes), 0, -1):
            all_less = True
            print(
                f"\rTrying subset size {i}\tClosest = {closest_sum / total_size:.2f}",
                end="",
                flush=True,
            )
            for subset in combinations(files_and_sizes, i):
                current_size = sum([el[1] for el in subset])
                if abs(target_train_size - current_size) < abs(
                    target_train_size - closest_sum
                ):
                    closest_sum = current_size
                    closest_subset = subset
                if current_size > target_train_size:
                    all_less = False
            if abs(closest_sum / total_size - train_part) <= 0.03:
                break
            if all_less:
                break
        if (
            closest_sum == total_size
            or closest_sum == 0
            or closest_sum == float("inf")
        ):
            raise ValueError(
                "something went wrong. try choosing train_files and val_files manually"
            )
        self.train_files = []
        self.val_files = []

        self.train_files = [el[0] for el in closest_subset]
        for el in files_and_sizes:
            if el[0] not in self.train_files:
                self.val_files.append(el[0])
        print(
            f"\rBest split = {closest_sum/total_size:.2f}", end="", flush=True
        )
        # return best we can do
        return closest_sum / total_size

    def train_batches_generator(self, batch_size=10_000):
        if self.train_files is None:
            raise ValueError(
                "train_files not given. run self.split_files_train_val or fill manually"
            )

        shuffled_files = self.train_files.copy()
        random.shuffle(shuffled_files)

        for file in shuffled_files:
            df = pd.read_csv(self.transformed_data_folder + "\\" + file)
            open_prices = df["OPEN"].values.astype("float32")

            df = df.drop(columns=["OPEN"])
            torch_df = torch.tensor(df.values).to(torch.float32)

            n = len(torch_df)
            i = 0
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            gc.collect()
            while (i + 1) * batch_size <= n:
                yield torch_df[i * batch_size : (i + 1) * batch_size].to(
                    self.device
                ), open_prices[i * batch_size : (i + 1) * batch_size]
                i += 1

    def train_batches_count(self, batch_size=10_000):

        if self.train_files is None:
            raise ValueError(
                "train_files not given. run self.split_files_train_val or fill manually"
            )

        total = 0
        i = 1
        files_count = len(self.train_files)
        for file in self.train_files:
            with open(self.transformed_data_folder + "\\" + file) as f:
                row_count = sum(1 for _ in f)
            print(
                f"\rCounting batches: file№ {i}\tout of {files_count}\t {i*100/files_count:.2f}%",
                end="",
                flush=True,
            )
            total += row_count // batch_size
            i += 1
        return total

    def val_batches_generator(self):
        if self.val_files is None:
            raise ValueError(
                "val_files not given. run self.split_files_train_val or fill manually"
            )

        shuffled_files = self.val_files.copy()
        random.shuffle(shuffled_files)

        for file in shuffled_files:
            df = pd.read_csv(self.transformed_data_folder + "\\" + file)
            open_prices = df["OPEN"].values.astype("float32")

            df = df.drop(columns=["OPEN"])
            torch_df = torch.tensor(df.values).to(torch.float32).to(self.device)

            if "cuda" in self.device:
                torch.cuda.empty_cache()
            gc.collect()
            yield torch_df, open_prices

    def val_batches_count(self):
        if self.val_files is None:
            raise ValueError(
                "val_files not given. run self.split_files_train_val or fill manually"
            )

        return len(self.val_files)
