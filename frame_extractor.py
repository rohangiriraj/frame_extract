import os
import subprocess
import glob
import numpy as np
import pandas as pd
import av
from tqdm import tqdm
from pathlib import Path


def generateweights(frame: np.ndarray) -> np.ndarray:
    ymax, xmax = frame.shape
    normfactor = ymax // 2
    y = np.arange(-ymax // 2, ymax - ymax // 2) / normfactor
    x = np.arange(-xmax * 2 // 3, xmax - xmax * 2 // 3) / normfactor
    xx, yy = np.meshgrid(x, y)
    weights = np.exp(-(xx ** 2 + yy ** 2))
    weights[:, : xmax // 3] = 0
    # print(f"The shape of the weights are: {weights.shape}")
    # Build the weights, please.
    weights = np.delete(weights, 640, 1)
    # print(len(weights[:, [0]]))
    return weights


def penalty_score(prev, this, weight) -> np.ndarray:

    this = np.square(this > 128)
    prev = np.square(prev > 128)
    # print(f"Shape of this: {this.shape}")
    # print(f"Shape of prev: {prev.shape}")
    return np.dot(np.square(this - prev), weight.T)


def conversion(path):
    # name = Path(path).name
    # print(f"The name of the file is {name}")
    MOVIE = path
    name_len = len(path)
    next_t = 2
    incr = 2
    prev = None
    weights = None
    threshold = 1e-3

    files = []
    container = av.open(path)
    container.seek(0)
    for frame in tqdm(container.decode(video=0)):
        if frame.is_corrupt or frame.time < next_t:
            continue
        next_t += incr
        this = frame.to_ndarray(format="gray")

        if prev is None:
            weights = generateweights(this)
            prev = np.zeros_like(this)
            continue

        score = np.mean(penalty_score(prev, this, weights))
        prev = this

        if score > threshold:
            filename = "screen-{:02.0f}m{:.0f}s.jpg".format(
                frame.time // 60, frame.time % 60
            )
            files.append(filename)
            frame.to_image().save(filename)
    subprocess.run(["convert"] + files + [path[: name_len - 4] + ".pdf"])
    print('Done')
    for f in files:
        os.unlink(f)


def main():
    video_path = input(
        "Enter the path to your video, along with the name and format of the video.\n"
    )
    if os.path.exists(video_path):
        conversion(video_path)
    else:
        print(
            "Such a specified path does not exist. Please re-run the program and enter the right path."
        )


if __name__ == "__main__":
    main()
