"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.a_size = new_size

    def process_frame_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        _img = Image.fromarray(state) # from ndarray to image
        _img = _img.convert('L') # to greyscale
        _img = _img.resize(self.a_size) # scale
        return np.asarray(_img, dtype=np.uint8) # from image to ndarray

    # def process_reward(self, reward):
    #     """Clip reward between -1 and 1."""
    #     if reward > 0:
    #         return 1
    #     elif reward < 0:
    #         return -1
    #     else:
    #         return 0

    # def process_frame_for_network(self, state):
    #     """Scale, convert to greyscale and store as float32.

    #     Basically same as process state for memory, but this time
    #     outputs float32 images.
    #     """
    #     _img = Image.fromarray(state) # from ndarray to image
    #     _img = _img.convert('L') # to greyscale
    #     _img = _img.resize(self.a_size) # scale
    #     return np.asarray(_img, dtype=np.float32) # from image to ndarray


