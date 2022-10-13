import numpy as np
import time

class generator:

    def __init__(self):
        self.H = 64
        self.W = 64
        self.prompt = None
        self.loaded = False

    def load_model(self):
        print('Loading model...')
        time.sleep(3)
        self.loaded = True
        print('Model loaded')

    def generate(self, w, h, prompt):
        random_img = np.random.randint(256, size=(h, w, 3)).astype(np.uint8)
        time.sleep(3)
        return random_img
