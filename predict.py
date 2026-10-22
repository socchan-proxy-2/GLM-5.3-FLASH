import time
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        for i in range(10):
            print(f"setup alive {i}", flush=True)
            time.sleep(3)

    def predict(self, prompt: str = Input(default="hi")) -> str:
        return "ok"
