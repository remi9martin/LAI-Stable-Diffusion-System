from lightning import CloudCompute, LightningWork


class StableDiffusionJob(LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=CloudCompute("cpu"), parallel=True)
        self.num_images_to_generate: int = 0

    def run(self, configs: list):
        print(configs)
