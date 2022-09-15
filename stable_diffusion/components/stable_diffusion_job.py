import os
import tarfile
import urllib

from lightning import CloudCompute, LightningWork

from stable_diffusion.utilities.enum import Stage


class StableDiffusionJob(LightningWork):
    def __init__(self, db, job_id: str):
        super().__init__(cloud_compute=CloudCompute("cpu"), parallel=True)
        self.job_id = job_id
        self._db = db
        self.num_images_to_generate: int = 0
        self._model = None

    def run(self):
        if self._model is None:
            # self._model = self.build_model()
            print("Building model...")
            self._model = 2

        db_configs = self._db.get()
        job_configs = [
            config
            for config in db_configs
            if config.stage == Stage.RUNNING and config.job_id == self.job_id
        ]

        print(job_configs)
        for config in job_configs:
            config.stage = Stage.SUCCEEDED
            self._db.put(config)

    @staticmethod
    def download_weights(url: str, target_folder: str):
        dest = target_folder + f"/{os.path.basename(url)}"
        urllib.request.urlretrieve(url, dest)
        file = tarfile.open(dest)

        # extracting file
        file.extractall(target_folder)

    def build_model(self):
        """The `build_model(...)` method returns a model and the returned model is set to `self._model` state."""

        import os

        import torch
        from diffusers import StableDiffusionPipeline

        print("loading model...")
        if torch.cuda.is_available():
            weights_folder = "resources/stable-diffusion-v1-4"
            os.makedirs(weights_folder, exist_ok=True)

            print("Downloading weights...")
            self.download_weights(
                "https://lightning-dream-app-assets.s3.amazonaws.com/diffusers.tar.gz",
                weights_folder,
            )

            repo_folder = f"{weights_folder}/Users/pritam/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096"
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_folder,
                revision="fp16",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")
            print("model loaded")
        else:
            pipe = None
            print("model set to None")
        return pipe
