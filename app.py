from lightning import LightningApp, LightningFlow


class StableDiffusion(LightningFlow):
    def run(self):
        print("Running StableDiffusion")


app = LightningApp(StableDiffusion())
