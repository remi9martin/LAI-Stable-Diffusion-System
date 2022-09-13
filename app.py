import os

from lightning import LightningApp, LightningFlow
from lightning.app.storage import Drive

from stable_diffusion import StableDiffusionFlow
from stable_diffusion.db import Database


class MainFlow(LightningFlow):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug or os.getenv("PL_DEBUG", False)

        # 1: Create Drive
        self.drive = Drive("lit://stable-diffusion")

        # 2: Stable Diffusion Flow for prompts and styles
        self.dream = StableDiffusionFlow()

        # 2: Controller
        # self.jobs_controller = JobsController(self.drive)

        # 3: Create the File Server to upload code or data.
        # self.file_server = FileServer(self.drive)

        # 4: Create the database.
        self.db = Database(models=[self.dream.model])

        self.ready = False

    def run(self):
        self.db.run()

        if not self.db.alive():
            return

        if not self.ready:
            print(
                f"The Stable Diffusion System is ready ! Database URL: {self.db.db_url}"
            )
            self.ready = True

        self.dream.run(self.db.db_url)

    def configure_layout(self):
        tab_1 = {"name": "Dream", "content": self.dream}
        # tab_1 = {"name": "Visualize", "content": "Tab 2 content"}
        # tab_1 = {"name": "Jobs Queue", "content": "Tab 3 content"}
        return [tab_1]


app = LightningApp(MainFlow())
