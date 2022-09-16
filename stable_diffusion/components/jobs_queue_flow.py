from typing import Optional

import streamlit as st
from lightning import LightningFlow
from lightning.app.frontend.stream_lit import StreamlitFrontend

from stable_diffusion.db import DatabaseConnector
from stable_diffusion.db.models import JobsQueue


class JobsQueueFlow(LightningFlow):

    model = JobsQueue

    def __init__(self):
        super().__init__()
        self.db_url = None
        self._database = None
        self.queue_values: list = []

    def run(self, db_url: str):
        self.db_url = db_url

        queue_configs = self.db.get()
        self.queue_values = [
            (queue_config.prompts, queue_config.styles, queue_config.num_images)
            for queue_config in queue_configs
        ]

    @property
    def db(self) -> DatabaseConnector:
        if self._database is None:
            assert self.db_url is not None
            self._database = DatabaseConnector(self.model, self.db_url + "/general/")
        return self._database

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_jobs_queue)


def render_jobs_queue(state):

    st.title("Jobs Queue :racing_car: :checkered_flag:")

    for prompts, styles, num_images in state.queue_values:

        st.text(
            "Prompts: "
            + str(prompts)
            + ", Styles: "
            + str(styles)
            + ", Num Images Each: "
            + str(num_images)
        )
        st.success(
            "Images Generated for the above prompts in 75 seconds :white_check_mark:"
        )
