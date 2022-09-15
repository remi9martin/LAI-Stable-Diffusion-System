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

    def run(self, db_url: str):
        self.db_url = db_url

    @property
    def db(self) -> DatabaseConnector:
        if self._database is None:
            assert self.db_url is not None
            self._database = DatabaseConnector(self.model, self.db_url + "/general/")
        return self._database

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_jobs_queue)


def render_jobs_queue(state):

    st.title("Jobs Queue! :rocket:")
