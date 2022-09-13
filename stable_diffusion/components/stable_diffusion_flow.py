from typing import Optional

import streamlit as st
from lightning import LightningFlow
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.structures import Dict

from stable_diffusion.db import DatabaseConnector
from stable_diffusion.db.models import PromptConfig
from stable_diffusion.utilities.enum import Stage


class StableDiffusionFlow(LightningFlow):

    model = PromptConfig

    def __init__(self):
        super().__init__()
        self.db_url = None
        self.jobs = Dict()
        self._database = None
        self.ready = False
        self.prompt_config = None
        self.prompts: list = []
        self.styles: list = []
        self.num_images: Optional[int] = None

    def run(self, db_url: str):
        self.db_url = db_url

        if self.prompt_config is not None and self.num_images is not None:
            for prompt, style in self.prompt_config:
                config = PromptConfig(
                    prompt=prompt, style=style, num_images=self.num_images
                )
                self.db.post(config)
            self.prompt_config = None
            self.num_images = None

        db_configs = self.db.get(self.model)
        if db_configs:
            print("DB configs:", db_configs)

    @property
    def db(self) -> DatabaseConnector:
        if self._database is None:
            assert self.db_url is not None
            self._database = DatabaseConnector(self.model, self.db_url + "/general/")
        return self._database

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_diffusion_flow)


def render_diffusion_flow(state):

    st.title("Welcome to Stable Diffusion System Demo! :rocket:")

    prompts = st.text_input("Enter your prompt here", value="Alan Turing")

    styles = st.multiselect(
        "Select the styles you want to use", ["Pablo Picasso", "Banksy", "Van Gogh"]
    )

    num_images = st.number_input(
        "How many images do you want to generate per (prompt, style)?",
        min_value=1,
        max_value=10,
        value=5,
    )

    dream = st.button("Dream", disabled=not bool(prompts and styles))

    if dream:
        prompts = prompts.split(",")

        state.prompt_config = [
            (prompt, style) for prompt in prompts for style in styles
        ]
        state.num_images = num_images
