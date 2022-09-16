import time
import uuid
from typing import Optional

import streamlit as st
from lightning import LightningFlow
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.structures import Dict

from stable_diffusion.components.stable_diffusion_job import StableDiffusionJob
from stable_diffusion.db import DatabaseConnector
from stable_diffusion.db.models import JobsQueue, PromptConfig
from stable_diffusion.utilities.enum import Stage
from stable_diffusion.utilities.utils import TIME_FORMAT


class StableDiffusionFlow(LightningFlow):

    prompt_model = PromptConfig
    queue_model = JobsQueue

    def __init__(self):
        super().__init__()
        self.db_url = None
        self.parallel_jobs: int = 5
        self.jobs = Dict()
        self.jobs_sack: list = []
        self._database = None
        self.ready = False
        self.prompts: Optional[list] = None
        self.styles: Optional[list] = None
        self.prompt_config: Optional[list] = None
        self.num_images: Optional[int] = None

    def run(self, db_url: str):
        self.db_url = db_url

        if self.prompt_config is not None and self.num_images is not None:
            if not self.ready:
                for idx in range(self.parallel_jobs):
                    print(f"Creating job {idx}")
                    self.jobs[str(idx)] = StableDiffusionJob(self.db, str(idx))
                self.ready = True

            # Add to the JobsQueue Table
            queue_uuid = str(uuid.uuid4())
            started_time = time.strftime(TIME_FORMAT)
            jobs_queue_config = JobsQueue(
                queue_id=queue_uuid,
                prompts=self.prompts,
                styles=self.styles,
                num_images=self.num_images,
                started_time=started_time,
            )
            self.db.post(jobs_queue_config)

            print(jobs_queue_config)

            for prompt, style in self.prompt_config:
                self.db.post(
                    PromptConfig(
                        queue_id=queue_uuid,
                        prompt=prompt,
                        style=style,
                        num_images=self.num_images,
                    )
                )
            self.prompt_config = None
            self.num_images = None

        db_configs = self.db.get(self.prompt_model)
        not_started_configs = [
            config for config in db_configs if config.stage == Stage.NOT_STARTED
        ]
        if len(not_started_configs) > 0 and len(self.jobs) > 0:
            print(not_started_configs)
            # TODO: implement job assignment
            self.jobs_sack = [
                [idx, job.num_images_to_generate]
                for idx, job in self.jobs.items()
                if not job.is_running
            ]

            def split_tasks(tasks, num_jobs):
                k, m = divmod(len(tasks), num_jobs)
                return (
                    tasks[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                    for i in range(num_jobs)
                )

            assigned_tasks = split_tasks(not_started_configs, len(self.jobs.keys()))
            print("Assigned Tasks", assigned_tasks)

            for idx, config_list in enumerate(assigned_tasks):
                if len(config_list) == 0:
                    continue
                print(f"Assigning {config_list} tasks to job {idx}")
                for config in config_list:
                    config.stage = Stage.RUNNING
                    config.job_id = str(idx)
                    self.db.put(config)
                    if not self.jobs[str(idx)].is_running:
                        self.jobs[str(idx)].run()

            print("Not started configs found")
            print(self.jobs_sack)

    @property
    def db(self) -> DatabaseConnector:
        if self._database is None:
            assert self.db_url is not None
            self._database = DatabaseConnector(
                self.prompt_model, self.db_url + "/general/"
            )
        return self._database

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_diffusion_flow)


def render_diffusion_flow(state):

    st.title("Stable Diffusion Studio App! :frame_with_picture: :zap:")

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

        state.prompts = prompts
        state.styles = styles
        state.prompt_config = [
            (prompt, style) for prompt in prompts for style in styles
        ]
        state.num_images = num_images
