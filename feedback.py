import chainlit.data as cl_data
import chainlit as cl
from langsmith import traceable, Client
import uuid


class CustomDataLayer(cl_data.BaseDataLayer):
    async def upsert_feedback(self, feedback: cl_data.base.Feedback) -> str:
        client = Client()
        run_id = uuid.uuid4()
        cl.message(f"Creating feedback for run_id: {run_id} \n{feedback}")

        client.create_feedback(
            run_id,
            key="correction",
            score=feedback.value,
            comment=feedback.comment,
        )


        return await super().upsert_feedback(feedback)

    async def build_debug_url(self, *args, **kwargs):
        pass

    async def create_element(self, *args, **kwargs):
        pass

    async def create_step(self, *args, **kwargs):
        pass

    async def create_user(self, *args, **kwargs):
        pass

    async def delete_element(self, *args, **kwargs):
        pass

    async def delete_feedback(self, *args, **kwargs):
        pass

    async def delete_step(self, *args, **kwargs):
        pass

    async def delete_thread(self, *args, **kwargs):
        pass

    async def get_element(self, *args, **kwargs):
        pass

    async def get_thread(self, *args, **kwargs):
        pass

    async def get_thread_author(self, *args, **kwargs):
        pass

    async def get_user(self, *args, **kwargs):
        pass

    async def list_threads(self, *args, **kwargs):
        pass

    async def update_step(self, *args, **kwargs):
        pass

    async def update_thread(self, *args, **kwargs):
        pass

