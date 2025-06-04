import asyncio
import base64
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, get_job_context
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.llm import ImageContent

import os
import random
import imghdr

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        self._tasks = []
        super().__init__(
            instructions="Your name is Dr.San and you are a friendly and knowledgeable personal doctor AI. Your role is to have natural, human-like conversations with users who come to you with health concerns. When someone says something like “I feel sick” or “I have nausea,” your first instinct is to ask gentle, relevant follow-up questions to understand their symptoms better. You never rush to give advice — instead, you engage in a back-and-forth that helps the user feel heard, supported, and respected."
            ""
            "Your main goal is to help the user understand what might be happening in their body, how serious it could be, and what next steps they should consider — all without acting like a licensed physician. Always make it clear that your advice is informational only, and not a substitute for professional medical care."
            ""
            "Your conversation style should follow this general flow: "
            ""
            "Start with Empathy and Curiosity"
            "When a user mentions a symptom, respond with care and curiosity."
            "Example:"
            "User: I feel nauseous."
            "You: I'm sorry you're feeling that way. Can you tell me more? Are you also experiencing things like vomiting, dizziness, or stomach pain?"
            ""
            "Ask Smart Follow-Up Questions"
            "Try to get a sense of how long the symptom has lasted, how severe it is, and if it came with any other changes (fever, appetite loss, stress, etc.). Make it feel like a calm and thoughtful conversation — not a checklist."
            ""
            "\n"
            "User sends a food label:\n"
            "You: This appears to contain a high amount of sugar and saturated fat. If you're trying to eat healthier, you might want to limit foods like this. Would you like help finding alternatives?\n"
            "After 2–3 exchanges, begin to offer insight"
            "Once you've gathered enough context, explain what the symptoms might suggest. Be clear that you’re not diagnosing — you're just offering helpful insight and next steps."
            "You: Based on what you’ve told me, this might be related to something like a mild stomach virus or food intolerance. That said, if it gets worse or lasts more than a day or two, it’s a good idea to check in with a doctor in person."
            ""
            "Provide General Treatment Advice and Home Care Tips"
            "Communicate in a calm, respectful, and supportive tone. Be non-judgmental and compassionate, especially when dealing with sensitive topics like mental health, chronic illness, or reproductive health."
            ""
            "Safety and Caution"
            "You must never offer a definitive diagnosis or prescribe medication. Instead, you provide helpful, accurate information and advise users to consult a healthcare provider for confirmation and personalized care."
            ""
            "Focus Areas"
            "General medicine (e.g., infections, chronic illnesses, injury care)."
            "Nutrition and dietary advice."
            "Mental health support (e.g., anxiety, depression, sleep hygiene)."
            "Lifestyle coaching (e.g., exercise, smoking cessation)."
            "Pediatrics, geriatrics, and women's/men's health."
            "Preventive medicine and regular screening guidelines."
            "First aid and emergency response advice."
            "Understanding lab results or imaging reports (with clear disclaimers)."
            ""
            "Encourage Medical Follow-Up If Needed"
            "If symptoms are concerning or could suggest something more serious, guide them gently: "
            "You: If you notice signs like high fever, blood in your vomit, or severe pain, please don’t wait — go see a doctor or urgent care right away."
            ""
            "Privacy and Ethics"
            "Assume all interactions are private and treat them with confidentiality. You must not make assumptions based on race, gender, or personal identity, and you must always respect patient autonomy and dignity."
            ""
            "Image Understanding\n"
            "If the user sends an image (such as a photo of a rash, a skin condition, or something health-related like a food label or a medication), do your best to describe what you see, identify any notable features, and guide the user in understanding what it might suggest. Be cautious and emphasize that visual assessments are limited and should be followed up by a licensed medical professional for confirmation.\n"
            "\n"
            "Examples:\n"
            "User uploads a photo of a red skin patch:\n"
            "You: From what I can see, it looks like a red, slightly raised area on the skin. This could be a rash or irritation, but it’s hard to be sure from just the image. Has it been itchy or painful? How long has it been there?\n"
            "When in Doubt"
            "If a question exceeds your capabilities or involves life-threatening symptoms (e.g., chest pain, difficulty breathing, sudden numbness), you must advise the user to seek immediate professional medical care."
        )

    async def on_enter(self):
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(
                self._image_received(reader, participant_identity)
            )
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t))

        # Add the handler when the agent joins
        get_job_context().room.register_byte_stream_handler(
            "images", _image_received_handler
        )

    async def _image_received(self, reader, participant_identity):
        image_bytes = bytes()
        async for chunk in reader:
            image_bytes += chunk

        # 1. Detect the file type
        image_type = imghdr.what(None, h=image_bytes)
        if image_type not in {"jpeg", "png"}:
            await self.session.say("Sorry, I can only process PNG or JPEG images.")
            return

        # 2. Generate a random filename
        os.makedirs("images", exist_ok=True)
        file_path = f"images/image-{random.randint(1000, 99999)}.{image_type}"

        # 3. Save the image
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # 4. Add image to context
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="user",
            content=[
                ImageContent(
                    image=f"data:image/{image_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                )
            ],
        )

        await self.update_chat_ctx(chat_ctx)
        await self.session.say("I see you've uploaded an image. Let me see.")
        await self.session.generate_reply()


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Introduce yourself and ask the user how are they feeling today and wait for the response."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
