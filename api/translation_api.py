import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.config import Configuration
from pipeline.full_pipeline import TranslationPipeline


class TranslationItem(BaseModel):
    text: str
    model: str


def create_app():
    config = Configuration()
    pipeline = TranslationPipeline(config)
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/translate")
    async def translate(translation: TranslationItem):
        text = translation.text
        model = translation.model
        translated_text, model_type = await pipeline(text, model)
        return {
            'IsSuccessed': True,
            'Message': 'Success',
            'ResultObj': {
                'src': text,
                'tgt': translated_text
            }
        }

    return app


if __name__ == "__main__":
    app_ = create_app()
    uvicorn.run(app_, host="0.0.0.0", port=8000)


