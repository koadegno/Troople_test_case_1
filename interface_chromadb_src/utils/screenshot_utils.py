import chromadb
from chromadb.utils import embedding_functions
import numpy as np

from dotenv import load_dotenv
import os

from PIL import Image
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


pipe = pipeline("image-to-text", model="naver-clova-ix/donut-base")

load_dotenv()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


class ChromadbClient(metaclass=Singleton):

    def __init__(self) -> None:
        super().__init__()
        self.chromadb_client = chromadb.PersistentClient()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_functions = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBED_MODEL"))
        self.text_embeddings_collection = self.chromadb_client.get_or_create_collection(name=os.getenv("DB_NAME"), embedding_function=self.embedding_functions, metadata={"hnsw:space": "cosine"})

        self.PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

    def add_image(self, text: str, img_id: str):
        """Add an image to chromadb

        Args:
            image (Image): The image in PIL format
            img_id (str): The id of the image
        """
        self.text_embeddings_collection.add(documents=[text], ids=[img_id])

    def ask(
        self,
        question: str,
    ):
        res = self.chromadb_client.query([question], n_results=3)
        context_text = "\n\n - -\n\n".join([image_description for image_description in res["documents"]])
        prompt_template = self.PROMPT_TEMPLATE.format(context_text, question)
        return self.call_LLM(prompt_template)

    def call_LLM(self, client:OpenAI, prompt: str) -> str:
        """Make a call to the OpenAI model and return the text result

        Args:
            client (OpenAI): The openAI client
            prompt (str): The whole prompt 

        Returns:
            str: the result of the call
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.1,
        )

        return response.choices[0].message.content


def describe_image(
    image: Image,
    description_text: str = "What is on the image, gives details and facts about the page title, page URL, page content or application content and any other information for a blind person.",
) -> str:
    """Describe the contents of an given image

    Args:
        image (Image): The image in format PIL.image
        text (str, optional): The text given to the model to make the description. Defaults to "What is on the image: ".

    Returns:
        str: The description of the image.
    """

    print("Preprocessing ...")
    enc_image = model.encode_image(image)

    return model.answer_question(enc_image, description_text, tokenizer)


if __name__ == "__main__":
    raw_image = Image.open(r"C:\Users\Django\Desktop\troople_test_project\data\photo_1732545986.4264119.jpg").convert("RGB")
    describe_image(raw_image)
