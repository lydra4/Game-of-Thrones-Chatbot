import logging
from typing import List, Optional, Tuple

import gradio as gr
import omegaconf
from inference.inference_pipeline import InferencePipeline
from langfuse.decorators import observe


class GradioApp:
    """
    A class to run a Gradio interface for a Retrieval-Augmented Generation (RAG) chatbot application.

    This app initializes an inference pipeline and serves a Gradio-based user interface,
    enabling users to ask questions and receive answers powered by a QA chain.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration object containing app and pipeline settings.
        logger (logging.Logger): Logger instance used for logging information.
        first_part (Optional[str]): First markdown section to be displayed in the UI.
        second_part (Optional[str]): Placeholder text to be shown in the textbox.
        inference_pipeline (InferencePipeline): Instance of the RAG pipeline for answering user questions.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the GradioApp with the configuration and logger.

        The method sets up the RAG pipeline by loading embedding models, the vector store,
        the language model, the retriever, and the QA chain.

        Args:
            cfg (omegaconf.DictConfig): Configuration object with all required parameters.
            logger (Optional[logging.Logger], optional): Logger for debugging and status updates. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.first_part: Optional[str] = None
        self.second_part: Optional[str] = None
        self.inference_pipeline = InferencePipeline(cfg=cfg, logger=self.logger)

        self.inference_pipeline.load_embedding_model()
        self.inference_pipeline._load_vectordb()
        self.inference_pipeline._initialize_llm()
        self.inference_pipeline._create_retriever()
        self.inference_pipeline._create_qa_chain()

    def _load_captions(self):
        """
        Loads markdown caption parts from a file defined in the configuration.

        This method reads the file, splits the contents by a delimiter (---),
        and stores them as markdown header and textbox placeholder for the UI.
        """
        with open(file=self.cfg.gradio.captions_path, mode="r", encoding="utf-8") as f:
            content = f.read()

        parts = content.strip().split("---")

        self.first_part = parts[0]
        self.second_part = parts[1]

    @observe()
    def chat_response(
        self, history: List[Tuple[str, str]], question: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Generates a chatbot response for a given question and updates the chat history.

        Args:
            history (List[Tuple[str, str]]): Current chat history as a list of (question, answer) pairs.
            question (str): The user's current question input.

        Returns:
            Tuple[List[Tuple[str, str]], str]: Updated chat history and an empty string to reset the textbox.
        """
        response = self.inference_pipeline.qa_chain.invoke(
            {"query": question},
            config={"callbacks": [self.inference_pipeline.langfuse_handler]},
        )
        answer = response["result"]
        history.append((question, answer))

        return history, ""

    @observe()
    def launch_app(self):
        """
        Launches the Gradio web application.

        This method sets up the Gradio interface using markdown headers, a chatbot display,
        and a textbox for user input. User inputs are routed through the QA pipeline for response generation.
        """
        self._load_captions()

        with gr.Blocks() as demo:
            gr.Markdown(self.first_part)
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(placeholder=self.second_part, show_label=False)

            def respond(history, question):
                updated_history, _ = self.chat_response(
                    history=history, question=question
                )
                return updated_history

            user_input.submit(respond, [chatbot, user_input], [chatbot]).then(
                lambda: "", [], [user_input]
            )

        demo.launch()
