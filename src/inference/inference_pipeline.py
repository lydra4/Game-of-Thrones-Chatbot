import locale
import logging
import os
import re
from typing import Optional

import omegaconf
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from ragas import EvaluationDataset
from utils.general_utils import load_embedding_model


class InferencePipeline:
    """
    A pipeline for performing Retrieval-Augmented Generation (RAG) inference.

    This pipeline integrates embedding models, vector databases, retrieval mechanisms,
    and language models to generate contextually relevant responses from a given set of questions.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration dictionary for the pipeline.
        logger (logging.Logger): Logger instance for capturing pipeline logs.
        langfuse (Langfuse): Langfuse instance for observability.
        langfuse_handler (CallbackHandler): Langfuse callback handler.
        embedding_model (HuggingFaceInstructEmbeddings): Embedding model used for vectorizing text.
        vectordb (FAISS): FAISS-based vector store for semantic search.
        llm (Union[ChatOpenAI, ChatGoogleGenerativeAI]): Language model for answering questions.
        retriever: Document retriever instance.
        qa_chain (RetrievalQA): QA chain combining retriever and LLM.
        qns_list (list): List of input questions.
        ans_list (list): List of generated answers.
        answer_file (str): File path to store inference results.
        cleaned_text_splitter (str): Human-readable name of text chunking strategy.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the inference pipeline with configuration and optional logger.

        Args:
            cfg (omegaconf.DictConfig): Configuration dictionary for the pipeline.
            logger (Optional[logging.Logger], optional): Logger instance. If not provided, uses default logger.
        """
        load_dotenv()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings_model_name: str = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg.embeddings.text_embeddings.model_name.split("/")[-1],
        )
        self.persist_directory: str = os.path.join(
            self.cfg.embeddings.text_embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
        )

    def _load_vectordb(self, vector_db_path: str, embedding_function):
        return Chroma(
            persist_directory=vector_db_path, embedding_function=embedding_function
        )

    def _load_prompt(
        self, path: str, input_variables: Optional[list] = None
    ) -> PromptTemplate:
        """
        Loads a prompt template from a file.

        Args:
            path (str): Path to the prompt template file.
            input_variables (Optional[list], optional): List of expected variables in the prompt.

        Returns:
            PromptTemplate: Parsed prompt template object.

        Raises:
            Exception: If loading or reading the prompt file fails.
        """
        file_name = os.path.basename(path)
        self.logger.info(f"Loading Prompt Template: {file_name}.\n")

        try:
            with open(
                file=path,
                mode="r",
                encoding="utf-8",
            ) as f:
                template = f.read()

            prompt = PromptTemplate(
                template=template, input_variables=input_variables or []
            )
            self.logger.info(f"{file_name} loaded successfully.\n")

            return prompt

        except Exception as e:
            self.logger.error(f"Failed to load {file_name}: {e}")
            raise

    def _initialize_llm(self) -> None:
        """
        Initializes the language model (LLM) such as OpenAI GPT or Google Gemini.

        Raises:
            ValueError: If the required API key is missing.
            Exception: If the model initialization fails.
        """
        load_dotenv()

        model = self.cfg.llm.model
        temperature = self.cfg.llm.temperature
        api_key_env_var = "OPENAI_API_KEY" if "gpt-" in model else "GEMINI_API_KEY"
        api_key = os.getenv(api_key_env_var)

        if not api_key:
            raise ValueError(f"{api_key_env_var} not found in enviroment variables.")

        self.logger.info(f"Initializing {model}.\n")

        try:
            if "gpt-" in model:
                self.llm = ChatOpenAI(
                    model=model, temperature=temperature, api_key=api_key
                )

            elif "gemini" in model:
                self.llm = ChatGoogleGenerativeAI(
                    model=model, temperature=temperature, api_key=api_key
                )

            else:
                raise ValueError(f"Unsupported model: {model}")

            self.logger.info(f"LLM successfully initialized with model: {model}.\n")

        except Exception as e:
            self.logger.error(f"Failed to initialize {model}: {e}")
            raise

    def _create_retriever(self) -> None:
        """
        Creates a retriever to search the FAISS vector database and optionally wraps it with:
            - MultiQueryRetriever for diverse reformulations.
            - CohereRerank for reranking results.

        Raises:
            ValueError: If reranking is enabled but the Cohere API key is not found.
        """
        self.logger.info("Initializing document retriever.\n")
        retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": self.cfg.retrieve.k,
                "search_type": self.cfg.retrieve.search_type,
            }
        )

        if self.cfg.retrieve.use_multiquery:
            prompt = self._load_prompt(
                path=self.cfg.retrieve.multiquery.path_to_multiquery_prompt
            )

            self.logger.info("Using Multiquery Retriever.\n")
            retriever = MultiQueryRetriever.from_llm(
                retriever=retriever,
                llm=self.llm,
                include_original=self.cfg.retrieve.multiquery.include_original,
                prompt=prompt,
            )

        if self.cfg.retrieve.reranker_model:
            self.logger.info(
                f"Wrapping retriever with reranker model: {self.cfg.retrieve.reranker_model}.\n"
            )

            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                self.logger.error("Cohere API key is missing.")
                raise ValueError("Cohere API key is missing.")

            compressor = CohereRerank(
                model=self.cfg.retrieve.reranker_model, cohere_api_key=cohere_api_key
            )
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

        else:
            self.retriever = retriever

    def _create_qa_chain(self):
        """
        Builds the RetrievalQA chain using the selected prompt, retriever, and LLM.
        """
        prompt = self._load_prompt(
            path=self.cfg.path_to_qa_prompt, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.cfg.retrieval.chain_type,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=self.cfg.retrieval.return_source_documents,
            verbose=self.cfg.retrieval.verbose,
        )

    def _open_questions(self):
        """
        Loads a list of questions and their reference answers from the configured text file.

        Raises:
            Exception: If the file cannot be read or parsed.
        """
        self.logger.info("Loading questions.\n")

        try:
            with open(
                file=self.cfg.llm.path_to_qns, mode="r", encoding=locale.getencoding()
            ) as f:
                lines = [line.rstrip("\n").strip() for line in f.readlines()]

                self.qns_list = [line.split(" - ", 1)[0].strip() for line in lines]
                self.ground_truth = [line.split(" - ", 1)[1].strip() for line in lines]

            self.logger.info(f"Loaded {len(self.qns_list)} questions.\n")

        except Exception as e:
            self.logger.error(f"Failed to load questions: {e}")
            raise

    def _infer(self):
        """
        Runs inference over the questions using the QA chain and saves responses to a file.

        Returns:
            EvaluationDataset: Dataset object containing question, answer, context, and references.
            int: Number of questions processed.

        Raises:
            Exception: If inference or file writing fails.
        """
        folder_to_answers = os.path.dirname(self.cfg.llm.path_to_ans)
        os.makedirs(name=folder_to_answers, exist_ok=True)

        self.logger.info(f"Saving answers to {self.cfg.llm.path_to_ans}.\n")
        data_list = []

        with open(
            file=self.cfg.llm.path_to_ans,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as self.answer_file:
            for question, ground_truth in zip(self.qns_list, self.ground_truth):
                retrieved_docs = self.retriever.invoke(
                    input=question, config={"callbacks": [self.langfuse_handler]}
                )

                for document in retrieved_docs:
                    document.page_content = document.page_content[
                        : self.cfg.retrieval.max_tokens
                    ]

                llm_response = self.qa_chain.invoke(
                    {"query": question, "context": retrieved_docs},
                    config={"callbacks": [self.langfuse_handler]},
                )

                self.logger.info(f"\nQuestion: {question}")
                self.logger.info(f"\nAnswer: {llm_response['result']}\n")

                data_list.append(
                    {
                        "user_input": question,
                        "reference": ground_truth,
                        "response": llm_response["result"],
                        "retrieved_contexts": [
                            " ".join([doc.page_content for doc in retrieved_docs])
                        ],
                    }
                )

                self.answer_file.write(f"{question} - {llm_response['result']}.\n")

        return EvaluationDataset.from_list(data=data_list), len(self.qns_list)

    def run_inference(self) -> EvaluationDataset:
        """
        Executes the full RAG pipeline: load models, create retriever, load questions, and run inference.

        Returns:
            EvaluationDataset: Dataset with results from the full inference pipeline.
        """
        text_embedding_model = load_embedding_model(
            model_name=self.cfg.embeddings.text_embeddings.model_name,
            show_progress=self.cfg.embeddings.text_embeddings.show_progress,
        )
        text_vectordb = self._load_vectordb(
            vector_db_path=self.cfg.embeddings.text_embeddings.embeddings_path,
            embedding_function=text_embedding_model,
        )
        image_vectordb = self._load_vectordb(
            vector_db_path=self.cfg.embeddings.image_embeddings.embeddings_path,
            embedding_function=OpenCLIPEmbeddings(
                model_name=self.cfg.embeddings.image_embeddings.model_name,
                checkpoint=self.cfg.embeddings.image_embeddings.checkpoint,
                model=None,
                preprocess=None,
                tokenizer=None,
            ),
        )
        self._initialize_llm()
        self._create_retriever()
        self._create_qa_chain()
        self._open_questions()
        return self._infer()
