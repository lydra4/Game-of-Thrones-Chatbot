import locale
import logging
import os
from typing import Optional

import omegaconf
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_cohere import CohereRerank
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from ragas import EvaluationDataset


class InferencePipeline:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.clip = OpenCLIPEmbeddings(
            model_name=self.cfg.embeddings.model_name,
            checkpoint=self.cfg.embeddings.checkpoint,
            model=None,
            preprocess=None,
            tokenizer=None,
        )
        self.chroma = Chroma(
            persist_directory=self.cfg.persist_directory,
            embedding_function=self.clip,
        )

        load_dotenv()
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("host"),
        )
        self.langfuse_handler = CallbackHandler()

        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.vectordb: Optional[FAISS] = None
        self.llm: Optional[ChatOpenAI] = None
        self.retriever = None
        self.qa_chain = None
        self.qns_list: Optional[list] = None
        self.ans_list: Optional[list] = None
        self.answer_file: Optional[str] = None
        self.cleaned_text_splitter: Optional[str] = None

    def _load_prompt(
        self, path: str, input_variables: Optional[list] = None
    ) -> PromptTemplate:
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
        self.load_embedding_model()
        self._load_vectordb()
        self._initialize_llm()
        self._create_retriever()
        self._create_qa_chain()
        self._open_questions()
        return self._infer()
