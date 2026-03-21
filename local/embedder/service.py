"""gRPC servicer that wraps a sentence-transformers model."""

import numpy as np

import embedder_pb2
import embedder_pb2_grpc


class EmbedderServicer(embedder_pb2_grpc.EmbedderServiceServicer):
    """Serves embedding requests using a loaded SentenceTransformer model."""

    def __init__(self, model, model_name: str):
        self._model = model
        self._model_name = model_name
        self._dimension = model.get_sentence_embedding_dimension()

    def Embed(self, request, context):
        texts = list(request.texts)
        if not texts:
            return embedder_pb2.EmbedResponse()

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = []
        for vec in vectors:
            embeddings.append(
                embedder_pb2.Embedding(values=vec.astype(np.float32).tolist())
            )

        return embedder_pb2.EmbedResponse(embeddings=embeddings)

    def Info(self, request, context):
        return embedder_pb2.InfoResponse(
            model=self._model_name,
            dimension=self._dimension,
        )
