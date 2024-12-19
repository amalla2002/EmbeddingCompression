import os
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
from mteb import MTEB
from mteb.tasks import Banking77Classification, STSBenchmarkSTS

client = OpenAI(api_key="API_KEY")

def specialProcess(input_csv='EmbeddingStore.csv', output_csv='EmbeddingValuesMap.csv', k=0.75, approach=1):
    df = pd.read_csv(input_csv)
    std_per_column = df.std()
    if approach == 2:
        std_per_column = (std_per_column - std_per_column.mean()) / std_per_column.std()

    indexed_arr = [(num, idx) for idx, num in enumerate(np.array(std_per_column))]
    sorted_by_abs = sorted(indexed_arr, key=lambda x: abs(x[0]))

    means_per_column = df.mean()
    embedding_indices = [i[1] for i in sorted_by_abs[:int(k * len(means_per_column))]]
    embedding_values = [means_per_column[i] for i in embedding_indices]

    embedding_values_map = pd.DataFrame({
        'index': embedding_indices,
        'value': embedding_values
    })
    embedding_values_map.to_csv(output_csv, index=False)

def specialProcess2(input_csv='EmbeddingStore.csv', output_csv='EmbeddingValuesMap.csv', k=0.5, approach=1):
    df = pd.read_csv(input_csv)
    std_per_column = df.std()
    if approach == 2:
        std_per_column = (std_per_column - std_per_column.mean()) / std_per_column.std()

    indexed_arr = [(num, idx) for idx, num in enumerate(np.array(std_per_column))]
    sorted_by_abs = sorted(indexed_arr, key=lambda x: abs(x[0]))

    means_per_column = df.mean()
    embedding_indices = [i[1] for i in sorted_by_abs[:int(k * len(means_per_column))]]
    embedding_values = [means_per_column[i] for i in embedding_indices]

    embedding_values_map = pd.DataFrame({
        'index': embedding_indices,
        'value': embedding_values
    })
    embedding_values_map.to_csv(output_csv, index=False)


class BaseOpenAIEmbedder:
    """Base class to handle OpenAI embeddings and dimension extraction."""
    def __init__(self, model_name, batch_size=16, dimensions=None):
        """
        Args:
            model_name (str): Name of the OpenAI embedding model.
            batch_size (int): Batch size for embedding requests.
            dimensions (int, optional): If specified, request embeddings with this many dimensions.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._embedding_dim = None
        self.dimensions = dimensions

    def _get_embeddings(self, texts):
        kwargs = {
            "model": self.model_name,
            "input": texts
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**kwargs)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def encode(self, sentences):
        raise NotImplementedError("Subclasses must implement encode.")


class CustomOpenAIEmbedder(BaseOpenAIEmbedder):
    def __init__(self, 
                 model_name,
                 embedding_store_file='EmbeddingStore.csv',
                 embedding_values_map_file='EmbeddingValuesMap.csv',
                 special_process_function=None,
                 batch_size=16,
                 dimensions=None):
        super().__init__(model_name, batch_size, dimensions)
        self.embedding_counter = 0
        self.embedding_store_file = embedding_store_file
        self.embedding_values_map_file = embedding_values_map_file
        self.special_process_function = special_process_function
        self.embedding_values_map = {}

        # Initialize CSV if it doesn't exist
        if not os.path.isfile(self.embedding_store_file):
            embedding_dim = self.dimensions if self.dimensions else 1536
            headers = [f"dim_{i}" for i in range(1, embedding_dim + 1)]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.embedding_store_file, index=False)

    def encode(self, sentences, **kwargs):
        adjusted_embeddings = []
        for batch_start in range(0, len(sentences), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_sentences = sentences[batch_start:batch_end]
            embeddings = self._get_embeddings(batch_sentences)

            for embedding in embeddings:
                self.embedding_counter += 1
                if self.embedding_counter <= 50:
                    self.save_embedding_to_store(embedding)
                    if self.embedding_counter == 50:
                        if self.special_process_function:
                            self.special_process_function(
                                input_csv=self.embedding_store_file,
                                output_csv=self.embedding_values_map_file
                            )
                            self.load_embedding_values_map()
                        else:
                            raise ValueError("special_process_function is not defined.")
                else:
                    embedding = self.override_embedding(embedding)
                adjusted_embeddings.append(embedding)
        return np.array(adjusted_embeddings)

    def save_embedding_to_store(self, embedding):
        embedding_df = pd.DataFrame([embedding], columns=[f"dim_{i}" for i in range(1, len(embedding)+1)])
        embedding_df.to_csv(self.embedding_store_file, mode='a', header=False, index=False)

    def load_embedding_values_map(self):
        if not os.path.isfile(self.embedding_values_map_file):
            raise FileNotFoundError(f"{self.embedding_values_map_file} does not exist.")
        df_map = pd.read_csv(self.embedding_values_map_file)
        self.embedding_values_map = dict(zip(df_map['index'], df_map['value']))

    def override_embedding(self, embedding):
        for index, value in self.embedding_values_map.items():
            if 0 <= index < len(embedding):
                embedding[index] = value
            else:
                print(f"Warning: Index {index} out of range.")
        return embedding


class NormalOpenAIEmbedder(BaseOpenAIEmbedder):
    def encode(self, sentences, **kwargs):
        all_embeddings = []
        for batch_start in range(0, len(sentences), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_sentences = sentences[batch_start:batch_end]
            embeddings = self._get_embeddings(batch_sentences)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)


class QuarterDimOpenAIEmbedder(BaseOpenAIEmbedder):
    def __init__(self, model_name, batch_size=16):
        super().__init__(model_name, batch_size, dimensions=None)
        self.base_dimensions = 1536
        self.quarter_dimensions = self.base_dimensions // 4

    def _get_embeddings(self, texts):
        kwargs = {
            "model": self.model_name,
            "input": texts,
            "dimensions": self.quarter_dimensions 
        }

        response = client.embeddings.create(**kwargs)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def encode(self, sentences, **kwargs):
        all_embeddings = []
        for batch_start in range(0, len(sentences), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_sentences = sentences[batch_start:batch_end]
            embeddings = self._get_embeddings(batch_sentences)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

class CombinedOpenAIEmbedder(CustomOpenAIEmbedder):
    def __init__(self, 
                 model_name,
                 embedding_store_file='EmbeddingStore.csv',
                 embedding_values_map_file='EmbeddingValuesMap.csv',
                 special_process_function=None,
                 batch_size=16):
        self.base_dimensions = 1536
        half_dimensions = self.base_dimensions // 2
        super().__init__(model_name,
                         embedding_store_file=embedding_store_file,
                         embedding_values_map_file=embedding_values_map_file,
                         special_process_function=special_process_function,
                         batch_size=batch_size,
                         dimensions=half_dimensions)

# Example tasks and model
model_name = "text-embedding-3-large"
tasks = [
    Banking77Classification(hf_subsets=["en"]),
    STSBenchmarkSTS(hf_subsets=["en"])
]

# 1. Custom run with k = 0.75
for task in tasks:
    benchmark = MTEB(tasks=[task])
    custom_model = CustomOpenAIEmbedder(
        model_name=model_name,
        embedding_store_file=f"Data/{model_name}_{task}_EmbeddingStore.csv",
        embedding_values_map_file=f"Data/{model_name}_{task}_EmbeddingValuesMap.csv",
        special_process_function=specialProcess,
        batch_size=16
    )
    custom_results = benchmark.run(custom_model, output_folder=f"Result/Combined/Custom_1/{model_name}")

# 2. Quarter-Dimension run directly via the API
for task in tasks:
    benchmark = MTEB(tasks=[task])
    quarterdim_model = QuarterDimOpenAIEmbedder(model_name=model_name, batch_size=16)
    quarterdim_results = benchmark.run(quarterdim_model, output_folder=f"Result/Combined/API/{model_name}")

# 3. Combined: Half Dimension via API, then apply Custom approach to halve again
for task in tasks:
    benchmark = MTEB(tasks=[task])
    combined_model = CombinedOpenAIEmbedder(
        model_name=model_name,
        embedding_store_file=f"Data/{model_name}_{task}_EmbeddingStore.csv",
        embedding_values_map_file=f"Data/{model_name}_{task}_EmbeddingValuesMap.csv",
        special_process_function=specialProcess2,
        batch_size=16
    )
    combined_results = benchmark.run(combined_model, output_folder=f"Result/Combined/Quarter/{model_name}")
