import time
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from mteb import MTEB
import mteb

def specialProcess(input_csv='EmbeddingStore.csv', output_csv='EmbeddingValuesMap.csv', k = 0.2):
    """
    Processes the first 50 embeddings and creates a mapping file.

    Args:
        input_csv (str): Path to the input CSV file containing embeddings.
        output_csv (str): Path to the output CSV file to store value-index pairs.
    """
    # Read the embeddings from EmbeddingStore.csv
    df = pd.read_csv(input_csv)
    
    # Grab the indexes of the standard deviations closest to 0
    std_per_column = df.std()
    indexed_arr = [(num, idx) for idx, num in enumerate(np.array(std_per_column))]
    sorted_by_abs = sorted(indexed_arr, key=lambda x: abs(x[0]))

    # Extract the (mean, index) for lowest k% absolute value
    means_per_column = df.mean()
    embedding_indices = [i[1] for i in sorted_by_abs[:int(k*len(means_per_column))]]
    embedding_values = [means_per_column[i] for i in embedding_indices]

    # Create a DataFrame with (index, value) pairs
    embedding_values_map = pd.DataFrame({
        'index': embedding_indices,
        'value': embedding_values
    })
    
    # Save the embedding values map to CSV
    embedding_values_map.to_csv(output_csv, index=False)
    
    # print(f"specialProcess: Created {output_csv} with {len(embedding_indices)} entries.")

class CustomSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name, 
                 embedding_store_file='EmbeddingStore.csv',
                 embedding_values_map_file='EmbeddingValuesMap.csv',
                 special_process_function=None,
                 *args, **kwargs):
        """
        Initializes the custom SentenceTransformer.

        Args:
            model_name (str): Name of the pre-trained SentenceTransformer model.
            embedding_store_file (str): Path to store the first 50 embeddings.
            embedding_values_map_file (str): Path to store the embedding values map.
            special_process_function (callable): Function to process embeddings after 50 are stored.
            *args, **kwargs: Additional arguments for the base SentenceTransformer.
        """
        super().__init__(model_name, *args, **kwargs)
        self.embedding_counter = 0
        self.embedding_store_file = embedding_store_file
        self.embedding_values_map_file = embedding_values_map_file
        self.special_process_function = special_process_function
        self.embedding_values_map = {}  # In-memory mapping

        # Initialize EmbeddingStore.csv with headers if it doesn't exist
        if not os.path.isfile(self.embedding_store_file):
            embedding_dim = self.get_sentence_embedding_dimension()
            headers = [f"dim_{i}" for i in range(1, embedding_dim + 1)]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.embedding_store_file, index=False)
            # print(f"Initialized {self.embedding_store_file} with headers.")

    def encode(self, sentences, batch_size=None, show_progress_bar=False, **kwargs):
        """
        Overrides the encode method to include custom operations:
        - Save the first 50 embeddings to EmbeddingStore.csv.
        - After the 50th embedding, trigger specialProcess.
        - From the 51st embedding onwards, override embeddings based on EmbeddingValuesMap.csv.

        Args:
            sentences (List[str]): List of sentences to encode.
            batch_size (int, optional): Batch size for encoding.
            show_progress_bar (bool, optional): Whether to display a progress bar.
            **kwargs: Additional arguments for the base encode method.

        Returns:
            np.ndarray: Array of processed embeddings.
        """
        # Use the default batch_size if not specified
        if batch_size is None:
            batch_size = self.get_default_batch_size()

        adjusted_embeddings = []

        for batch_start in range(0, len(sentences), batch_size):
            batch_end = batch_start + batch_size
            batch_sentences = sentences[batch_start:batch_end]

            # Generate embeddings for the current batch
            embeddings = super().encode(batch_sentences, batch_size=batch_size, 
                                        show_progress_bar=False, convert_to_numpy=True, **kwargs)

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
                
                elif self.embedding_counter > 50:
                    embedding = self.override_embedding(embedding)

                adjusted_embeddings.append(embedding)

        # Convert the list of adjusted embeddings to a NumPy array
        adjusted_embeddings = np.array(adjusted_embeddings)

        return adjusted_embeddings

    def save_embedding_to_store(self, embedding):
        """
        Saves an embedding to EmbeddingStore.csv.

        Args:
            embedding (np.ndarray): The embedding to save.
        """
        # Convert embedding to a DataFrame row
        embedding_df = pd.DataFrame([embedding], columns=[f"dim_{i}" for i in range(1, len(embedding)+1)])
        
        # Append to the CSV file without headers
        embedding_df.to_csv(self.embedding_store_file, mode='a', header=False, index=False)
        # print(f"Saved embedding #{self.embedding_counter} to {self.embedding_store_file}.")

    def load_embedding_values_map(self):
        """
        Loads the embedding values map from EmbeddingValuesMap.csv into memory.
        """
        if not os.path.isfile(self.embedding_values_map_file):
            raise FileNotFoundError(f"{self.embedding_values_map_file} does not exist.")

        df_map = pd.read_csv(self.embedding_values_map_file)
        self.embedding_values_map = dict(zip(df_map['index'], df_map['value']))
        # print(f"Loaded embedding values map from {self.embedding_values_map_file}.")

    def override_embedding(self, embedding):
        """
        Overrides embedding values based on EmbeddingValuesMap.csv.

        Args:
            embedding (np.ndarray): The original embedding.

        Returns:
            np.ndarray: The overridden embedding.
        """
        for index, value in self.embedding_values_map.items():
            if 0 <= index < len(embedding):
                embedding[index] = value
            else:
                print(f"Warning: Index {index} out of bounds for embedding dimension {len(embedding)}.")
        # print(f"Overridden embedding #{self.embedding_counter} based on in-memory map.")
        return embedding

    def operation_after_each_embedding(self, global_index):
        """
        Optional: Define any operation to perform after each embedding is processed.

        Args:
            global_index (int): The global index of the embedding.
        """
        # print(f"Processed embedding #{global_index}")

model_names = [
    "all-MiniLM-L6-v2",
    "thenlper/gte-large",
    "all-mpnet-base-v2",
]
tasks = [
    "Banking77Classification",      # Classification
    "STSBenchmark",                 # STS
    # "MSMARCO",                      # Retrieval
    # "TwentyNewsgroupsClustering"    # Classification & Clustering 
]
k = 0.4
times = []
for model in model_names:
    for task in tasks:
        # Initialize the MTEB benchmark
        benchmark = MTEB(tasks = [task])

        # Instantiate the custom model with the specialProcess function
        custom_model = CustomSentenceTransformer(
            model_name=model,
            embedding_store_file=f"Data/{model}_{task}_EmbeddingStore.csv",
            embedding_values_map_file=f"Data/{model}_{task}_EmbeddingValuesMap.csv",
            special_process_function=specialProcess
        )
        normal_model = SentenceTransformer(model)
        
        # Run the evaluation
        start_time = time.time()
        resultsCustom = benchmark.run(custom_model, output_folder=f"Result/Custom/{model}")
        resultsNormal = benchmark.run(normal_model, output_folder=f"Result/Normal/{model}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(f"{elapsed_time:.6f}_{model}_{task}\n")

with open("Result/timing_results.txt", "w") as file:
    file.writelines(times)