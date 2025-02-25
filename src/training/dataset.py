from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any, List, Iterator, Tuple
import mlx.core as mx
import numpy as np
from .tokenize import LLAMATokenizer

class ParrotDataset:
    def __init__(
        self,
        tokenizer: LLAMATokenizer,
        max_length: int = 256,
        batch_size: int = 32
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset = None

    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        **kwargs
    ) -> None:
        """Load a dataset from Hugging Face hub"""
        self.dataset = load_dataset(dataset_name, 'wikitext-2-v1', split=split, **kwargs)
        self.text_column = text_column

    def load_from_files(
        self,
        files: List[str],
        text_column: str = "text"
    ) -> None:
        """Load dataset from local text files"""
        self.dataset = load_dataset('text', data_files=files)
        self.text_column = text_column

    def tokenize(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and format the examples"""

        tokens = self.tokenizer.encode(examples[self.text_column])

        return {
            'tokens': tokens
        }

    def prepare_dataset(self) -> None:
        """Prepare the dataset by tokenizing and formatting"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_from_huggingface or load_from_files first.")
        
        self.processed_dataset = self.dataset.map(
            self.tokenize,
            remove_columns=self.dataset.column_names,
            batched=False,
        )

    def get_batch_iterator(self, shuffle: bool = True) -> Iterator[Tuple[mx.array, mx.array]]:
        """Get a batch iterator for MLX training, returning input_ids and targets."""
        if not hasattr(self, 'processed_dataset'):
            raise ValueError("Dataset not prepared. Call prepare_dataset first.")

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        def generate_batches():
            examples = list(self.processed_dataset)
            if shuffle:
                np.random.shuffle(examples)

            current_batch = []
            for example in examples:
                tokens = example['tokens']
                token_chunks = list(chunks(tokens, self.max_length))

                for chunk in token_chunks:
                    # Pad if necessary
                    if len(chunk) < self.max_length:
                        chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
                    current_batch.append(chunk)

                    if len(current_batch) == self.batch_size:
                        batch_array = np.array(current_batch, dtype=np.int32)
                        input_ids = mx.array(batch_array)
                        # Create targets: shift the tokens one position to the left
                        target_array = batch_array.copy()
                        target_array[:, :-1] = batch_array[:, 1:]
                        target_array[:, -1] = self.tokenizer.pad_token_id  # set last token target to pad
                        targets = mx.array(target_array)
                        
                        yield input_ids, targets
                        current_batch = []

            # Handle any remaining examples as a final batch
            if current_batch:
                while len(current_batch) < self.batch_size:
                    current_batch.append([self.tokenizer.pad_token_id] * self.max_length)
                
                batch_array = np.array(current_batch, dtype=np.int32)
                input_ids = mx.array(batch_array)
                target_array = batch_array.copy()
                target_array[:, :-1] = batch_array[:, 1:]
                target_array[:, -1] = self.tokenizer.pad_token_id
                targets = mx.array(target_array)
                
                yield input_ids, targets

        return generate_batches()
