from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any, List, Iterator, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from .tokenize import LLAMATokenizer

class ParrotDataset(TorchDataset):
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
        self.processed_tokens = []

    def load_from_huggingface(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        **kwargs
    ) -> None:
        """
        Load dataset from Hugging Face datasets.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            dataset_config: Specific configuration of the dataset (e.g., 'wikitext-2-v1')
            split: Dataset split to use (e.g., 'train', 'validation')
            text_column: Column name containing the text data
            **kwargs: Additional arguments to pass to load_dataset
        """
        if dataset_config:
            self.dataset = load_dataset(dataset_name, dataset_config, split=split, **kwargs)
        else:
            self.dataset = load_dataset(dataset_name, split=split, **kwargs)
        self.text_column = text_column

    def load_from_files(
        self,
        files: List[str],
        text_column: str = "text"
    ) -> None:
        """
        Load dataset from local text files.
        
        Args:
            files: List of file paths to load
            text_column: Name to use for the text column
        """
        self.dataset = load_dataset('text', data_files=files)
        self.text_column = text_column

    def prepare_dataset(self) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Process all examples
        for example in self.dataset:
            tokens = self.tokenizer.encode(example[self.text_column])
            # Split into chunks of max_length
            for i in range(0, len(tokens) - self.max_length + 1, self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) == self.max_length:
                    self.processed_tokens.append(chunk)

    def __len__(self):
        return len(self.processed_tokens)

    def __getitem__(self, idx):
        tokens = self.processed_tokens[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        # Create targets by shifting input
        targets = torch.roll(input_ids, -1)
        targets[-1] = self.tokenizer.pad_token_id
        return input_ids, targets

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def tokenize(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and format the examples"""

        tokens = self.tokenizer.encode(examples[self.text_column])

        return {
            'tokens': tokens
        }

    def get_batch_iterator(self, shuffle: bool = True) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
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
