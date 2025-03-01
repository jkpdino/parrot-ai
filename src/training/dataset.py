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
            for i in range(0, len(tokens), self.max_length):  # Changed to step by max_length without -1
                chunk = tokens[i:i + self.max_length]
                # Pad shorter chunks to max_length if needed
                if len(chunk) < self.max_length:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
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