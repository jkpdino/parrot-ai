import argparse
from pathlib import Path
import torch
from typing import Optional

from models.gpt import GPT
from models.config import GPTConfig
from training.config import TrainingConfig
from training.tokenize import create_tokenizer

def load_model(checkpoint_path: str, model_name: str) -> tuple[GPT, TrainingConfig]:
    """Load model and config from checkpoint."""
    # Load the config from the same directory
    config_path = Path("config") / (model_name + ".yaml")
    config = TrainingConfig.from_yaml(config_path)
    
    # Initialize model
    model = GPT(GPTConfig.from_yaml(model_name))
    
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # Set to evaluation mode
    
    return model, config

def generate_response(
    model: GPT,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8
) -> str:
    """Generate a response for the given prompt."""
    try:
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(
                tokens,
                max_length=max_length,
                temperature=temperature
            )
        
        return tokenizer.decode(generated[0].tolist())
    except Exception as e:
        return f"Error generating response: {str(e)}"

def chat_loop(model: GPT, tokenizer, max_length: int = 100, temperature: float = 0.8):
    """Interactive chat loop."""
    print("\nParrotLM Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("----------------------------------------")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input:
                continue
            
            response = generate_response(
                model,
                tokenizer,
                user_input,
                max_length=max_length,
                temperature=temperature
            )
            
            print(f"\nParrot: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description='Chat with ParrotLM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name of the model configuration')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('prompt', nargs='?', type=str,
                       help='Optional prompt for single-shot generation')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, config = load_model(args.checkpoint, args.model_name)
    tokenizer = create_tokenizer()
    
    if args.prompt:
        # Single-shot generation
        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(response)
    else:
        # Interactive chat mode
        chat_loop(
            model,
            tokenizer,
            max_length=args.max_length,
            temperature=args.temperature
        )

if __name__ == "__main__":
    main()