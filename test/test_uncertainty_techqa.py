import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Add pytorch_geometric to path if needed
sys.path.insert(0, '/content/pytorch_geometric')

from torch_geometric.llm.models import LLM, GRetriever


def download_techqa_data(dataset_dir="techqa"):
    """Download TechQA dataset if it doesn't exist."""
    json_path = Path(dataset_dir) / "train.json"
    corpus_path = Path(dataset_dir) / "corpus"
    
    if json_path.exists() and corpus_path.exists():
        print(f"Dataset already exists in {dataset_dir}")
        return
    
    print("Downloading TechQA dataset...")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download corpus zip
    zip_path = hf_hub_download(
        repo_id="nvidia/TechQA-RAG-Eval",
        repo_type="dataset",
        filename="corpus.zip",
    )
    
    # Download train.json
    json_download_path = hf_hub_download(
        repo_id="nvidia/TechQA-RAG-Eval",
        repo_type="dataset",
        filename="train.json",
    )
    
    # Extract corpus
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Copy train.json
    import shutil
    shutil.copy(json_download_path, json_path)
    
    print(f"Dataset downloaded to {dataset_dir}")


def test_uncertainty_rag(args):
    """Test LLM with uncertainty estimation on TechQA dataset."""
    
    # Download data if needed
    download_techqa_data(args.dataset_dir)
    
    # Load Q&A pairs
    json_path = Path(args.dataset_dir) / "train.json"
    with open(json_path) as f:
        data = json.load(f)
    
    # Setup LLM with uncertainty
    print(f"Setting up LLM: {args.model_name}")
    print(f"Using {args.num_gpus} GPU(s)")
    
    llm = LLM(
        model_name=args.model_name,
        sys_prompt=args.sys_prompt,
        n_gpus=args.num_gpus,
        dtype=torch.bfloat16,
        uncertainty_estim=args.use_uncertainty,
        uncertainty_cfg={
            "h_star": args.h_star,
            "isr_threshold": args.isr_threshold,
            "m": args.m,
            "n_samples": args.n_samples,
            "B_clip": args.b_clip,
            "clip_mode": "one-sided",
            "skeleton_policy": args.skeleton_policy,
            "temperature": args.temperature,
            "max_tokens_decision": 8,
            "backend": args.backend,
            "mask_refusals_in_loss": True,
        } if args.use_uncertainty else None,
        decision_backend_kwargs={} if args.use_uncertainty else None,
    )
    
    print(f"Uncertainty enabled: {llm.uncertainty_estim}")
    
    # Create GRetriever (no GNN for this simple test)
    model = GRetriever(llm=llm, gnn=None, use_lora=False)
    
    # Filter out impossible questions
    valid_data = [item for item in data if not item.get("is_impossible", False)]
    test_data = valid_data[:args.num_questions]
    
    print(f"\nTesting on {len(test_data)} questions...")
    print("=" * 80)
    
    results = []
    
    for item in tqdm(test_data, desc="Processing questions"):
        question = item["question"]
        answer = item["answer"]
        
        # Format question with mock context (in real scenario, use actual retrieved docs)
        if args.use_mock_context:
            context = f"Technical documentation: {answer[:args.context_length]}"
        else:
            context = "No context provided (zero-shot)"
        
        formatted_q = f"""
[QUESTION]
{question}
[END_QUESTION]

[RETRIEVED_CONTEXTS]
{context}
[END_RETRIEVED_CONTEXTS]
"""
        
        # Run inference
        result = model.llm.inference(
            question=[formatted_q],
            context=[""],
            max_tokens=args.max_tokens,
            return_uncertainty=args.use_uncertainty,
            abstain_on_low_ISR=args.abstain_on_low_isr,
        )
        
        # Parse result
        if args.use_uncertainty and isinstance(result, tuple):
            texts, uncertainties = result
            generated = texts[0] if texts else ""
            uncertainty = uncertainties[0] if uncertainties else None
            
            result_dict = {
                "question": question[:100],
                "expected": answer[:100],
                "generated": generated[:100],
                "isr": uncertainty.isr if uncertainty else None,
                "decision": "ANSWER" if (uncertainty and uncertainty.decision_answer) else "REFUSE",
                "full_question": question,
                "full_answer": answer,
                "full_generated": generated,
            }
        else:
            generated = result[0] if isinstance(result, list) else str(result)
            result_dict = {
                "question": question[:100],
                "expected": answer[:100],
                "generated": generated[:100],
                "full_question": question,
                "full_answer": answer,
                "full_generated": generated,
            }
        
        results.append(result_dict)
        
        if args.verbose:
            print(f"\nQ: {result_dict['question']}")
            print(f"Expected: {result_dict['expected']}")
            print(f"Generated: {result_dict['generated']}")
            if args.use_uncertainty:
                print(f"ISR: {result_dict['isr']:.3f}, Decision: {result_dict['decision']}")
            print("-" * 80)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if args.use_uncertainty:
        isr_values = [r['isr'] for r in results if r['isr'] is not None]
        answers = [r for r in results if r['decision'] == 'ANSWER']
        refusals = [r for r in results if r['decision'] == 'REFUSE']
        
        print(f"Total questions: {len(results)}")
        print(f"Answered: {len(answers)} ({len(answers)/len(results)*100:.1f}%)")
        print(f"Refused: {len(refusals)} ({len(refusals)/len(results)*100:.1f}%)")
        print(f"Average ISR: {sum(isr_values)/len(isr_values):.3f}")
        print(f"Min ISR: {min(isr_values):.3f}")
        print(f"Max ISR: {max(isr_values):.3f}")
    else:
        print(f"Total questions processed: {len(results)}")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM with uncertainty estimation on TechQA dataset"
    )
    
    # Dataset args
    parser.add_argument('--dataset_dir', type=str, default='techqa',
                        help='Directory to store/load TechQA dataset')
    parser.add_argument('--num_questions', type=int, default=10,
                        help='Number of questions to test')
    
    # Model args
    parser.add_argument('--model_name', type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='HuggingFace model name')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--sys_prompt', type=str,
                        default='Answer questions based on the provided context.',
                        help='System prompt for the LLM')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum tokens to generate')
    
    # Uncertainty args
    parser.add_argument('--use_uncertainty', action='store_true',
                        help='Enable uncertainty estimation')
    parser.add_argument('--h_star', type=float, default=0.05,
                        help='Entropy threshold for uncertainty')
    parser.add_argument('--isr_threshold', type=float, default=1.0,
                        help='ISR threshold for abstention')
    parser.add_argument('--m', type=int, default=6,
                        help='Number of semantic sets')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of samples per semantic set')
    parser.add_argument('--b_clip', type=float, default=12.0,
                        help='Clipping value for importance sampling')
    parser.add_argument('--skeleton_policy', type=str, default='auto',
                        choices=['auto', 'evidence_erase', 'none'],
                        help='Skeleton generation policy')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature')
    parser.add_argument('--backend', type=str, default='hf',
                        choices=['hf', 'ollama', 'anthropic'],
                        help='Backend for decision model')
    parser.add_argument('--abstain_on_low_isr', action='store_true',
                        help='Abstain when ISR is below threshold')
    
    # Context args
    parser.add_argument('--use_mock_context', action='store_true',
                        help='Use partial answer as mock context (for testing)')
    parser.add_argument('--context_length', type=int, default=100,
                        help='Length of mock context to use')
    
    # Output args
    parser.add_argument('--output_file', type=str, default=None,
                        help='JSON file to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each question')
    
    args = parser.parse_args()
    
    # Run test
    results = test_uncertainty_rag(args)
    
    print("\nTest complete!")


if __name__ == '__main__':
    main()