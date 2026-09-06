"""
backend/model_service.py
=========================
Loads all 6 trained INTERVUE models and provides inference for:
  - Question generation (interview_tuned.pt)
  - Answer evaluation (evaluator.pt)
  - Follow-up generation (final_model.pt)

Uses the 16K tokenizer from tokenizer/saved/tokenizer.json.
All models run on CPU by default; set device="cuda" for GPU.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.generator.model import GeneratorConfig, create_large_model, InterviewGenerator
from models.generator.train_utils import load_tokenizer


class ModelService:
    """Singleton service that lazily loads all 6 trained models."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._models = {}  # stage_name -> InterviewGenerator
        self._loaded = False

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tok_path = ROOT / "tokenizer" / "saved" / "tokenizer.json"
            if not tok_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer not found at {tok_path}. "
                    "Train it first: python tokenizer/train_tokenizer.py"
                )
            self._tokenizer = load_tokenizer(tok_path)
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _load_model(self, stage_name: str, ckpt_filename: str) -> InterviewGenerator:
        """Load a single model from checkpoint."""
        ckpt_path = ROOT / "models" / "generator" / "saved" / ckpt_filename
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Run training first: python models/generator/train.py --stage all"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        # Strip module. prefix from DataParallel checkpoints
        if any(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        # Build model from saved config
        cfg = ckpt.get("config", {})
        if cfg and "embed_dim" in cfg:
            config = GeneratorConfig(**cfg)
            model = InterviewGenerator(config)
        else:
            model = create_large_model(vocab_size=cfg.get("vocab_size", self.vocab_size))
        model.load_state_dict(sd)
        model.to(self.device)
        model.eval()

        print(f"  Loaded {stage_name}: {ckpt_filename} "
              f"(epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.4f})")
        return model

    STAGE_FILES = {
        "pretrain": "pretrained.pt",
        "domain": "domain_tuned.pt",
        "instruction": "instruction_tuned.pt",
        "interview": "interview_tuned.pt",
        "evaluator": "evaluator.pt",
        "followup": "final_model.pt",
    }

    def load_all(self):
        """Load all models. Safe to call once at startup."""
        if self._loaded:
            return

        print("=" * 50)
        print("  Loading INTERVUE models...")
        print("=" * 50)

        for name, filename in self.STAGE_FILES.items():
            try:
                self._models[name] = self._load_model(name, filename)
            except Exception as e:
                print(f"  WARN: {e}")

        self._loaded = True
        print(f"  Loaded {len(self._models)}/{len(self.STAGE_FILES)} models on {self.device}")
        print("=" * 50)

    def get_model(self, stage: str) -> Optional[InterviewGenerator]:
        """Get or lazily load a specific model by stage name."""
        if stage in self._models:
            return self._models[stage]

        filename = self.STAGE_FILES.get(stage)
        if not filename:
            return None

        try:
            model = self._load_model(stage, filename)
            self._models[stage] = model
            return model
        except Exception as e:
            print(f"  [ModelService] Lazy load skipped for stage '{stage}': {e}")
            return None

    def _format_prompt(self, system: str = None, user: str = None,
                       assistant_prefix: str = None) -> str:
        """Format a prompt with special tokens."""
        parts = []
        if system:
            parts.append(f"<|system|> {system}<|end|>")
        if user:
            parts.append(f"<|user|> {user}<|end|>")
        if assistant_prefix:
            parts.append(f"<|assistant|> {assistant_prefix}")
        else:
            parts.append("<|assistant|>")
        return "".join(parts)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text into a tensor of token IDs."""
        ids = self.tokenizer.encode(text).ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text, removing special tokens and prompt."""
        # Remove batch dimension if present
        if token_ids.dim() > 1:
            token_ids = token_ids[0]
        ids = token_ids.tolist()

        # Get special token IDs
        stop_ids = set()
        for tok in ["[SEP]", "<|end|>", "[PAD]", "[UNK]"]:
            tid = self.tokenizer.token_to_id(tok)
            if tid is not None:
                stop_ids.add(tid)

        # Find the last <|assistant|> token to extract response only
        assistant_id = self.tokenizer.token_to_id("<|assistant|>")
        last_assistant_pos = -1
        for i, tid in enumerate(ids):
            if tid == assistant_id:
                last_assistant_pos = i

        # Extract tokens after the last <|assistant|>
        if last_assistant_pos >= 0:
            response_ids = ids[last_assistant_pos + 1:]
        else:
            response_ids = ids

        # Stop at end tokens
        trimmed = []
        for tid in response_ids:
            if tid in stop_ids:
                break
            trimmed.append(tid)

        text = self.tokenizer.decode(trimmed)
        # Fix byte-level BPE encoding artifacts
        text = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
        return text

    @torch.no_grad()
    def generate(self, stage: str, prompt: str,
                 max_new_tokens: int = 50,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50) -> str:
        """Generate text from a loaded model."""
        model = self.get_model(stage)
        if model is None:
            raise RuntimeError(f"Model '{stage}' not loaded")

        input_ids = self._tokenize(prompt)
        eos_id = self.tokenizer.token_to_id("<|end|>")
        if eos_id is None:
            eos_id = self.tokenizer.token_to_id("[SEP]")

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
        )
        return self._decode(output_ids)

    @torch.no_grad()
    def evaluate(self, prompt: str, answer: str) -> dict:
        """Use the evaluator model to score an answer."""
        model = self.get_model("evaluator")
        if model is None:
            raise RuntimeError("Evaluator model not loaded")

        full_text = self._format_prompt(
            system="You are an expert technical interviewer evaluating a candidate's answer.",
            user=f"Question: {prompt}\nCandidate Answer: {answer}\n\nEvaluate this answer on a scale of 0-100. "
                 "Consider: correctness, completeness, clarity, and depth. "
                 "Provide your score and brief feedback.",
        )
        input_ids = self._tokenize(full_text)
        eos_id = self.tokenizer.token_to_id("<|end|>")
        if eos_id is None:
            eos_id = self.tokenizer.token_to_id("[SEP]")

        output_ids = model.generate(
            input_ids,
            max_new_tokens=35,
            temperature=0.3,  # Lower temperature for more deterministic scoring
            top_p=0.8,
            top_k=30,
            eos_token_id=eos_id,
        )
        response = self._decode(output_ids)

        # Try to extract numeric score from response
        import re
        score_match = re.search(r'(\d{1,3})', response)
        score = int(score_match.group(1)) if score_match else 50
        score = max(0, min(100, score))

        return {
            "score": score,
            "feedback": response,
            "model_used": "evaluator",
        }


# Convenience singleton accessor
_service = None

def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
