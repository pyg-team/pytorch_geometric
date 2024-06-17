import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tape.config import DatasetName, FeatureType
from tape.dataset.utils import generate_string_hash
from tqdm import tqdm

warnings.filterwarnings('ignore')  # Ignore HuggingFace libraries warnings


@dataclass
class TransformersTokenizerArgs:
    batch_size: int = 32
    truncation: bool = True
    padding: bool = True
    max_length: int = 512


@dataclass
class SentenceTransformerArgs:
    batch_size: int = 32
    show_progress_bar: bool = True
    precision: Literal['float32', 'int8', 'uint8', 'binary',
                       'ubinary'] = 'float32'


@dataclass
class LmEncoderArgs:
    dataset_name: DatasetName  # Used for creating file name to save embeddings
    feature_type: FeatureType  # Used for creating file name to save embeddings
    model_name_or_path: str
    model_library: Literal['transformers', 'sentence_transformer']
    transformers_encoder_args: Optional[TransformersTokenizerArgs] = None
    sentence_transformer_encoder_args: Optional[SentenceTransformerArgs] = None
    device: Optional[str] = None
    cache_dir: str = '.cache'


class LmEncoder:
    """Language model article encoder."""
    def __init__(self, args: LmEncoderArgs) -> None:
        self.args = args
        self.device = args.device or ('cuda'
                                      if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        cache_dir = Path.cwd() / args.cache_dir

        embd_cache_dir = cache_dir / 'embeddings'
        embd_cache_dir.mkdir(exist_ok=True, parents=True)
        file_name = (
            f'{args.feature_type.value}_{args.dataset_name.value}'
            f'_{args.model_name_or_path.replace("/", "--")}.safetensors')
        self.embd_cache_path = embd_cache_dir / file_name
        self._sent_hash_to_embedding = self._load_cache()

        if args.model_library == 'transformers':
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(
                args.model_name_or_path, cache_dir=cache_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, cache_dir=cache_dir)
        elif args.model_library == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(args.model_name_or_path,
                                             device=self.device,
                                             cache_folder=cache_dir)
        else:
            raise Exception('Invalid model library!')

    def _load_cache(self):
        input_hash_to_embedding = {}
        if self.embd_cache_path.exists():
            print('Loading cached embeddings...')
            with safe_open(str(self.embd_cache_path), framework="pt",
                           device=self.device) as f:
                for k in f.keys():
                    input_hash_to_embedding[k] = f.get_tensor(k)
        return input_hash_to_embedding

    def save_cache(self) -> None:
        save_file(self._sent_hash_to_embedding, str(self.embd_cache_path))
        print(f'Saved embedding file to "{self.embd_cache_path}"')

    @torch.inference_mode()
    def _hf_encoder(self, sentences: List[str], **kwargs):
        encoded_sentences = self.tokenizer(
            sentences,
            truncation=kwargs.get('truncation', True),
            padding=kwargs.get('padding', True),
            return_tensors='pt',
            max_length=kwargs.get('max_length', 512),
        ).to(self.device)
        # Encode the queries (use the [CLS] last hidden states
        # as the representations)
        embeddings = self.model(**encoded_sentences).last_hidden_state[:, 0, :]
        torch.cuda.empty_cache()
        return embeddings

    def _get_embeddings(self, sentences: List[str], **kwargs):
        if self.args.model_library == 'transformers':
            _kwargs = asdict(self.args.transformers_encoder_args)
            _kwargs.update(kwargs)  # kwargs overrides the default config
            batch_size = _kwargs['batch_size']
            embeddings = []
            for step in tqdm(range(0, len(sentences), batch_size),
                             total=len(sentences) // batch_size,
                             desc='Batches'):
                embeddings.append(
                    self._hf_encoder(
                        sentences=sentences[step:step + batch_size],
                        **_kwargs))
            embeddings = torch.cat(embeddings)
        elif self.args.model_library == 'sentence_transformer':
            _kwargs = asdict(self.args.sentence_transformer_encoder_args)
            _kwargs.update(kwargs)  # kwargs overrides the default config
            _kwargs.pop('convert_to_tensor', None)
            embeddings = self.model.encode(sentences, convert_to_tensor=True,
                                           **_kwargs)
        return embeddings

    def __call__(self, sentences: List[str], **kwargs) -> torch.Tensor:
        missing_sentences_idxs = []
        embeddings = []
        for idx, sent_hash in enumerate(map(generate_string_hash, sentences)):
            if (embd := self._sent_hash_to_embedding.get(sent_hash)) is None:
                missing_sentences_idxs.append((idx, sent_hash))
            else:
                sentences[idx] = None
            embeddings.append(embd)
        if missing_sentences_idxs:
            missing_embeddings = self._get_embeddings(
                sentences=list(filter(None, sentences)), **kwargs)
            for (idx, sent_hash), embedding in zip(missing_sentences_idxs,
                                                   missing_embeddings):
                embeddings[idx] = embedding
                self._sent_hash_to_embedding[sent_hash] = embedding
        return embeddings
