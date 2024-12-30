from torch import nn, Tensor as T
import torch
import math
from transformers import AutoTokenizer, AutoModel, logging as hf_logger
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Union, Tuple, Type

class LongTokenizer:
    def __init__(self,
                 tokenizer_name_or_path: str,
                 max_document_length: Optional[int] = None, #length to which the input document will be truncated before tokenization. Default to None (no truncation).
                 max_chunk_length: Optional[int] = None, #maximum length of the chunks. Default to None (model max length).
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=False)
        self.max_document_length = float('inf') if max_document_length is None else max_document_length
        self.max_chunk_length = self.tokenizer.model_max_length if max_chunk_length is None else max_chunk_length


    def __call__(self, texts: Union[str, List[str]]) -> Tuple[T, T]:
        """
        Args:
            texts: a list of (short or long) texts to tokenize.
        Returns:
            token_ids: 3D tensor of size [batch_size, max_chunks, max_chunk_length].
            attention masks: 3D tensor of size [batch_size, max_chunks, max_chunk_length].
        """
        # Tokenize inputs texts: (i) without truncating if max_document_length is inf; (ii) by truncating to max_document_length otherwise.
        # NB: not truncating long texts might raise an OOM error during training if too little GPU memory.
        if np.isinf(self.max_document_length):
            tokenized = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        else:
            tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_document_length, return_tensors="pt")

        # Split long tokenized texts into smaller chunks (only if max_chunk_length is smaller than max_doc_length).
        if self.max_document_length > self.max_chunk_length:
            return self.__chunk_tokenized_inputs(tokenized['input_ids'], tokenized['attention_mask'])
        else:
            return tokenized['input_ids'], tokenized['attention_mask']


    def __chunk_tokenized_inputs(self, token_ids: T, attention_masks: T) -> Tuple[T,T]:
        """Chunk the tokenized inputs returned by HuggingFace tokenizer into fixed-lengths units.
        Args:
            token_ids: 2D tensor of size [batch_size, batch_max_seq_len].
            attention_masks: 2D tensor of size [batch_size, batch_max_seq_len].
        Returns:
            token_ids: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            attention_masks: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
        """
        batch_size, batch_max_seq_len = token_ids.shape
        window_size = self.max_chunk_length // 10 #sliding window with 10% overlap.

        if batch_max_seq_len <= self.max_chunk_length:
            # If the max sequence length from the current batch is smaller than the defined chunk size, simply return the tensors with a dim of 1 on the 'batch_max_chunks' dimension.
            return token_ids.unsqueeze(1), attention_masks.unsqueeze(1)
        else:
            # Remove first column from 2D tensor (corresponding to the CLS tokens of the long sequences).
            token_ids = token_ids[:, 1:] #T[batch_size, batch_max_seq_len-1]
            attention_masks = attention_masks[:, 1:] #T[batch_size, batch_max_seq_len-1]
            batch_max_seq_len -= 1
            max_chunk_length = self.max_chunk_length - 1

            # Pad 2D tensor so that the 'batch_seq_len' is a multiple of 'max_chunk_length' (otherwise unfold ignore remaining tokens).
            num_windows = math.floor((batch_max_seq_len - max_chunk_length)/(max_chunk_length - window_size))
            num_repeated_tokens = num_windows * window_size
            batch_seq_len = math.ceil((batch_max_seq_len + num_repeated_tokens)/max_chunk_length) * max_chunk_length
            token_ids = nn.functional.pad(input=token_ids, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=self.tokenizer.pad_token_id)
            attention_masks = nn.functional.pad(input=attention_masks, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=0)

            # Split tensor along y-axis (i.e., along the 'batch_max_seq_len' dimension) with overlapping of 'window_size'
            # and create a new 3D tensor of size [batch_size, max_chunk_length-1, batch_max_seq_len/max_chunk_length].
            token_ids = token_ids.unfold(dimension=1, size=max_chunk_length, step=max_chunk_length-window_size)
            attention_masks = attention_masks.unfold(dimension=1, size=max_chunk_length, step=max_chunk_length-window_size)
            batch_max_chunks = token_ids.size(1)

            # Append CLS token id before each chunk if the latter does not start with a PAD token. If so, append the PAD token id instead.
            cls_token_ids = torch.full((batch_size, batch_max_chunks, 1), self.tokenizer.cls_token_id)
            cls_token_ids[token_ids[:,:,0].unsqueeze(2) == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            token_ids = torch.cat([cls_token_ids, token_ids], axis=2)

            # Add attention masks of 1 for the new CLS tokens, and masks of 0 for the new PAD tokens.
            cls_attention_masks = torch.ones(batch_size, batch_max_chunks, 1)
            cls_attention_masks[cls_token_ids == self.tokenizer.pad_token_id] = 0
            attention_masks = torch.cat([cls_attention_masks, attention_masks], axis=2)

            # Return tokens and masks.
            return token_ids, attention_masks

class Pooling(nn.Module):
    """Performs pooling (mean or max) on the token embeddings.
    Using pooling, it generates from a variable sized text passage a fixed sized passage embedding.
    """
    def __init__(self, pooling_mode: str):
        super(Pooling, self).__init__()
        assert pooling_mode in ['mean', 'max'], f"ERROR: Unknown pooling strategy '{pooling_mode}'"
        self.pooling_mode = pooling_mode

    def forward(self, token_embeddings: T, attention_masks: T) -> T:
        """
        Args:
            token_embeddings: 3D tensor of size [batch_size, seq_len, embedding_dim].
            attention_masks: 2D tensor of size [batch_size, seq_len].
        Returns:
            text_vectors: 2D tensor of size [batch_size, embedding_dim].
        """
        if self.pooling_mode == 'max':
            # Set all values of the [PAD] embeddings to large negative values (so that they are never considered as maximum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = -1e+9 if token_embeddings.dtype == torch.float32 else -1e+4
            # Compute the maxima along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            text_vectors = torch.max(token_embeddings, dim=1).values
        else:
            # Set all values of the [PAD] embeddings to zeros (so that they are not taken into account in the sum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = 0.0
            # Compute the means by first summing along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            # Then, divide all values of a passage vector by the original passage length.
            sum_mask = attention_masks_expanded.sum(dim=1) # -> Tensor[batch_size, embedding_dim] where each value is the length of the corresponding passage.
            sum_mask = torch.clamp(sum_mask, min=1e-7) # Make sure not to have zeros by lower bounding all elements to 1e-7.
            text_vectors = sum_embeddings / sum_mask # Divide each dimension by the sequence length.
        return text_vectors

class HierBERT(nn.Module):
    def __init__(self, model_name_or_path: str, pooling_mode: str, n_pos_chunks: int):
        super(HierBERT, self).__init__()
        # Word-wise BERT-based encoder.
        self.word_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dim = self.word_encoder.config.hidden_size

        # Chunk-wise sinusoidal positional embeddings.
        self.chunk_pos_embeddings = nn.Embedding(num_embeddings=n_pos_chunks+1,
                                                 embedding_dim=self.dim,
                                                 padding_idx=0,
                                                 _weight=HierBERT.sinusoidal_init(n_pos_chunks+1, self.dim, 0))
        # Chunk-wise Transformer-based encoder.
        self.chunk_encoder = nn.Transformer(
            num_encoder_layers=2,
            num_decoder_layers=0,
            d_model=self.dim,               #self.word_encoder.config.hidden_size
            dim_feedforward=self.dim*4,     #self.word_encoder.config.intermediate_size,
            nhead=12,                       #self.word_encoder.config.num_attention_heads,
            activation='gelu',              #self.word_encoder.config.hidden_act,
            dropout=0.1,                    #self.word_encoder.config.hidden_dropout_prob,
            layer_norm_eps=1e-5,            #self.word_encoder.config.layer_norm_eps,
            batch_first=True).encoder

        # Pooling layer.
        self.pooler = Pooling(pooling_mode)

    def forward(self, token_ids: T, attention_masks: T, input_ids: Optional[T] = None) -> Tuple[T, T, T]:
        """
        Args:
            token_ids: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            attention masks: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            input_ids: 1D tensor of size [batch_size]. Optional: unique IDs of input texts.
        Returns:
            text_representations: 2D tensor of size [batch_size, embedding_dim].
        """
        # Reshape to 2D tensor of size [batch_size x batch_max_chunks, max_chunk_length].
        token_ids_reshaped = token_ids.contiguous().view(-1, token_ids.size(-1))
        attention_masks_reshaped = attention_masks.contiguous().view(-1, attention_masks.size(-1))

        # Compute context-aware word embeddings with the BERT-based encoder.
        token_embeddings = self.word_encoder(input_ids=token_ids_reshaped, attention_mask=attention_masks_reshaped, return_dict=False)[0]

        # Reshape back to 4D tensor of size [batch_size, batch_max_chunks, max_chunk_length, hidden_size]
        token_embeddings = token_embeddings.contiguous().view(*tuple(token_ids.size()), self.dim)

        # Keep only [CLS] embeddings (and corresponding attention masks) for each chunk in the batch.
        chunk_embeddings = token_embeddings[:,:,0,:] #-> 3D tensor of size [batch_size, batch_max_chunks, hidden_size]
        chunk_masks = attention_masks[:,:,0] #-> 2D tensor of size [batch_size, batch_max_chunks]

        # Compute chunk positional embeddings and add them to the chunk embeddings.
        chunk_positions = torch.tensor(range(1, token_ids.size(1)+1), dtype=torch.int, device=token_ids.device) * chunk_masks.int()
        chunk_embeddings += self.chunk_pos_embeddings(chunk_positions)

        # Compute context-aware chunk embeddings with the Transformer-based encoder.
        chunk_embeddings = self.chunk_encoder(chunk_embeddings)

        # Pool the context-aware chunk embeddings to distill a global representation for each document in the batch.
        doc_embeddings = self.pooler(chunk_embeddings, chunk_masks)
        return doc_embeddings

    @staticmethod
    def sinusoidal_init(num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        """https://github.com/pytorch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py#L36
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1) #zero pad
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    

def encode(model: Type[torch.nn.Module], #
           tokenizer: Type[LongTokenizer], #
           texts: Union[str, List[str]], #
           batch_size: int, #
           text_ids: Optional[Union[str, List[str]]] = None, #
           show_progress: bool = True, #
           device: str = 'cuda' if torch.cuda.is_available() else 'cpu', #
    ):
    if isinstance(texts, str):
        texts = [texts]
    if text_ids is not None:
        if isinstance(text_ids, str):
            text_ids = [text_ids]
        assert len(texts) == len(text_ids), "Length of 'text_ids' doesn't match length of 'texts'."

    # Sort texts by length to get batches of similar lengths.
    length_sorted_idx = np.argsort([-len(t) for t in texts])
    texts_sorted = [texts[idx] for idx in length_sorted_idx]
    if text_ids is not None:
        ids_sorted = [text_ids[idx] for idx in length_sorted_idx]

    model.eval()
    model.to(device)
    all_embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size), desc=f"- Encoding batches of {batch_size} docs", disable=not show_progress, leave=False):
        if text_ids is not None:
            # Get ids of batch of documents.
            ids_batch = ids_sorted[start_idx:start_idx+batch_size]
            ids_batch = torch.tensor(ids_batch, dtype=torch.int)

        # Tokenize batch of long documents.
        texts_batch = texts_sorted[start_idx:start_idx+batch_size]
        token_ids, attention_masks = tokenizer(texts_batch)

        # Send tensors to device.
        token_ids = token_ids.to(device)
        attention_masks = attention_masks.to(device)
        if text_ids is not None:
            ids_batch = ids_batch.to(device)

        # Encode.
        with torch.no_grad():
            if text_ids is not None:
                embeddings = model(token_ids, attention_masks, ids_batch)
            else:
                embeddings = model(token_ids, attention_masks)
            all_embeddings.extend(embeddings)

    # Sort the embeddings back in the original order of the input docs and returns torch tensor.
    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    return torch.stack(all_embeddings).detach().cpu()