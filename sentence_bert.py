import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Union
from tqdm import tqdm


class SentenceBERT:
    def __init__(self, model_name: str, pooling_mode: str = 'mean', device: str = None):
        """
        Initializes the SentenceBERT model with specified pooling.

        Args:
            model_name (str): Name of the pre-trained BERT model.
            pooling_mode (str): Pooling strategy. Supported options:
                                - 'mean': Mean Pooling
                                - 'cls': [CLS] Token Pooling
                                - 'mean_cls': Concatenation of Mean and [CLS] Pooling
            device (str): Device to run the model on ('cpu' or 'cuda'). If None, automatically selects.
        """
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Define pooling mode
        supported_pooling = ['mean', 'cls', 'mean_cls']
        if pooling_mode not in supported_pooling:
            raise ValueError(f"Pooling mode '{pooling_mode}' not supported. Choose from {supported_pooling}.")
        self.pooling_mode = pooling_mode

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling to the token embeddings.

        Args:
            last_hidden_state (torch.Tensor): Hidden states from BERT (batch_size, seq_length, hidden_size).
            attention_mask (torch.Tensor): Attention masks indicating non-padded tokens (batch_size, seq_length).

        Returns:
            torch.Tensor: Mean-pooled sentence embeddings (batch_size, hidden_size).
        """
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum the embeddings
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        
        # Sum the mask to get the number of valid tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Compute mean embeddings
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings

    def cls_pooling(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Applies [CLS] token pooling to the token embeddings.

        Args:
            last_hidden_state (torch.Tensor): Hidden states from BERT (batch_size, seq_length, hidden_size).

        Returns:
            torch.Tensor: [CLS] token embeddings (batch_size, hidden_size).
        """
        cls_embeddings = last_hidden_state[:, 0, :]  # [CLS] token is the first token
        return cls_embeddings

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False) -> torch.Tensor:
        """
        Encodes input texts into sentence embeddings based on the selected pooling strategy.

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): If True, displays a progress bar.

        Returns:
            torch.Tensor: Sentence embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        total = len(texts)
        if show_progress_bar:
            iterator = tqdm(range(0, total, batch_size), desc="Encoding")
        else:
            iterator = range(0, total, batch_size)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                encoded_inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                
                input_ids = encoded_inputs['input_ids'].to(self.device)
                attention_mask = encoded_inputs['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
                
                # Apply pooling based on the selected mode
                if self.pooling_mode == 'mean':
                    batch_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
                elif self.pooling_mode == 'cls':
                    batch_embeddings = self.cls_pooling(last_hidden_state)
                elif self.pooling_mode == 'mean_cls':
                    mean_emb = self.mean_pooling(last_hidden_state, attention_mask)
                    cls_emb = self.cls_pooling(last_hidden_state)
                    batch_embeddings = torch.cat((mean_emb, cls_emb), dim=1)  # Concatenate along the hidden_size dimension
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")
                
                embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all batch embeddings
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings
    

# class SentenceBERT:

#     def __init__(self, model_name, pooling, device):

#         transformer = models.Transformer(model_name)
#         pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling)
#         self.model = SentenceTransformer(modules=[transformer, pooling], device=device)

#     def encode(self, texts):

#         if isinstance(texts, str):
#             texts = [texts]

#         embeddings = self.model.encode(list(texts), show_progress_bar=True)

#         return embeddings
    
