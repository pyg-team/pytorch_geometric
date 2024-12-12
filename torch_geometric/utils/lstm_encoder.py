import torch
import torch.nn.functional as F
import torch.nn as nn

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

def get_dict(data_folder, dict_file):
    """
    Load a word-to-ID dictionary from a specified file.

    This function expects a file in the given path (data_folder + dict_file) where each line
    contains a word and an integer ID, separated by whitespace. It parses the file and returns
    a dictionary mapping each word to its associated integer ID.

    Parameters
    ----------
    data_folder : str
        The path to the folder containing the dictionary file.
    dict_file : str
        The name of the dictionary file to load.

    Returns
    -------
    dict
        A dictionary mapping words (str) to their integer IDs (int).

    Example
    -------
    Suppose we have a file `data_folder/vocab.txt`:
    ```
    hello 0
    world 1
    ```
    Calling `get_dict("data_folder/", "vocab.txt")` will return:
    `{"hello": 0, "world": 1}`
    """
    if dict_file is None:
        return {}
    path = data_folder + dict_file
    word2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, idx = parts
                word2id[word] = int(idx)
    return word2id


class LSTMInstruction(nn.Module):
    """
    A neural network module that encodes input queries using an LSTM and 
    generates a series of relational instructions (contextual embeddings) 
    through multiple reasoning steps.

    Parameters
    ----------
    args : dict
        A dictionary of arguments including keys like:
        - 'use_cuda': bool, whether to use GPU
        - 'q_type': some query type identifier
        - 'lm_dropout': float, dropout probability for LSTM layers
        - 'linear_dropout': float, dropout probability for linear layers
        - 'data_folder': str, path to the data directory
        - 'word2id': str, filename of word-to-id mapping
        - 'num_step' / 'num_ins' / 'num_layer' or 'num_expansion_ins'/'num_backup_ins': 
           integers specifying the number of reasoning steps
        - 'word_dim', 'entity_dim': dimensions for embeddings
    constraint : bool
        A boolean that decides which type of instruction layer count to use.
    word_embedding : nn.Embedding or callable
        A word embedding module used to embed tokenized words into vectors.
    num_word : int
        A sentinel ID indicating the 'padding' or OOV token index. 

    Notes
    -----
    This model:
    1. Encodes the input query text using an LSTM-based encoder.
    2. Iteratively refines a relational instruction embedding over several steps
       by attending over the query representation.
    """

    def __init__(self, args, constraint, word_embedding=None, num_word=None):
        super(LSTMInstruction, self).__init__()
        self.constraint = constraint
        self._parse_args(args)
        self.share_module_def()

        self.word2id = get_dict(args['data_folder'], args['word2id'])
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def _parse_args(self, args):
        """
        Parse arguments and initialize attributes from a given dictionary.

        Parameters
        ----------
        args : dict
            Dictionary containing all the initialization arguments.

        Notes
        -----
        Updates class attributes such as device, q_type, dropout probabilities, 
        and entity/word dimensions based on provided args.
        """
        self.device = torch.device('cuda' if args.get('use_cuda', False) else 'cpu')
        
        self.q_type = args['q_type']
        if 'num_step' in args:
            self.num_ins = args['num_step']
        elif 'num_ins' in args:
            self.num_ins = args['num_ins']
        elif 'num_layer' in args:
            self.num_ins = args['num_layer']
        elif 'num_expansion_ins' in args and 'num_backup_ins' in args:
            self.num_ins = args['num_backup_ins'] if self.constraint else args['num_expansion_ins']
        else:
            self.num_ins = 1
        
        self.lm_dropout = args['lm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.lm_frozen = args.get('lm_frozen', False)

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    def share_module_def(self):
        """
        Define shared layers or dropout modules used across multiple steps.

        Notes
        -----
        This initializes dropout layers for LSTM outputs and linear transformations.
        """
        self.lstm_drop = nn.Dropout(p=self.lm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

    def init_hidden(self, num_layer, batch_size, hidden_size):
        """
        Initialize hidden states for the LSTM encoder.

        Parameters
        ----------
        num_layer : int
            Number of layers in the LSTM.
        batch_size : int
            Size of the input batch.
        hidden_size : int
            Dimensionality of the hidden state in the LSTM.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            A tuple (h_0, c_0) representing the initial hidden and cell states 
            for the LSTM. Both are zero tensors on the specified device.
        """
        return (torch.zeros(num_layer, batch_size, hidden_size).to(self.device),
                torch.zeros(num_layer, batch_size, hidden_size).to(self.device))

    def encode_question(self, query_text, store=True):
        """
        Encode the query using the LSTM encoder.

        Parameters
        ----------
        query_text : torch.Tensor
            A tensor of shape (batch_size, max_query_word) containing 
            token indices for the query.
        store : bool, optional (default=True)
            If True, stores the encoded states and node embeddings as class attributes 
            for later usage.

        Returns
        -------
        (torch.Tensor, torch.Tensor) or torch.Tensor
            If store=True, returns (query_hidden_emb, query_node_emb).
            If store=False, returns query_hidden_emb only.

        Notes
        -----
        - query_hidden_emb is of shape (batch_size, max_query_word, entity_dim).
        - query_node_emb is of shape (batch_size, 1, entity_dim) 
          and is derived from the last hidden state of the LSTM.
        """
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(
            self.lstm_drop(query_word_emb),
            self.init_hidden(1, batch_size, self.entity_dim)
        )
        if store:
            self.instruction_hidden = h_n
            self.instruction_mem = c_n
            self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
            self.query_hidden_emb = query_hidden_emb
            self.query_mask = (query_text != self.num_word).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return query_hidden_emb

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        """
        Retrieve a single node embedding from the query embeddings.

        Parameters
        ----------
        query_hidden_emb : torch.Tensor
            A tensor of shape (batch_size, max_hyper, emb) containing 
            per-token/node embeddings.
        action : torch.Tensor
            A tensor of shape (batch_size,) specifying the index of the node
            to extract from each example.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, 1, emb) containing the selected node embeddings.

        Notes
        -----
        This function selects from query_hidden_emb the embeddings corresponding 
        to the node indices given by 'action'.
        """
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep.unsqueeze(1)

    def init_reason(self, query_text):
        """
        Initialize reasoning structures for the relational instructions.

        Parameters
        ----------
        query_text : torch.Tensor
            A tensor containing the token indices of the query.

        Notes
        -----
        - Encodes the question and sets up placeholders for relational instructions 
          and attention weights.
        - Initializes a relational instruction vector to zeros.
        """
        self.batch_size = query_text.size(0)
        self.max_query_word = query_text.size(1)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(self.batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        """
        Perform a single reasoning step to update the relational instruction embedding.

        Parameters
        ----------
        relational_ins : torch.Tensor
            A tensor of shape (batch_size, entity_dim) representing the current relational instruction embedding.
        step : int, optional (default=0)
            The current reasoning step index.
        query_node_emb : torch.Tensor, optional (default=None)
            A tensor of shape (batch_size, 1, entity_dim) representing the query-level node embedding. 
            If None, uses the stored query_node_emb from encode_question.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Updated relational_ins: (batch_size, entity_dim)
            attn_weight: (batch_size, max_query_word, 1)
            The updated relational instruction embedding and attention weights over the query.

        Notes
        -----
        - Applies linear transformations and attention to update the relational instruction vector.
        - Uses a learned attention mechanism over query_hidden_emb.
        """
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(
            self.linear_drop(
                torch.cat((relational_ins, q_i, q_i - relational_ins, q_i * relational_ins), dim=-1)
            )
        )
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight

    def encoder_def(self):
        """
        Define the LSTM encoder used to encode the query.

        Notes
        -----
        Initializes an LSTM for encoding query text into hidden states. 
        Uses word_dim as input size and entity_dim as hidden size.
        """
        word_dim = self.word_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)

    def forward(self, query_text, lm=None):
        """
        Forward pass through the model.

        Parameters
        ----------
        query_text : torch.Tensor
            A tensor of shape (batch_size, max_query_word) containing tokenized query input.
        lm : nn.Module or None
            If provided, replaces the default node_encoder LSTM with the given language model.

        Returns
        -------
        (list, list)
            instructions : list of torch.Tensor
                A list of length num_ins, where each element is (batch_size, entity_dim) representing
                the relational instructions at each step.
            attn_list : list of torch.Tensor
                A list of length num_ins, where each element is (batch_size, max_query_word, 1)
                representing the attention weights over the query at each step.

        Notes
        -----
        - Initializes reasoning state.
        - Iteratively calls get_instruction to refine the relational instruction.
        - Returns all intermediate instructions and attention distributions.
        """
        if lm is not None:
            self.node_encoder = lm
        self.init_reason(query_text)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list
