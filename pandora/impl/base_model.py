#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseModel(object):
    """
    Abstract class of the model.

    Parameters
    ===========
    token_len : int
        Character-length to which all tokens have been uniformized
        (through truncation or padding with zeros).
    token_char_vector_dict : dict
        The lookup dict used for the (one-hot) indexation of token characters.
    lemma_len : int
        Character-length to which all lemmas
        have been uniformized (through truncation or padding with zeros).
    lemma_char_vector_dict : dict
        The lookup dict used for the (one-hot) indexation of lemma characters.
    nb_encoding_layers : int
        The number of encoding layers, e.g. for the encoding LSTM.
    nb_dense_dims : int
        The dimensionality of the hidden layers; both for the recurrent layers
        and the latent representation.
    nb_tags : int
        Number of distinct pos tags in classification.
    nb_morph_cats : int
        Number of distinct morphological tags in classifcation.
    nb_lemmas : int
        Number of distinct lemma tags in classification.
    nb_train_tokens : int
        Number of distinct input tokens.
    include_token : bool
        Whether to include embedding representation of original input tokens.
    include_context : bool
        Whether to include an embedding representation
        of context tokens surrounding the original focus token.
    include_lemma : bool
        Whether to include a prediction of the lemma.
    include_pos : bool
        Whether to include a prediction of the pos tag.
    include_morph : bool
        Whether to include a prediction of morphological tags.
    nb_context_tokens : int
        Total number of context tokens (left + right) which
        will be included in the token representations.
    nb_embedding_dims : int
        Dimensionality of the embedding vectors used for the
        representation of context tokens.
    char_embed_dim : int
        Dimensionality of the embedding vectors used for the
        representation of focus characters.
    pretrained_embeddings : array-like
        If not None, this is place to pass pretrained embedding
        vectors as a numpy-array to the model, with a shape
        corresponding to: (nb_tokens, nb_embedding_dims).
    nb_filters : int
        Size of the convolutional, ngram-like filters to apply when
        `focus_repr` is 'convolutional'. The stride is always set to 1.
    focus_repr = str ('convolutional' or 'recurrent')
        Whether to use a 'convolutional' token representation
        or a 'recurrent' representation using a bidirectional LSTM.
    dropout_level : float (default = .15)
        A float between 0 and 1 expressing the dropout value used during
        training. Affects various layers: consult the model definition.
    load : bool (default = False)
        If False, a fresh model will be initialized. Else, `load`
        must be a string pointing to an existing model directory
        from which an existing model will be re-loaded.
    """

    def print_summary(self):
        """
        Print a summary of the model layer by layer, and recount number
        of parameters, non-trainable parameters and total number of parameters
        """
        raise NotImplementedError

    def adjust_lr(self, adjust_rate=0.5):
        """
        Halves the current learning rate of the model.
        """
        raise NotImplementedError

    def epoch(self, train_in, train_out):
        # TODO: according docstring train_in is a dict with any combination
        # of keys in 'X_focus', 'X_context', 'X_lemma', but it seems it is
        # rather 'focus_in', 'context_in', 'lemma_in'
        """
        Runs the model for one epoch on the data.

        Parameters
        ===========
        train_in : dict
            A dictionary with input data that has at least
            one of the following keys:
                * 'X_focus': focus token representation
                    at the character-level. This should hold a
                    numpy array (np.int32) of shape:
                    (nb_samples, token_len) where each token is
                    represented by integer indices corresponding
                    to the original characters (padded to a uniform length).
                * 'X_contexts': representation of the token-level
                    context (right and left concatenated). This should
                    hold a numpy matrix (np.int32) of shape:
                    (nb_samples, nb left_context words + nb right context words).
                    Each instance is represented by integer indices
                    corresponding to the original tokens
                    (padded to a uniform length if necessary).

        train_out : dict
            A dictionary with output data that has at least
            one of the following keys:
                * 'X_lemma': representation of the lemma.
                  * If include_lemma is 'generate', this representation
                    will be at the character-level and the value
                    should be a matrix of shape:
                    (nb_samples, lemma_len, nb_characters). Each target
                    should thus be a binary matrix representation
                    of the output lemma.
                  * If include_lemma is 'label', the target lemmas
                    are represented as atomic labels using a binary
                    matrix of shape: (nb_samples, nb_lemmas).
                * 'X_pos': representation of the pos tags
                  in a binary matrix of shape (nb_samples, nb_tags).
                * 'X_morph': representation of the morph tags
                  in a binary matrix of shape (nb_samples, nb_tags).
        """
        raise NotImplementedError

    def predict(self, input_data, batch_size=None):
        """
        Returns predictions for the data in `input_data`:

        Parameters
        ===========
        input_data : dict
            A dictionary with input data that has at least
            one of the following keys:
                * 'X_focus': focus token representation
                    at the character-level. This should hold a
                    numpy array (np.int32) of shape:
                    (nb_samples, token_len) where each token is
                    represented by integer indices corresponding
                    to the original characters (padded to a uniform length).
                * 'X_contexts': representation of the token-level
                    context (right and left concatenated). This should
                    hold a numpy matrix (np.int32) of shape:
                    (nb_samples, nb left_context words + nb right context words).
                    Each instance is represented by integer indices
                    corresponding to the original tokens
                    (padded to a uniform length if necessary).
        batch_size : int or None, optional

        Returns
        ===========
        A dictionary mapping any combination of 'lemma_out', 'pos_out' and
        'morph_out' to their respective np.arrays predictions for the
        corresponding input_data. The shape of the predictions should
        corresponding to the `train_out` data that goes into `self.epoch()`.
        """
        raise NotImplementedError

    @staticmethod
    def load(model_dir, **kwargs):
        """
        Load model saved as per `self.save`
        """
        raise NotImplementedError

    def save(self, model_dir):
        """
        Serialize the model
        """
        raise NotImplementedError
