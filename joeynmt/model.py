# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np
from torch import cat
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch
from joeynmt.helpers import ConfigurationError


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 factor_vocab: Vocabulary,
                 method=None,
                 factor_embed: Embeddings = None
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.factor_embed = factor_embed
        self.method = method
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.factor_vocab = factor_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None, factor: Tensor = None) -> (
            Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        :param factor: factor input
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask,
                                                     factor=factor)

        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor, factor: Tensor = None) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        :param factor:
        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        if factor is not None:
            if self.method == 'add':
                emb = self.summed(self.src_embed(src), self.factor_embed(factor))
            elif self.method == 'concatenate':
                emb = self.concatenated(self.src_embed(src), self.factor_embed(factor))
        else:
            emb = self.src_embed(src)

        return self.encoder(emb, src_length, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:  # Batch has attribute factor
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """

        out, hidden, att_probs, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask, factor=batch.factor)

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def concatenated(self, src: Tensor, fac: Tensor):
        if not src.size()[0] == fac.size()[0] and src.size()[2] == fac.size()[2]:
            # check if dimensions match for concatenate
            raise ConfigurationError("Can't concatenate embeddings, dimensions do not match")
            # dimensions match if batch size and embedding size are same
        return cat((src, fac), 1)  # concatenate horizontally

    def summed(self, src: Tensor, fac: Tensor):
        # check if dimensions match for adding
        if not src.size()[1] == fac.size()[1] and src.size()[0] == fac.size()[0]:
            # match if batch size and length matches
            raise ConfigurationError("Can't sum embeddings, dimensions do not match")
        return src.add(fac)

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask, factor=batch.factor)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # greedy decoding
        if beam_size < 2:
            stacked_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output, eos_index=self.eos_index,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoder=self.decoder,
                max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                beam_search(
                    size=beam_size, encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    src_mask=batch.src_mask, embed=self.trg_embed,
                    max_output_length=max_output_length,
                    alpha=beam_alpha, eos_index=self.eos_index,
                    pad_index=self.pad_index,
                    bos_index=self.bos_index,
                    decoder=self.decoder)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                factor_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param factor_vocab:
    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]
    factor_padding_idx = None  # set default to None

    if factor_vocab is not None:  # if factor vocab exists, factors are being used
        factor_padding_idx = factor_vocab.stoi[PAD_TOKEN]
        factor_embed = Embeddings(  # create embedding for factors
            **cfg["encoder"]["factor_embeddings"], vocab_size=len(factor_vocab),
            padding_idx=factor_padding_idx)
    else:
        factor_embed = None

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab, factor_vocab=factor_vocab,
                  method=cfg["encoder"].get("factor_combine", None), factor_embed=factor_embed)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx, factor_padding_idx)

    return model
