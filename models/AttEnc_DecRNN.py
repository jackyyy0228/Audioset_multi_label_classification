import torch.nn as nn
import torch.nn.functional as F
import torch

class AttEnc_DecRNN(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=torch.softmax):
        super(AttEnc_DecRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, candidates = None, logistic_joint_decoding = False):
        
        encoder_hidden, output, contexts = self.encoder(input_variable)
        
        if logistic_joint_decoding:
            logit_output = torch.cat((output,torch.ones((output.shape[0],2), dtype=torch.float32).to(output.device) * 0.5), dim = 1)
        else:
            logit_output = None

        if self.decoder.n_layers > 1:
            _, batch_size, hidden_size = encoder_hidden[0].shape
            zero = torch.zeros((self.decoder.n_layers - 1, batch_size, hidden_size),dtype = torch.float32).to(encoder_hidden[0].device)
            encoder_hidden =  tuple([torch.cat((h,zero.clone()), dim = 0) for h in encoder_hidden])

        decoder_outputs, decoder_hidden, ret_dict = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden, encoder_outputs = contexts,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              candidates = candidates, logit_output = logit_output)

        return decoder_outputs, decoder_hidden, ret_dict, output

