import torch
from torch import nn
import torch.nn.functional as F
from transformers import PegasusConfig, PegasusForConditionalGeneration


class PointerPegasus(nn.Module):
    def __init__(self, checkpoint, tokenizer, device):
        super(PointerPegasus, self).__init__()
        self.device = device
        self.configuration = PegasusConfig.from_pretrained(checkpoint,
                                                           output_hidden_states=True,
                                                           output_attentions=True,
                                                           return_dict=True)
        self.pegasus = PegasusForConditionalGeneration.from_pretrained(checkpoint,
                                                                       config=self.configuration).to(device)
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab_size ", self.vocab_size)
        print("self.hidden_size ", self.configuration.hidden_size)
        Pointer_input_size = 3 * self.configuration.hidden_size
        self.Pointer = nn.Linear(Pointer_input_size, 1, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None):
        output = self.pegasus(input_ids=input_ids,
                              attention_mask=attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask,
                              return_dict=True)
        # print(len(output["cross_attentions"]))
        # average all the 16 heads
        cross_attention_sum = torch.sum(output["cross_attentions"][-1].squeeze(0), dim=0) / 16
        encoder_last_hidden_state = output["encoder_last_hidden_state"].squeeze(0)
        decoder_hidden_states = output["decoder_hidden_states"][-1].squeeze(0)
        decoder_embedding = output["decoder_hidden_states"][0].squeeze(0)
        logits = output["logits"].squeeze(0)
        # print("logits ", logits.shape)
        # print("cross attention: ", cross_attention_sum.shape)
        # print("encoder_last_hidden_state: ", encoder_last_hidden_state.shape)
        h = torch.matmul(cross_attention_sum, encoder_last_hidden_state)
        # print(h.shape)

        linear_input = torch.cat((h, decoder_hidden_states, decoder_embedding), dim=1)
        # print("linear_input size: ", linear_input.size())
        linear_output = self.Pointer(linear_input)
        # print("linear_output size: ", linear_output.size())
        gen_probs = self.activation(linear_output)
        # print("gen_probs: ", gen_probs.size())
        distribution = F.softmax(logits, dim=1)
        # print("distribution: ", distribution.size())
        gen_logits = torch.mul(gen_probs, distribution)
        # print("gen_logits: ", gen_logits.size())
        copy_logits = torch.zeros(gen_logits.size()).to(self.device)
        for t in range(len(decoder_hidden_states)):
            for count_idx, idx in enumerate(input_ids):
                copy_logits[t][idx] = cross_attention_sum[t][count_idx]
        copy_logits = (1 - gen_probs) * copy_logits
        # print("copy logits: ", copy_logits.size())
        final_probs = torch.add(gen_logits, copy_logits)
        # print("final_probs: ", final_probs)
        # print(torch.max(final_probs, 1))
        return gen_probs, final_probs

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128):
        # beam search only for the final large model
        with torch.no_grad():
            st = torch.tensor([0]).to(self.device)
            output_ids = torch.tensor([[0]]).to(self.device)
            for t in range(1, max_new_tokens):
                _, final_probs = self.forward(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              decoder_input_ids=output_ids)
                # greedy decoding here
                output_probs, output_ids = torch.max(final_probs, 1)
                # print(output_ids)
                output_ids_list = output_ids.cpu().tolist()
                output_ids = torch.tensor([[st] + output_ids_list]).to(self.device)
                if output_ids_list[-1] == 1:
                    break
                # print(output_ids)
        return output_ids