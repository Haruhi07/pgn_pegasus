import torch
from torch import nn
import torch.nn.functional as F
from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusPreTrainedModel, PegasusModel
from transformers.models.pegasus.modeling_pegasus import shift_tokens_right


class PointerPegasus(PegasusPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        r"embed_positions.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]
    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        self.model = PegasusModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        Pointer_input_size = 3 * config.d_model
        self.Pointer = nn.Linear(Pointer_input_size, 1, bias=True)
        self.activation = nn.Sigmoid()

    def forward(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # average all the 16 heads
        # cross_attention shape: (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        last_cross_attention_mean = torch.mean(outputs["cross_attentions"][-1], dim=1)
        # encoder_last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        encoder_last_hidden_state = outputs["encoder_last_hidden_state"]
        # decoder_hidden_states shape: (num_layers+1, batch_size, sequence_length, hidden_size)
        decoder_last_hidden_states = outputs["decoder_hidden_states"][-1]
        decoder_embedding = outputs["decoder_hidden_states"][0]
        # logits shape: (batch_size, sequence_length, config.vocab_size)
        logits = outputs["logits"]
        # h shape: (batch_size, sequence_length, hidden_size)
        h = torch.matmul(last_cross_attention_mean, encoder_last_hidden_state)
        print(h.shape)

        linear_input = torch.cat((h, decoder_last_hidden_states, decoder_embedding), dim=1)
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
        for t in range(len(decoder_last_hidden_states)):
            for count_idx, idx in enumerate(input_ids):
                copy_logits[t][idx] = cross_attention_mean[t][count_idx]
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