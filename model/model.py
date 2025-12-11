from transformers import AutoModelForCausalLM, AutoTokenizer
import torch





class NSLM(torch.nn.Module):

    def __init__(self, model_name: str):
        super(NSLM, self).__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.processor_class = AutoTokenizer.from_pretrained(model_name)
        if self.processor_class.pad_token is None:
            self.processor_class.pad_token = self.processor_class.eos_token

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=50, num_return_sequences=1):
        outputs = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
        )
        return outputs