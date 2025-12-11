from transformers import AutoModelForCausalLM, AutoTokenizer
import torch





class NSLM(torch.nn.Module):

    def __init__(self, model_name: str):
        super(NSLM, self).__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.processor_class = AutoTokenizer.from_pretrained(model_name)
        if self.processor_class.pad_token is None:
            self.processor_class.pad_token = self.processor_class.eos_token