import datasets
from model.model import NSLM


model = NSLM(model_name="google/long-t5-tglobal-base", num_ent_labels=20, num_rel_labels=30)

dataset = datasets.load_dataset("json", data_files="dataset.jsonl")

