import json
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer

with open('infer_generation.json', 'r') as file:
    generated_data = json.load(file)

with open('Flickr30K_EE_test.json', 'r') as file:
    original_data = json.load(file)

gen = {}
gts = {}

generated_dict = {}
for generated in generated_data:
    sample_id = generated['sample_id']
    generated_dict[sample_id] = generated['GT-Cap']

for original in original_data:
    sample_id = original['sample_id']
    if sample_id in generated_dict:
        gts[sample_id] = [original['GT-Cap']]
        gen[sample_id] = [generated_dict[sample_id]]

gts_t = PTBTokenizer.tokenize(gts)
gen_t = PTBTokenizer.tokenize(gen)

val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
val_cider, _ = Cider().compute_score(gts_t, gen_t)
val_spice, _ = Spice().compute_score(gts_t, gen_t)

method = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
for metric, score in zip(method, val_bleu):
    print(f'{metric}: {score}')

print(f'METEOR: {val_meteor}')
print(f'ROUGE_L: {val_rouge}')
print(f'CIDEr: {val_cider}')
print(f'SPICE: {val_spice}')
