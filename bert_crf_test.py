from transformers import BertTokenizerFast
from models import BertCRF

model = BertCRF.from_pretrained('./results/checkpoint-xxxx', num_labels=9)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
id2label = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def decode(label_ids, input_ids, offsets_mapping, id2label):
    print(label_ids)
    print(input_ids)
    print(offsets_mapping)
    print(id2label)
    result = []
    for k in range(len(label_ids)):
        words = []
        labels = []
        for i in range(len(label_ids[k])):
            start_ind, end_ind = offset_mapping[k][i]
            word = tokenizer.convert_ids_to_tokens([int(input_ids[k][i])])[0]
            is_subword = end_ind - start_ind != len(word)
            if is_subword:
                if word.startswith('##'):
                    words[-1] += word[2:]
            else:
                words.append(word)
                labels.append(id2label[int(label_ids[k][i])])
        result.append(
            {'words': words,
             'labels': labels}
        )
    return result


corpus = [
    'The european commission have reached a investment deal with China government.',
    'The Arctic are very vulnerable to the effects of Climate Change and Global Warming.'
]

inputs = tokenizer(corpus, max_length=512, padding=True, truncation=True, return_tensors='pt',
                   return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping").cpu().numpy().tolist()

outputs = model(**inputs)
print(decode(outputs.logits.numpy().tolist(), inputs['input_ids'].numpy().tolist(), offset_mapping, id2label))
