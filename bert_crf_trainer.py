from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from models import BertCRF

train_dataset, test_dataset = load_dataset('conll2003', split=['train', 'test'])
print(train_dataset, test_dataset)

model = BertCRF.from_pretrained('bert-base-cased', num_labels=9)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


def tokenize(batch):
    result = {
        'label_ids': [],
        'input_ids': [],
        'token_type_ids': [],
    }
    max_length = tokenizer.max_model_input_sizes['bert-base-cased']

    for tokens, label in zip(batch['tokens'], batch['label_ids']):
        tokenids = tokenizer(tokens, add_special_tokens=False)

        token_ids = []
        label_ids = []
        for ids, lab in zip(tokenids['input_ids'], label):
            if len(ids) > 1 and lab % 2 == 1:
                token_ids.extend(ids)
                chunk = [lab + 1] * len(ids)
                chunk[0] = lab
                label_ids.extend(chunk)
            else:
                token_ids.extend(ids)
                chunk = [lab] * len(ids)
                label_ids.extend(chunk)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        label_ids.insert(0, 0)
        label_ids.append(0)
        result['input_ids'].append(token_ids)
        result['label_ids'].append(label_ids)
        result['token_type_ids'].append(token_type_ids)

    result = tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)
    for i in range(len(result['input_ids'])):
        diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
        result['label_ids'][i] += [0] * diff
    return result


train_dataset = train_dataset.remove_columns(['id', 'pos_tags', 'chunk_tags'])
train_dataset = train_dataset.rename_column('ner_tags', 'label_ids')
test_dataset = test_dataset.remove_columns(['id', 'pos_tags', 'chunk_tags'])
test_dataset = test_dataset.rename_column('ner_tags', 'label_ids')

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])
test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])


def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    f1 = f1_score(labels, preds, average='macro')
    print(classification_report(labels, preds))
    return {
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    save_strategy=IntervalStrategy.EPOCH,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

print(trainer.evaluate())
